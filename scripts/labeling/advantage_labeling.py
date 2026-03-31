"""Aggregate per-episode value predictions and derive advantage-based action quality labels."""

import logging
import pathlib

import numpy as np

from .dataset_utils import (
    read_episode_lengths,
    write_columns_to_dataset,
)

logger = logging.getLogger(__name__)


def merge_and_label(
    repo_id: str,
    values_dir: str,
    positive_fraction: float = 0.3,
    gamma: float = 0.99,
    output_dir: str | None = None,
):
    """Read saved value predictions, compute per-timestep advantages, and write is_good_action labels.

    The advantage at each timestep is computed as a discounted sum of
    baseline-subtracted rewards, where reward(t) = progress(t+1) - progress(t)
    and the baseline is 1 / mean_episode_length.
    """
    episode_lengths = read_episode_lengths(repo_id)
    values_path = pathlib.Path(values_dir)

    all_values: dict[int, np.ndarray] = {}
    for ep_idx in sorted(episode_lengths.keys()):
        fpath = values_path / f"ep_{ep_idx:06d}.npy"
        if not fpath.exists():
            raise FileNotFoundError(f"Missing value file for episode {ep_idx}: {fpath}")
        all_values[ep_idx] = np.load(fpath)

    total_frames = sum(len(v) for v in all_values.values())
    logger.info("Loaded values for %d episodes, %d frames from %s", len(all_values), total_frames, values_dir)

    # Ensure each loaded value array has the expected number of frames
    for ep_idx, vals in all_values.items():
        expected = episode_lengths[ep_idx]
        if len(vals) != expected:
            raise ValueError(f"Episode {ep_idx}: value length {len(vals)} != expected {expected}")

    # Build advantages: discounted cumulative rewards minus a per-step baseline
    mean_ep_len = np.mean(list(episode_lengths.values()))
    baseline_reward = 1.0 / mean_ep_len
    logger.info(
        "Computing advantages with gamma=%.2f, mean_ep_len=%.1f, baseline_reward=%.6f",
        gamma, mean_ep_len, baseline_reward,
    )
    all_advantages = {}
    for ep_idx, values in all_values.items():
        ep_len = len(values)
        # Per-step reward is the change in progress; zero for the final step
        rewards = np.zeros(ep_len, dtype=np.float64)
        if ep_len > 1:
            rewards[:-1] = np.diff(values)
        # Reverse sweep: accumulate discounted baseline-subtracted rewards
        advantages = np.zeros(ep_len, dtype=np.float64)
        running = 0.0
        for t in range(ep_len - 1, -1, -1):
            running = (rewards[t] - baseline_reward) + gamma * running
            advantages[t] = running
        all_advantages[ep_idx] = advantages

    # Set the advantage threshold so that the top positive_fraction of timesteps are labeled good
    all_adv_flat = np.concatenate(list(all_advantages.values()))
    np_percentile = (1.0 - positive_fraction) * 100.0
    threshold = np.percentile(all_adv_flat, np_percentile)
    logger.info(
        "Advantage stats: mean=%.4f, std=%.4f, threshold(p%.0f, positive_fraction=%.2f)=%.4f",
        all_adv_flat.mean(), all_adv_flat.std(), np_percentile, positive_fraction, threshold,
    )

    # Apply the threshold to produce binary labels
    is_good_action = {}
    n_good = 0
    n_total = 0
    for ep_idx, adv in all_advantages.items():
        labels = (adv > threshold).astype(np.int64)
        is_good_action[ep_idx] = labels
        n_good += labels.sum()
        n_total += len(labels)
    logger.info("is_good_action: %d/%d = %.1f%% positive", n_good, n_total, 100 * n_good / n_total)

    # Prepare float32 columns for writing to parquet
    predicted_value = {ep: vals.astype(np.float32) for ep, vals in all_values.items()}
    advantage = {ep: adv.astype(np.float32) for ep, adv in all_advantages.items()}

    write_columns_to_dataset(
        repo_id,
        columns={
            "is_good_action": is_good_action,
            "predicted_value": predicted_value,
            "advantage": advantage,
        },
        column_meta={
            "is_good_action": {"dtype": "int64", "shape": [1], "names": ["is_good_action"]},
            "predicted_value": {"dtype": "float32", "shape": [1], "names": ["predicted_value"]},
            "advantage": {"dtype": "float32", "shape": [1], "names": ["advantage"]},
        },
        output_dir=output_dir,
    )


def _run_vf_merge(args):
    """Entry point for the vf_merge CLI subcommand."""
    merge_and_label(
        repo_id=args.repo_id,
        values_dir=args.values_dir,
        positive_fraction=args.positive_fraction,
        gamma=args.gamma,
        output_dir=args.output_dir,
    )

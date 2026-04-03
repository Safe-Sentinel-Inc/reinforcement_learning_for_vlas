"""Assign binned progress values and intervention flags to dataset episodes."""

import dataclasses
import importlib.util
import json
import logging
import pathlib
import sys

import numpy as np

from .dataset_utils import (
    compute_fold_assignments,
    read_episode_lengths,
    resolve_success_episodes,
    write_columns_to_dataset,
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class LabelConfig:
    """Metadata describing labeling parameters for a dataset.

    Loaded from a user-provided ``config.py`` file that declares episode-level
    success/failure annotations and other labeling settings.
    """

    task_name: str
    # Labeling metadata
    success_episodes: str | list[int] = "all"
    failed_episodes: tuple[int, ...] = ()
    all_human: bool = False
    intervention_episodes: dict = dataclasses.field(default_factory=dict)
    stage_boundaries: tuple[int, ...] = ()


def load_label_config(config_path: str | pathlib.Path) -> LabelConfig:
    """Import a Python config file and return its labeling parameters as a LabelConfig."""
    config_path = pathlib.Path(config_path)
    if not config_path.is_file():
        config_path = config_path / "config.py"
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    spec = importlib.util.spec_from_file_location("_label_cfg", config_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_label_cfg"] = mod
    spec.loader.exec_module(mod)
    return LabelConfig(
        task_name=mod.TASK_NAME,
        success_episodes=getattr(mod, "SUCCESS_EPISODES", "all"),
        failed_episodes=tuple(getattr(mod, "FAILED_EPISODES", ())),
        all_human=getattr(mod, "ALL_HUMAN", False),
        intervention_episodes=dict(getattr(mod, "INTERVENTION_EPISODES", {})),
        stage_boundaries=tuple(getattr(mod, "STAGE_BOUNDARIES", ())),
    )


def compute_binned_value_progress(
    episode_lengths: dict[int, int],
    success_episodes: set[int],
    num_bins: int = 200,
    return_min: float = 0.0,
    return_max: float = 1.0,
) -> dict[int, np.ndarray]:
    """Map each timestep to a discretized progress bin.

    For successful episodes, progress scales linearly from 0 to 1 over the full length.
    For failed episodes, progress follows the longest episode's pace and the
    final 200 steps are linearly decayed to zero.
    """
    bin_edges = np.linspace(return_min, return_max, num_bins + 1)
    t_max = max(episode_lengths.values())

    result = {}
    for ep_idx, ep_len in episode_lengths.items():
        if ep_idx in success_episodes:
            # Successful episode: uniform progress ramp from start to finish
            progress = np.linspace(0, 1, ep_len)
        else:
            # Failed episode: progress at the rate of the longest episode
            full_progress = np.linspace(0, 1, t_max)
            progress = full_progress[:ep_len].copy()
            # Taper the tail of a failed episode down to zero progress
            decay_len = min(200, ep_len)
            if decay_len > 0:
                decay_start = ep_len - decay_len
                decay_factors = np.linspace(1, 0, decay_len)
                progress[decay_start:] *= decay_factors

        bins = np.digitize(progress, bin_edges) - 1
        bins = np.clip(bins, 0, num_bins - 1)
        result[ep_idx] = bins

    return result


def compute_intervention_labels(
    episode_lengths: dict[int, int],
    intervention_ranges: dict[int, list[list[int]]],
) -> dict[int, np.ndarray]:
    """Generate binary per-timestep flags marking human-corrected intervals.

    Timesteps within intervention ranges are always treated as demonstrations,
    independent of any advantage-based filtering.
    """
    result = {}
    for ep_idx, ep_len in episode_lengths.items():
        labels = np.zeros(ep_len, dtype=np.bool_)
        if ep_idx in intervention_ranges:
            for start, end in intervention_ranges[ep_idx]:
                labels[start:end] = True
        result[ep_idx] = labels
    return result


def _run_add_labels(args):
    """Entry point for the add_labels CLI subcommand."""

    success_spec: str | list[int] = "all"
    failed_list: list[int] = []
    all_human: bool = False
    intervention_ranges: dict[int, list[list[int]]] = {}
    stage_boundaries: list[int] = []

    cfg = load_label_config(args.config)
    repo_id = args.repo_id if args.repo_id else cfg.task_name
    success_spec = cfg.success_episodes
    failed_list = list(cfg.failed_episodes)
    all_human = cfg.all_human
    intervention_ranges = {int(k): v for k, v in cfg.intervention_episodes.items()}

    # Stage boundaries define where multi-phase tasks transition
    stage_boundaries = cfg.stage_boundaries
    logger.info("Loaded config from %s (task=%s)", args.config, cfg.task_name)
    logger.info("Stage boundaries: %s", stage_boundaries)

    # Fetch episode lengths from metadata without loading the full dataset
    episode_lengths = read_episode_lengths(repo_id)
    total_frames = sum(episode_lengths.values())
    logger.info("Found %d episodes, %d frames", len(episode_lengths), total_frames)

    # Determine which episodes count as successful
    all_ep_indices = set(episode_lengths.keys())
    success_set = resolve_success_episodes(success_spec, failed_list, all_ep_indices)

    # Ensure all specified success episodes actually exist in the dataset
    invalid = success_set - all_ep_indices
    if invalid:
        raise SystemExit(
            f"These success episodes don't exist in dataset: {sorted(invalid)}"
        )

    logger.info("Success episodes: %d / %d", len(success_set), len(episode_lengths))
    failed_set = all_ep_indices - success_set
    if failed_set:
        logger.info("Failed episodes: %s", sorted(failed_set))

    # Compute discretized progress values for each timestep
    binned = compute_binned_value_progress(
        episode_lengths,
        success_set,
        num_bins=args.num_bins,
        return_min=args.return_min,
        return_max=args.return_max,
    )
    binned = {k: v.astype(np.int64) for k, v in binned.items()}

    # Build per-timestep intervention flags
    if all_human:
        logger.info("all_human=True: marking ALL timesteps as human demonstrations")
        intervention = {
            ep: np.ones(ep_len, dtype=np.int64)
            for ep, ep_len in episode_lengths.items()
        }
    else:
        if intervention_ranges:
            logger.info("Intervention episodes: %s", sorted(intervention_ranges.keys()))
        intervention = compute_intervention_labels(episode_lengths, intervention_ranges)
        intervention = {k: v.astype(np.int64) for k, v in intervention.items()}

    # Assign a stage index to each timestep based on boundary positions
    stage_labels = {}
    if stage_boundaries:
        logger.info("Computing stage labels with boundaries: %s", stage_boundaries)
        for ep_idx, ep_len in episode_lengths.items():
            stages = np.ones(ep_len, dtype=np.int64)

            for stage_idx, boundary in enumerate(stage_boundaries):
                if boundary < ep_len:
                    stages[boundary:] = stage_idx + 2
                else:
                    break

            stage_labels[ep_idx] = stages

        all_stages = np.concatenate([stage_labels[ep] for ep in sorted(stage_labels)])
        unique, counts = np.unique(all_stages, return_counts=True)
        for stage, count in zip(unique, counts):
            logger.info("Stage %d: %d frames (%.1f%%)",
                       stage, count, 100*count/len(all_stages))
    else:
        logger.info("No stage boundaries specified, using default (all frames in stage 1)")
        for ep_idx, ep_len in episode_lengths.items():
            stage_labels[ep_idx] = np.ones(ep_len, dtype=np.int64)

    # Log summary statistics before writing
    all_bins = np.concatenate([binned[ep] for ep in sorted(binned)])
    logger.info("binned_value range: [%d, %d], mean: %.2f", all_bins.min(), all_bins.max(), all_bins.mean())
    all_intv = np.concatenate([intervention[ep] for ep in sorted(intervention)])
    logger.info("intervention: %d/%d timesteps marked", all_intv.sum(), len(all_intv))

    write_columns_to_dataset(
        repo_id,
        columns={
            "binned_value": binned,
            "intervention": intervention,
            "stage": stage_labels
        },
        column_meta={
            "binned_value": {"dtype": "int64", "shape": [1], "names": ["binned_value"]},
            "intervention": {"dtype": "int64", "shape": [1], "names": ["intervention"]},
            "stage": {"dtype": "int64", "shape": [1], "names": ["stage"]},
        },
        output_dir=args.output_dir,
        lenient=args.lenient,
    )

    # Optionally partition episodes into K cross-validation folds
    num_folds = args.num_folds
    if num_folds > 0:
        fold_assignments = compute_fold_assignments(
            sorted(episode_lengths.keys()), num_folds, seed=args.fold_seed,
        )

        # Broadcast each episode's fold ID across all its timesteps
        fold_col = {}
        for ep_idx, ep_len in episode_lengths.items():
            fold_col[ep_idx] = np.full(ep_len, fold_assignments[ep_idx], dtype=np.int64)

        write_columns_to_dataset(
            repo_id,
            columns={"fold": fold_col},
            column_meta={"fold": {"dtype": "int64", "shape": [1], "names": ["fold"]}},
            output_dir=args.output_dir,
            lenient=args.lenient,
        )

        # Persist the fold mapping as a JSON sidecar file
        from lerobot.common.constants import HF_LEROBOT_HOME
        if args.output_dir:
            ds_path = pathlib.Path(args.output_dir) / repo_id
        else:
            ds_path = HF_LEROBOT_HOME / repo_id
        folds_path = ds_path / "meta" / "folds.json"
        with open(folds_path, "w") as f:
            json.dump({str(k): v for k, v in fold_assignments.items()}, f, indent=2)

        # Report how many episodes and frames ended up in each fold
        for fold_id in range(num_folds):
            fold_eps = [ep for ep, fold in fold_assignments.items() if fold == fold_id]
            fold_frames = sum(episode_lengths[ep] for ep in fold_eps)
            logger.info(
                "Fold %d: %d episodes, %d frames", fold_id, len(fold_eps), fold_frames,
            )
        logger.info("Fold assignments written to %s", folds_path)

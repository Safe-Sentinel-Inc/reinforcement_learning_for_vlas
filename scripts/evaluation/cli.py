"""Command-line interface and core logic for offline evaluation of PI06/Recap checkpoints.

Orchestrates the full offline evaluation pipeline: argument parsing, policy loading,
advantage-conditioned inference comparison, metric aggregation, and report generation.
"""

import argparse
import dataclasses
import json
import logging
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pyarrow.parquet as pq

from openpi.models import model as _model
from openpi.models import pi0 as _pi0
from openpi.policies import policy as _policy
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms

from .data_loading import (
    CAMERA_NAMES,
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    DEFAULT_DATASET,
    _decode_image,
    _episode_path,
    _load_episode_lengths,
    _load_folds,
    _load_task_map,
    _parse_episode_spec,
    _progress,
    _read_policy_rows,
)
from .metrics import (
    _build_episode_records,
    _safe_corr,
    _sample_indices,
)
from .plotting import (
    _plot_advantage_quality,
    _plot_episode_summary,
    _plot_feature_support,
    _plot_policy_condition,
    _plot_vf_overview,
    _save_episode_csv,
)


def parse_args() -> argparse.Namespace:
    """Define and parse all command-line arguments for the offline evaluation script."""
    parser = argparse.ArgumentParser(description="Pure offline evaluation for PI06/Recap checkpoints.")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET, help="LeRobot dataset root directory.")
    parser.add_argument("--output-dir", required=True, help="Directory for offline evaluation outputs.")
    parser.add_argument("--episodes", default=None, help="Comma-separated episode indices. If omitted, sample from folds.")
    parser.add_argument(
        "--max-episodes-per-fold",
        type=int,
        default=3,
        help="When --episodes is omitted, pick up to N episodes per fold for offline evaluation.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument("--plot-max-points", type=int, default=20000, help="Maximum points for global scatter plots.")
    parser.add_argument("--policy-config-name", default=DEFAULT_CONFIG, help="Policy config name.")
    parser.add_argument("--policy-checkpoint-dir", default=DEFAULT_CHECKPOINT, help="Policy checkpoint directory.")
    parser.add_argument("--policy-max-frames-per-episode", type=int, default=6, help="Frames per episode for policy comparison.")
    parser.add_argument("--policy-num-steps", type=int, default=4, help="Diffusion steps for policy inference.")
    parser.add_argument(
        "--policy-params-dtype",
        choices=["bfloat16", "float16"],
        default="float16",
        help="Checkpoint parameter dtype used for offline policy comparison.",
    )
    parser.add_argument("--skip-policy-eval", action="store_true", help="Skip policy inference comparison.")
    return parser.parse_args()


def _create_analysis_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: Path,
    *,
    num_steps: int,
    params_dtype: str,
) -> _policy.Policy:
    """Instantiate a Policy from a saved checkpoint for offline analysis.

    Loads model parameters and normalization statistics, then builds the full
    input/output transform pipeline. Disables advantage dropout so the advantage
    conditioning token is always present during evaluation.
    """
    dtype = {"bfloat16": jnp.bfloat16, "float16": jnp.float16}[params_dtype]
    model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=dtype))
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.asset_id is None:
        raise ValueError("Asset id is required to load norm stats.")
    norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    # Ensure the advantage token is never dropped during evaluation
    inference_model_transforms = []
    for t in data_config.model_transforms.inputs:
        if isinstance(t, transforms.TokenizePrompt) and t.advantage_conditioning:
            t = dataclasses.replace(t, advantage_dropout_rate=0.0)
        inference_model_transforms.append(t)

    return _policy.Policy(
        model,
        transforms=[
            transforms.InjectDefaultPrompt(None),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *inference_model_transforms,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ],
        sample_kwargs={"num_steps": num_steps},
    )


def _prepare_observation(policy: _policy.Policy, sample: dict[str, Any]) -> tuple[dict[str, Any], _model.Observation]:
    """Apply the policy's input transforms to a sample and return a batched Observation."""
    transformed = policy._input_transform(jax.tree.map(lambda x: x, sample))  # noqa: SLF001
    batched = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], transformed)
    return batched, _model.Observation.from_dict(batched)


def _masked_mean(tokens: jax.Array, mask: jax.Array) -> np.ndarray:
    """Compute the mean of token embeddings weighted by a binary mask, ignoring padding positions."""
    weights = mask.astype(tokens.dtype)[..., None]
    denom = jnp.clip(weights.sum(axis=1), 1.0)
    pooled = (tokens * weights).sum(axis=1) / denom
    return np.asarray(jax.device_get(pooled), dtype=np.float32)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return the cosine similarity between two flat vectors, or 0.0 if either has near-zero norm."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-8:
        return 0.0
    return float(np.dot(a, b) / denom)


def _extract_prefix_features(model, observation: _model.Observation) -> dict[str, np.ndarray]:
    """Run the PaliGemma prefix encoder and return per-camera and global feature vectors.

    Produces both pre-fusion (raw embeddings) and post-fusion (after transformer layers)
    representations for each camera, plus a global mask-averaged post-fusion vector.
    """
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    prefix_tokens = jax.device_get(prefix_tokens)
    prefix_mask = jax.device_get(prefix_mask)
    prefix_ar_mask = jax.device_get(prefix_ar_mask)

    # Determine how many tokens belong to each camera vs. the language prompt
    lang_token_len = int(observation.tokenized_prompt.shape[1]) if observation.tokenized_prompt is not None else 0
    image_token_total = prefix_tokens.shape[1] - lang_token_len
    camera_token_len = image_token_total // len(CAMERA_NAMES)

    # Forward pass through the language model to produce fused representations
    prefix_attn_mask = _pi0.make_attn_mask(jnp.asarray(prefix_mask), jnp.asarray(prefix_ar_mask))
    positions = jnp.cumsum(jnp.asarray(prefix_mask), axis=1) - 1
    (prefix_out, _), _ = model.PaliGemma.llm([jnp.asarray(prefix_tokens), None], mask=prefix_attn_mask, positions=positions)
    prefix_out = np.asarray(jax.device_get(prefix_out), dtype=np.float32)

    result = {
        "global_postfusion": _masked_mean(jnp.asarray(prefix_out), jnp.asarray(prefix_mask))[0],
    }
    # Extract pre-fusion and post-fusion mean features for each camera's token span
    cursor = 0
    for camera_name in CAMERA_NAMES:
        next_cursor = cursor + camera_token_len
        result[f"camera_prefusion/{camera_name}"] = np.asarray(prefix_tokens[:, cursor:next_cursor].mean(axis=1), dtype=np.float32)[0]
        result[f"camera_postfusion/{camera_name}"] = np.asarray(prefix_out[:, cursor:next_cursor].mean(axis=1), dtype=np.float32)[0]
        cursor = next_cursor
    return result


def _sample_policy_frames(dataset_root: Path, episodes: list[int], max_frames_per_episode: int, task_map: dict[int, str]) -> list[dict[str, Any]]:
    """Select a subset of frames from each episode and load their full observation data for policy comparison."""
    sampled_rows: list[dict[str, Any]] = []
    for episode_index in episodes:
        table = pq.read_table(_episode_path(dataset_root, episode_index), columns=["frame_index"])
        total_frames = int(table.num_rows)
        frame_indices = _sample_indices(total_frames, max_frames_per_episode)
        rows = _read_policy_rows(dataset_root, episode_index, frame_indices)
        for row in rows:
            sampled_rows.append(
                {
                    "episode_index": int(row["episode_index"]),
                    "frame_index": int(row["frame_index"]),
                    "task_index": int(row["task_index"]),
                    "prompt": task_map[int(row["task_index"])],
                    "state": np.asarray(row["state"], dtype=np.float32),
                    "base_0_rgb": _decode_image(row["base_0_rgb"]),
                    "left_wrist_0_rgb": _decode_image(row["left_wrist_0_rgb"]),
                    "right_wrist_0_rgb": _decode_image(row["right_wrist_0_rgb"]),
                    "predicted_value": float(row["predicted_value"]),
                    "advantage_label": float(row["advantage"]),
                    "is_good_action": int(row["is_good_action"]),
                    "intervention": bool(row["intervention"]),
                }
            )
    sampled_rows.sort(key=lambda row: (row["episode_index"], row["frame_index"]))
    return sampled_rows


def _run_policy_comparison(
    dataset_root: Path,
    episodes: list[int],
    task_map: dict[int, str],
    config_name: str,
    checkpoint_dir: Path,
    num_steps: int,
    params_dtype: str,
    max_frames_per_episode: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run advantage-conditioned inference on sampled frames and measure behavioral differences.

    For each frame, runs the policy twice (advantage=True and advantage=False) with
    identical diffusion noise, then records action gaps, cosine similarities, and
    prefix representation differences. Returns per-frame results and per-episode aggregates.
    """
    train_config = _config.get_config(config_name)
    policy = _create_analysis_policy(train_config, checkpoint_dir, num_steps=num_steps, params_dtype=params_dtype)
    model = policy._model  # noqa: SLF001
    action_horizon = int(train_config.model.action_horizon)
    action_dim = int(train_config.model.action_dim)
    rng = np.random.default_rng(seed)

    sampled_rows = _sample_policy_frames(dataset_root, episodes, max_frames_per_episode, task_map)
    results: list[dict[str, Any]] = []
    episode_pos_actions: dict[int, list[np.ndarray]] = {}
    episode_neg_actions: dict[int, list[np.ndarray]] = {}
    episode_prefix: dict[int, list[np.ndarray]] = {}

    for idx, row in enumerate(sampled_rows):
        base_sample = {
            "state": row["state"],
            "prompt": row["prompt"],
            "base_0_rgb": row["base_0_rgb"],
            "left_wrist_0_rgb": row["left_wrist_0_rgb"],
            "right_wrist_0_rgb": row["right_wrist_0_rgb"],
        }
        # Use identical noise for both conditions so differences reflect only the advantage token
        noise = rng.normal(size=(action_horizon, action_dim)).astype(np.float32)
        pos_obs = base_sample | {"advantage": True}
        neg_obs = base_sample | {"advantage": False}

        pos_outputs = policy.infer(pos_obs, noise=noise)
        neg_outputs = policy.infer(neg_obs, noise=noise)
        pos_actions = np.asarray(pos_outputs["actions"], dtype=np.float32)
        neg_actions = np.asarray(neg_outputs["actions"], dtype=np.float32)

        _, pos_observation = _prepare_observation(policy, pos_obs)
        _, neg_observation = _prepare_observation(policy, neg_obs)
        pos_prefix = _extract_prefix_features(model, pos_observation)
        neg_prefix = _extract_prefix_features(model, neg_observation)

        # Compute per-frame metrics comparing positive vs negative conditioning
        frame_result = {
            "episode_index": row["episode_index"],
            "frame_index": row["frame_index"],
            "predicted_value": row["predicted_value"],
            "advantage_label": row["advantage_label"],
            "is_good_action": row["is_good_action"],
            "intervention": row["intervention"],
            "action_gap": float(np.linalg.norm((pos_actions - neg_actions).reshape(-1))),
            "action_cosine": _cosine_similarity(pos_actions.reshape(-1), neg_actions.reshape(-1)),
            "prefix_condition_gap": float(np.linalg.norm(pos_prefix["global_postfusion"] - neg_prefix["global_postfusion"])),
            "camera_similarity_prefusion": float(
                np.mean(
                    [
                        _cosine_similarity(pos_prefix["camera_prefusion/base_0_rgb"], pos_prefix["camera_prefusion/left_wrist_0_rgb"]),
                        _cosine_similarity(pos_prefix["camera_prefusion/base_0_rgb"], pos_prefix["camera_prefusion/right_wrist_0_rgb"]),
                        _cosine_similarity(pos_prefix["camera_prefusion/left_wrist_0_rgb"], pos_prefix["camera_prefusion/right_wrist_0_rgb"]),
                    ]
                )
            ),
            "camera_similarity_postfusion": float(
                np.mean(
                    [
                        _cosine_similarity(pos_prefix["camera_postfusion/base_0_rgb"], pos_prefix["camera_postfusion/left_wrist_0_rgb"]),
                        _cosine_similarity(pos_prefix["camera_postfusion/base_0_rgb"], pos_prefix["camera_postfusion/right_wrist_0_rgb"]),
                        _cosine_similarity(pos_prefix["camera_postfusion/left_wrist_0_rgb"], pos_prefix["camera_postfusion/right_wrist_0_rgb"]),
                    ]
                )
            ),
            "pos_actions": pos_actions,
            "neg_actions": neg_actions,
            "pos_prefix": pos_prefix["global_postfusion"],
        }
        results.append(frame_result)
        episode_pos_actions.setdefault(row["episode_index"], []).append(pos_actions.reshape(-1))
        episode_neg_actions.setdefault(row["episode_index"], []).append(neg_actions.reshape(-1))
        episode_prefix.setdefault(row["episode_index"], []).append(pos_prefix["global_postfusion"])

        if idx == 0:
            _progress(
                f"[offline] first policy comparison ready: pos={pos_actions.shape}, neg={neg_actions.shape}, "
                f"gap={frame_result['action_gap']:.3f}"
            )

    aggregates = {
        "episode_pos_actions": episode_pos_actions,
        "episode_neg_actions": episode_neg_actions,
        "episode_prefix": episode_prefix,
    }
    return results, aggregates


def main() -> None:
    """Entry point: parse arguments, run value-function analysis, optionally compare policy outputs, and save the report."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    dataset_root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset metadata: task descriptions, fold assignments, and episode lengths
    task_map = _load_task_map(dataset_root)
    folds = _load_folds(dataset_root)
    episode_lengths = _load_episode_lengths(dataset_root)
    selected_episodes = _parse_episode_spec(args.episodes, folds, args.max_episodes_per_fold, args.seed)
    selected_episodes = [ep for ep in selected_episodes if ep in episode_lengths]

    _progress(f"[offline] dataset={dataset_root}")
    _progress(f"[offline] selected episodes={selected_episodes}")

    # Build per-episode metrics and pooled aggregates, then generate diagnostic plots
    episode_records, aggregate = _build_episode_records(dataset_root, selected_episodes, folds)
    _save_episode_csv(episode_records, output_dir / "episode_summary.csv")
    _plot_vf_overview(episode_records, aggregate, output_dir / "vf_overview.png")
    _plot_advantage_quality(episode_records, aggregate, output_dir / "advantage_quality.png")
    _plot_episode_summary(episode_records, output_dir / "episode_summary.png")

    # Assemble the JSON report with aggregate statistics
    report: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "episodes": selected_episodes,
        "num_episodes": len(selected_episodes),
        "aggregate_metrics": {
            "mean_predicted_value": float(aggregate["predicted_value"].mean()),
            "mean_advantage": float(aggregate["advantage"].mean()),
            "positive_fraction": float(aggregate["is_good_action"].mean()),
            "intervention_fraction": float(aggregate["intervention"].mean()),
            "value_progress_corr": _safe_corr(aggregate["predicted_value"], aggregate["progress"]),
            "value_binned_corr": _safe_corr(aggregate["predicted_value"], aggregate["binned_progress"]),
            "advantage_delta_value_corr": _safe_corr(aggregate["local_advantage"], aggregate["local_dvalue"]),
        },
    }

    if not args.skip_policy_eval:
        _progress("[offline] starting policy condition comparison...")
        policy_results, policy_aggregates = _run_policy_comparison(
            dataset_root,
            selected_episodes,
            task_map,
            args.policy_config_name,
            Path(args.policy_checkpoint_dir).resolve(),
            args.policy_num_steps,
            args.policy_params_dtype,
            args.policy_max_frames_per_episode,
            args.seed,
        )
        _plot_feature_support(policy_results, policy_aggregates, output_dir / "feature_support.png")
        report["policy_metrics"] = _plot_policy_condition(policy_results, policy_aggregates, output_dir / "policy_condition.png")
        report["policy_samples"] = len(policy_results)
    else:
        _progress("[offline] skip policy evaluation.")

    (output_dir / "offline_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _progress(f"[offline] saved report to {output_dir}")


if __name__ == "__main__":
    main()

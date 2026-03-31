import argparse
import dataclasses
import io
import json
import logging
import os
from pathlib import Path
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from openpi.models import model as _model
from openpi.models import pi0 as _pi0
from openpi.policies import policy as _policy
from openpi.training import config as _config
from openpi.training import checkpoints as _checkpoints
import openpi.transforms as transforms


CAMERA_NAMES = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
DEFAULT_CONFIG = "pi06_rl_pretrain_airbot_clothes_folding"
DEFAULT_CHECKPOINT = "checkpoints/pi06_rl_pretrain_airbot_clothes_folding/pi06_fold_clothes_v2/96000"
DEFAULT_OUTPUT = "assets/model_features/pi06_fold_clothes_v2_ep0_single"


def _progress(message: str) -> None:
    logging.info(message)
    print(message, flush=True)


def _format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rem = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m{rem:04.1f}s"
    hours, rem = divmod(minutes, 60)
    return f"{int(hours)}h{int(rem)}m"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract single-episode PI06 VLA features.")
    parser.add_argument("--config-name", default=DEFAULT_CONFIG, help="Training config name.")
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT, help="Checkpoint directory.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT, help="Output directory.")
    parser.add_argument("--lerobot-root", default="./lerobot_data", help="HF_LEROBOT_HOME root directory.")
    parser.add_argument("--episode-index", type=int, default=0, help="Episode index to analyze.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for batched inference.")
    parser.add_argument("--num-steps", type=int, default=10, help="Diffusion steps for policy inference.")
    parser.add_argument("--max-frames", type=int, default=None, help="Only process the first N frames of the episode.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for action sampling.")
    parser.add_argument("--log-every", type=int, default=256, help="Log progress every N frames.")
    parser.add_argument(
        "--params-dtype",
        choices=["bfloat16", "float16"],
        default="float16",
        help="Checkpoint parameter dtype for analysis. float16 reduces memory usage.",
    )
    return parser.parse_args()


def _load_task_map(dataset_root: Path) -> dict[int, str]:
    tasks_path = dataset_root / "meta" / "tasks.jsonl"
    with tasks_path.open("r", encoding="utf-8") as f:
        return {int(row["task_index"]): row["task"] for row in map(json.loads, f) if row}


def _decode_image(cell: dict[str, Any]) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(cell["bytes"])))


def _row_to_sample(row: dict[str, Any], task_map: dict[int, str]) -> dict[str, Any]:
    sample = {
        "base_0_rgb": _decode_image(row["base_0_rgb"]),
        "left_wrist_0_rgb": _decode_image(row["left_wrist_0_rgb"]),
        "right_wrist_0_rgb": _decode_image(row["right_wrist_0_rgb"]),
        "state": np.asarray(row["state"], dtype=np.float32),
        "prompt": task_map[int(row["task_index"])],
        "index": int(row["index"]),
        "episode_index": int(row["episode_index"]),
        "frame_index": int(row["frame_index"]),
        "task_index": int(row["task_index"]),
        "is_good_action": int(row["is_good_action"]) if row.get("is_good_action") is not None else None,
        "intervention": bool(row["intervention"]) if row.get("intervention") is not None else None,
        "binned_value": int(row["binned_value"]) if row.get("binned_value") is not None else None,
        "advantage": float(row["advantage"]) if row.get("advantage") is not None else None,
    }
    if row.get("predicted_value") is not None:
        sample["predicted_value"] = float(row["predicted_value"])
    return sample


def _load_episode_samples(dataset_root: Path, episode_index: int) -> list[dict[str, Any]]:
    parquet_path = dataset_root / "data" / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}.parquet"
    task_map = _load_task_map(dataset_root)
    rows = pq.read_table(parquet_path).to_pylist()
    return [_row_to_sample(row, task_map) for row in rows]


def _prepare_batch(policy, raw_samples: list[dict[str, Any]]) -> tuple[dict[str, Any], _model.Observation]:
    transformed_samples = [policy._input_transform(jax.tree.map(lambda x: x, sample)) for sample in raw_samples]  # noqa: SLF001
    batched = jax.tree.map(lambda *xs: jnp.asarray(np.stack(xs, axis=0)), *transformed_samples)
    return batched, _model.Observation.from_dict(batched)


def _masked_mean(tokens: jax.Array, mask: jax.Array) -> np.ndarray:
    weights = mask.astype(tokens.dtype)[..., None]
    denom = jnp.clip(weights.sum(axis=1), 1.0)
    pooled = (tokens * weights).sum(axis=1) / denom
    return np.asarray(jax.device_get(pooled), dtype=np.float32)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    denom = np.clip(denom, 1e-8, None)
    return np.sum(a * b, axis=-1) / denom


def _extract_prefix_features(model, observation: _model.Observation) -> dict[str, np.ndarray]:
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    prefix_tokens = jax.device_get(prefix_tokens)
    prefix_mask = jax.device_get(prefix_mask)
    prefix_ar_mask = jax.device_get(prefix_ar_mask)

    lang_token_len = int(observation.tokenized_prompt.shape[1]) if observation.tokenized_prompt is not None else 0
    image_token_total = prefix_tokens.shape[1] - lang_token_len
    camera_token_len = image_token_total // len(CAMERA_NAMES)

    camera_prefusion: dict[str, np.ndarray] = {}
    camera_postfusion: dict[str, np.ndarray] = {}

    prefix_attn_mask = _pi0.make_attn_mask(jnp.asarray(prefix_mask), jnp.asarray(prefix_ar_mask))
    positions = jnp.cumsum(jnp.asarray(prefix_mask), axis=1) - 1
    (prefix_out, _), _ = model.PaliGemma.llm([jnp.asarray(prefix_tokens), None], mask=prefix_attn_mask, positions=positions)
    prefix_out = np.asarray(jax.device_get(prefix_out), dtype=np.float32)

    cursor = 0
    for camera_name in CAMERA_NAMES:
        next_cursor = cursor + camera_token_len
        camera_prefusion[camera_name] = np.asarray(prefix_tokens[:, cursor:next_cursor].mean(axis=1), dtype=np.float32)
        camera_postfusion[camera_name] = np.asarray(prefix_out[:, cursor:next_cursor].mean(axis=1), dtype=np.float32)
        cursor = next_cursor

    global_prefusion = _masked_mean(jnp.asarray(prefix_tokens), jnp.asarray(prefix_mask))
    global_postfusion = _masked_mean(jnp.asarray(prefix_out), jnp.asarray(prefix_mask))

    return {
        "global_prefusion": global_prefusion,
        "global_postfusion": global_postfusion,
        "camera_prefusion/base_0_rgb": camera_prefusion["base_0_rgb"],
        "camera_prefusion/left_wrist_0_rgb": camera_prefusion["left_wrist_0_rgb"],
        "camera_prefusion/right_wrist_0_rgb": camera_prefusion["right_wrist_0_rgb"],
        "camera_postfusion/base_0_rgb": camera_postfusion["base_0_rgb"],
        "camera_postfusion/left_wrist_0_rgb": camera_postfusion["left_wrist_0_rgb"],
        "camera_postfusion/right_wrist_0_rgb": camera_postfusion["right_wrist_0_rgb"],
    }


def _run_policy_batch(policy, observation: _model.Observation, transformed_batch: dict[str, Any]) -> np.ndarray:
    policy._rng, sample_rng = jax.random.split(policy._rng)  # noqa: SLF001
    outputs = {
        "state": transformed_batch["state"],
        "actions": policy._sample_actions(sample_rng, observation, **policy._sample_kwargs),  # noqa: SLF001
    }
    outputs = jax.tree.map(np.asarray, outputs)
    outputs = policy._output_transform(outputs)  # noqa: SLF001
    return np.asarray(outputs["actions"], dtype=np.float32)


def _create_analysis_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: Path,
    *,
    num_steps: int,
    params_dtype: str,
) -> _policy.Policy:
    dtype = {"bfloat16": jnp.bfloat16, "float16": jnp.float16}[params_dtype]
    model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=dtype))
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.asset_id is None:
        raise ValueError("Asset id is required to load norm stats.")
    norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

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


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_start = time.perf_counter()

    os.environ["HF_LEROBOT_HOME"] = str(Path(args.lerobot_root).resolve())
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_config = _config.get_config(args.config_name)
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    dataset_root = Path(os.environ["HF_LEROBOT_HOME"]) / train_config.data.repo_id

    _progress(f"[extract] checkpoint={checkpoint_dir}")
    _progress(f"[extract] dataset={dataset_root}, episode={args.episode_index}")
    _progress(
        f"[extract] jax devices={[str(device) for device in jax.devices()]}, "
        f"local_device_count={jax.local_device_count()}, default_backend={jax.default_backend()}"
    )

    policy = _create_analysis_policy(
        train_config,
        checkpoint_dir,
        num_steps=args.num_steps,
        params_dtype=args.params_dtype,
    )
    model = policy._model  # noqa: SLF001
    policy._rng = jax.random.key(args.seed)  # noqa: SLF001

    raw_samples = _load_episode_samples(dataset_root, args.episode_index)
    if args.max_frames is not None:
        raw_samples = raw_samples[: args.max_frames]
    _progress(f"[extract] loaded {len(raw_samples)} frames from episode_{args.episode_index}")

    global_prefusion_features: list[np.ndarray] = []
    global_postfusion_features: list[np.ndarray] = []
    pred_actions: list[np.ndarray] = []
    action_features: list[np.ndarray] = []
    camera_prefusion_features: dict[str, list[np.ndarray]] = {camera: [] for camera in CAMERA_NAMES}
    camera_postfusion_features: dict[str, list[np.ndarray]] = {camera: [] for camera in CAMERA_NAMES}
    metadata_rows: list[dict[str, Any]] = []
    processed_frames = 0

    for batch_start in range(0, len(raw_samples), args.batch_size):
        batch_timer = time.perf_counter()
        batch_end = min(batch_start + args.batch_size, len(raw_samples))
        batch_raw_samples = raw_samples[batch_start:batch_end]
        prepare_start = time.perf_counter()
        transformed_batch, observation = _prepare_batch(policy, batch_raw_samples)
        prepare_elapsed = time.perf_counter() - prepare_start

        prefix_start = time.perf_counter()
        prefix_features = _extract_prefix_features(model, observation)
        prefix_elapsed = time.perf_counter() - prefix_start

        action_start = time.perf_counter()
        batch_actions = _run_policy_batch(policy, observation, transformed_batch)
        action_elapsed = time.perf_counter() - action_start

        global_prefusion_features.extend(list(prefix_features["global_prefusion"]))
        global_postfusion_features.extend(list(prefix_features["global_postfusion"]))
        pred_actions.extend(list(batch_actions))
        action_features.extend([actions.reshape(-1) for actions in batch_actions])

        for camera_name in CAMERA_NAMES:
            camera_prefusion_features[camera_name].extend(list(prefix_features[f"camera_prefusion/{camera_name}"]))
            camera_postfusion_features[camera_name].extend(list(prefix_features[f"camera_postfusion/{camera_name}"]))

        for offset, raw_sample in enumerate(batch_raw_samples):
            base_pre = prefix_features["camera_prefusion/base_0_rgb"][offset]
            left_pre = prefix_features["camera_prefusion/left_wrist_0_rgb"][offset]
            right_pre = prefix_features["camera_prefusion/right_wrist_0_rgb"][offset]
            base_post = prefix_features["camera_postfusion/base_0_rgb"][offset]
            left_post = prefix_features["camera_postfusion/left_wrist_0_rgb"][offset]
            right_post = prefix_features["camera_postfusion/right_wrist_0_rgb"][offset]

            metadata_rows.append(
                {
                    "sample_index": batch_start + offset,
                    "episode_index": raw_sample["episode_index"],
                    "frame_index": raw_sample["frame_index"],
                    "dataset_index": raw_sample["index"],
                    "prompt": raw_sample["prompt"],
                    "task_index": raw_sample["task_index"],
                    "is_good_action": raw_sample.get("is_good_action"),
                    "intervention": raw_sample.get("intervention"),
                    "advantage": raw_sample.get("advantage"),
                    "predicted_value": raw_sample.get("predicted_value"),
                    "sim_prefusion/base_left": float(_cosine_similarity(base_pre[None, :], left_pre[None, :])[0]),
                    "sim_prefusion/base_right": float(_cosine_similarity(base_pre[None, :], right_pre[None, :])[0]),
                    "sim_prefusion/left_right": float(_cosine_similarity(left_pre[None, :], right_pre[None, :])[0]),
                    "sim_postfusion/base_left": float(_cosine_similarity(base_post[None, :], left_post[None, :])[0]),
                    "sim_postfusion/base_right": float(_cosine_similarity(base_post[None, :], right_post[None, :])[0]),
                    "sim_postfusion/left_right": float(_cosine_similarity(left_post[None, :], right_post[None, :])[0]),
                }
            )

        batch_elapsed = time.perf_counter() - batch_timer
        processed_frames = batch_end
        if batch_start == 0:
            _progress(
                "[extract] first batch ready: "
                f"prefix={prefix_features['global_prefusion'].shape}, "
                f"pred_actions={batch_actions.shape}, batch={len(batch_raw_samples)}"
            )

        if (
            batch_start == 0
            or processed_frames % args.log_every == 0
            or processed_frames == len(raw_samples)
        ):
            elapsed = time.perf_counter() - run_start
            avg_per_frame = elapsed / max(processed_frames, 1)
            remaining_frames = len(raw_samples) - processed_frames
            eta = remaining_frames * avg_per_frame
            _progress(
                "[extract] processed "
                f"{processed_frames}/{len(raw_samples)} frames | "
                f"batch={len(batch_raw_samples)} | "
                f"prepare={prepare_elapsed:.2f}s | "
                f"prefix={prefix_elapsed:.2f}s | "
                f"actions={action_elapsed:.2f}s | "
                f"batch_total={batch_elapsed:.2f}s | "
                f"avg/frame={avg_per_frame:.2f}s | "
                f"elapsed={_format_seconds(elapsed)} | "
                f"eta={_format_seconds(eta)}"
            )

    np.save(output_dir / "global_prefusion_prefix_features.npy", np.stack(global_prefusion_features).astype(np.float32))
    np.save(output_dir / "global_postfusion_prefix_features.npy", np.stack(global_postfusion_features).astype(np.float32))
    np.save(output_dir / "pred_actions.npy", np.stack(pred_actions).astype(np.float32))
    np.save(output_dir / "action_features.npy", np.stack(action_features).astype(np.float32))
    for camera_name in CAMERA_NAMES:
        safe_name = camera_name.replace("/", "_")
        np.save(output_dir / f"{safe_name}_prefusion_features.npy", np.stack(camera_prefusion_features[camera_name]).astype(np.float32))
        np.save(output_dir / f"{safe_name}_postfusion_features.npy", np.stack(camera_postfusion_features[camera_name]).astype(np.float32))

    with (output_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for row in metadata_rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    summary = {
        "config_name": args.config_name,
        "checkpoint_dir": str(checkpoint_dir),
        "episode_index": args.episode_index,
        "num_frames": len(raw_samples),
        "max_frames": args.max_frames,
        "num_steps": args.num_steps,
        "batch_size": args.batch_size,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _progress(f"[extract] saved features to {output_dir}")


if __name__ == "__main__":
    main()

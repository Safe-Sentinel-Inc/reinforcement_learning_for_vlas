"""Data loading utilities for the offline evaluation pipeline.

Provides functions to read LeRobot-format Parquet datasets, decode images,
parse episode specifications, and load metadata such as task maps and fold assignments.
"""

import io
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


CAMERA_NAMES = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
DEFAULT_DATASET = "./lerobot_data/safe-sentinel-co/insert_pin_slow_v21"
DEFAULT_CONFIG = "pi06_rl_pretrain_airbot_clothes_folding"
DEFAULT_CHECKPOINT = "checkpoints/pi06_rl_pretrain_airbot_clothes_folding/pi06_fold_clothes_v2/96000"


def _progress(message: str) -> None:
    """Log a message and print it immediately to stdout for real-time feedback."""
    logging.info(message)
    print(message, flush=True)


def _load_json(path: Path) -> dict[str, Any]:
    """Read and parse a single JSON file, returning its contents as a dictionary."""
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSON-Lines file and return a list of parsed dictionaries, skipping blank lines."""
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_task_map(dataset_root: Path) -> dict[int, str]:
    """Load the task index to task description mapping from the dataset metadata."""
    tasks = _load_jsonl(dataset_root / "meta" / "tasks.jsonl")
    return {int(row["task_index"]): row["task"] for row in tasks}


def _load_episode_lengths(dataset_root: Path) -> dict[int, int]:
    """Load a mapping from episode index to frame count from the dataset metadata."""
    entries = _load_jsonl(dataset_root / "meta" / "episodes.jsonl")
    return {int(row["episode_index"]): int(row["length"]) for row in entries}


def _load_folds(dataset_root: Path) -> dict[int, int]:
    """Load the episode-to-fold assignment mapping from the dataset metadata."""
    return {int(k): int(v) for k, v in _load_json(dataset_root / "meta" / "folds.json").items()}


def _episode_path(dataset_root: Path, episode_index: int) -> Path:
    """Construct the filesystem path to the Parquet file for a given episode index."""
    return dataset_root / "data" / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}.parquet"


def _parse_episode_spec(spec: str | None, folds: dict[int, int], max_per_fold: int, seed: int) -> list[int]:
    """Parse a comma-separated episode list, or sample up to max_per_fold episodes from each fold."""
    if spec:
        return [int(part.strip()) for part in spec.split(",") if part.strip()]

    rng = np.random.default_rng(seed)
    by_fold: dict[int, list[int]] = {}
    for ep_idx, fold in folds.items():
        by_fold.setdefault(fold, []).append(ep_idx)

    selected: list[int] = []
    for fold in sorted(by_fold):
        candidates = sorted(by_fold[fold])
        rng.shuffle(candidates)
        selected.extend(sorted(candidates[:max_per_fold]))
    return sorted(selected)


def _decode_image(cell: dict[str, Any]) -> np.ndarray:
    """Decode a raw-bytes image cell from a Parquet row into a NumPy array."""
    return np.asarray(Image.open(io.BytesIO(cell["bytes"])))


def _read_episode_numeric(dataset_root: Path, episode_index: int) -> dict[str, Any]:
    """Read the scalar numeric columns for one episode and return them as NumPy arrays."""
    columns = [
        "binned_value",
        "predicted_value",
        "advantage",
        "is_good_action",
        "intervention",
        "frame_index",
        "task_index",
    ]
    table = pq.read_table(_episode_path(dataset_root, episode_index), columns=columns)
    arrays = {}
    for col in columns:
        arrays[col] = np.asarray(table.column(col).to_numpy())
    arrays["episode_index"] = episode_index
    arrays["num_frames"] = int(table.num_rows)
    return arrays


def _read_policy_rows(dataset_root: Path, episode_index: int, frame_indices: np.ndarray) -> list[dict[str, Any]]:
    """Read full observation rows (including images) for specific frame indices within an episode."""
    columns = [
        *CAMERA_NAMES,
        "state",
        "actions",
        "task_index",
        "frame_index",
        "index",
        "episode_index",
        "predicted_value",
        "advantage",
        "is_good_action",
        "intervention",
    ]
    table = pq.read_table(_episode_path(dataset_root, episode_index), columns=columns)
    sampled = table.take(pa.array(frame_indices, type=pa.int64()))
    return sampled.to_pylist()

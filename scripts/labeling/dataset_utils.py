"""Helpers for reading, writing, and transforming LeRobot v2 parquet datasets."""

import concurrent.futures
import json
import logging
import os
import pathlib
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)


def read_episode_lengths(repo_id: str) -> dict[int, int]:
    """Return a mapping of episode_index to frame count by parsing episodes.jsonl."""
    from lerobot.common.constants import HF_LEROBOT_HOME

    episodes_path = HF_LEROBOT_HOME / repo_id / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        raise FileNotFoundError(f"episodes.jsonl not found: {episodes_path}")

    lengths: dict[int, int] = {}
    with open(episodes_path) as f:
        for line in f:
            ep = json.loads(line)
            lengths[ep["episode_index"]] = ep["length"]
    return lengths


def parse_range_string(spec: str) -> set[int]:
    """Convert a comma-separated range specification (e.g. "0-100,200,300-400") to a set of integers."""
    result: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            result.update(range(int(lo), int(hi) + 1))
        else:
            result.add(int(part))
    return result


def read_fold_assignments(repo_id: str) -> dict[int, int]:
    """Load the episode-to-fold mapping stored in meta/folds.json."""
    from lerobot.common.constants import HF_LEROBOT_HOME

    folds_path = HF_LEROBOT_HOME / repo_id / "meta" / "folds.json"
    if not folds_path.exists():
        raise FileNotFoundError(f"folds.json not found: {folds_path}")
    with open(folds_path) as f:
        raw = json.load(f)
    return {int(k): int(v) for k, v in raw.items()}


def compute_fold_assignments(
    episode_indices: list[int],
    num_folds: int,
    seed: int = 42,
) -> dict[int, int]:
    """Shuffle episodes and deal them round-robin into K folds.

    Returns a dict mapping each episode_index to its fold ID (0 through num_folds-1).
    """
    rng = np.random.RandomState(seed)
    shuffled = sorted(episode_indices)
    rng.shuffle(shuffled)
    assignments = {}
    for i, ep_idx in enumerate(shuffled):
        assignments[ep_idx] = i % num_folds
    return assignments


def resolve_success_episodes(
    spec: str | list[int],
    failed: list[int] | tuple[int, ...],
    all_episodes: set[int],
) -> set[int]:
    """Determine which episodes are successful based on the given specification.

    Args:
        spec: Either "all", an explicit list of indices, or a range string like "0-100,200-300".
        failed: Episode indices to exclude regardless of the spec.
        all_episodes: The full set of valid episode indices in the dataset.
    """
    if spec == "all":
        result = set(all_episodes)
    elif isinstance(spec, (list, tuple)):
        result = set(int(x) for x in spec)
    elif isinstance(spec, str):
        result = parse_range_string(spec)
    else:
        raise ValueError(f"Unsupported success_episodes spec: {spec!r}")
    result -= set(int(x) for x in failed)
    return result


def _write_single_episode(
    ep_idx: int,
    parquet_path: pathlib.Path,
    columns: dict[str, dict[int, np.ndarray]],
    lenient: bool,
) -> int:
    """Update one episode's parquet file with new columns using atomic write.

    Returns the row count on success, or 0 if the file was missing.
    """
    if not parquet_path.exists():
        logger.warning("Parquet not found: %s, skipping ep %d", parquet_path, ep_idx)
        return 0

    table = pq.read_table(parquet_path)

    for col_name, ep_data in columns.items():
        if col_name in table.column_names:
            table = table.remove_column(table.column_names.index(col_name))

        if ep_idx in ep_data:
            arr = ep_data[ep_idx]
            if len(arr) != table.num_rows:
                if lenient:
                    logger.warning(
                        "ep %d col %s: len %d != rows %d, adjusting",
                        ep_idx, col_name, len(arr), table.num_rows,
                    )
                    if len(arr) > table.num_rows:
                        arr = arr[: table.num_rows]
                    else:
                        arr = np.pad(arr, (0, table.num_rows - len(arr)))
                else:
                    raise ValueError(
                        f"ep {ep_idx} col {col_name}: array length {len(arr)} "
                        f"!= parquet rows {table.num_rows}. "
                        "Use --lenient to pad/truncate instead of failing."
                    )
        else:
            arr = np.zeros(table.num_rows, dtype=np.int64)

        table = table.append_column(col_name, pa.array(arr))

    tmp_path = parquet_path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp_path)
    tmp_path.rename(parquet_path)
    return table.num_rows


def write_columns_to_dataset(
    repo_id: str,
    columns: dict[str, dict[int, np.ndarray]],
    column_meta: dict[str, dict],
    output_dir: str | None = None,
    lenient: bool = False,
    max_workers: int = 8,
):
    """Append or overwrite columns across all episode parquet files in a LeRobot v2 dataset.

    Writes are atomic (tmp file then rename) and parallelized across episodes.
    """
    from lerobot.common.constants import HF_LEROBOT_HOME
    import shutil

    src_path = HF_LEROBOT_HOME / repo_id
    if output_dir:
        dst_path = pathlib.Path(output_dir) / repo_id
        if dst_path.exists():
            shutil.rmtree(dst_path)
        shutil.copytree(src_path, dst_path)
        logger.info("Copied dataset to %s", dst_path)
    else:
        dst_path = src_path
        logger.info("Modifying dataset in-place at %s", dst_path)

    info_path = dst_path / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    episodes_path = dst_path / "meta" / "episodes.jsonl"
    with open(episodes_path) as f:
        episodes = [json.loads(line) for line in f]

    chunks_size = info.get("chunks_size", 1000)
    tasks = []
    for ep in episodes:
        ep_idx = ep["episode_index"]
        chunk_idx = ep_idx // chunks_size
        parquet_path = (
            dst_path / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{ep_idx:06d}.parquet"
        )
        tasks.append((ep_idx, parquet_path))

    total_written = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_write_single_episode, ep_idx, pq_path, columns, lenient): ep_idx
            for ep_idx, pq_path in tasks
        }
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Writing columns",
        ):
            total_written += future.result()

    # Register any new columns in the dataset's feature metadata
    changed = False
    for col_name, meta in column_meta.items():
        if col_name not in info.get("features", {}):
            info["features"][col_name] = meta
            changed = True
    if changed:
        with open(info_path, "w") as f:
            json.dump(info, f, indent=4)

    logger.info("Wrote %d column(s) to %d rows -> %s", len(columns), total_written, dst_path)

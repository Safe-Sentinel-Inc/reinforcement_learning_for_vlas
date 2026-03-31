"""Statistical metrics and episode-level aggregation for offline evaluation.

Contains correlation helpers, sampling utilities, binned statistics, and the
main function that builds per-episode metric records from raw Parquet data.
"""

from pathlib import Path
from typing import Any

import numpy as np

from .data_loading import _read_episode_numeric


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the Pearson correlation between two arrays, returning NaN if either is constant or too short."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.size < 2 or b.size < 2:
        return float("nan")
    if np.allclose(a.std(), 0.0) or np.allclose(b.std(), 0.0):
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _linear_slope(values: np.ndarray) -> float:
    """Fit a linear trend to the values and return the slope over the normalized [0, 1] range."""
    values = np.asarray(values, dtype=np.float32)
    if values.size < 2:
        return 0.0
    x = np.linspace(0.0, 1.0, values.size, dtype=np.float32)
    return float(np.polyfit(x, values, 1)[0])


def _sample_indices(total: int, max_points: int) -> np.ndarray:
    """Return evenly spaced indices into a sequence, capped at max_points."""
    if total <= max_points:
        return np.arange(total, dtype=np.int64)
    return np.linspace(0, total - 1, max_points, dtype=np.int64)


def _bin_progress_statistics(progress: np.ndarray, values: np.ndarray, bins: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Partition the [0, 1] progress range into equal bins and compute the mean value in each bin."""
    edges = np.linspace(0.0, 1.0, bins + 1, dtype=np.float32)
    centers = 0.5 * (edges[:-1] + edges[1:])
    stats = np.full(bins, np.nan, dtype=np.float32)
    for i in range(bins):
        if i == bins - 1:
            mask = (progress >= edges[i]) & (progress <= edges[i + 1])
        else:
            mask = (progress >= edges[i]) & (progress < edges[i + 1])
        if np.any(mask):
            stats[i] = float(np.mean(values[mask]))
    return centers, stats


def _build_episode_records(dataset_root: Path, episodes: list[int], folds: dict[int, int]) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Load numeric data for each episode and compute per-episode and pooled aggregate statistics.

    Returns a list of per-episode metric dictionaries and a dict of concatenated
    arrays across all episodes for global-level analysis.
    """
    episode_metrics: list[dict[str, Any]] = []
    all_values: list[np.ndarray] = []
    all_advantages: list[np.ndarray] = []
    all_good: list[np.ndarray] = []
    all_intervention: list[np.ndarray] = []
    all_progress: list[np.ndarray] = []
    all_binned_progress: list[np.ndarray] = []
    all_local_dvalue: list[np.ndarray] = []
    all_local_adv: list[np.ndarray] = []

    for episode_index in episodes:
        data = _read_episode_numeric(dataset_root, episode_index)
        num_frames = data["num_frames"]
        progress = np.linspace(0.0, 1.0, num_frames, dtype=np.float32)
        values = data["predicted_value"].astype(np.float32)
        advantages = data["advantage"].astype(np.float32)
        good = data["is_good_action"].astype(np.float32)
        intervention = data["intervention"].astype(np.float32)
        binned_progress = data["binned_value"].astype(np.float32) / 200.0

        episode_metrics.append(
            {
                "episode_index": episode_index,
                "fold": folds[episode_index],
                "num_frames": num_frames,
                "mean_predicted_value": float(values.mean()),
                "std_predicted_value": float(values.std()),
                "value_slope": _linear_slope(values),
                "value_progress_corr": _safe_corr(values, progress),
                "value_binned_corr": _safe_corr(values, binned_progress),
                "mean_advantage": float(advantages.mean()),
                "std_advantage": float(advantages.std()),
                "positive_fraction": float(good.mean()),
                "intervention_fraction": float(intervention.mean()),
                "advantage_delta_value_corr": _safe_corr(advantages[:-1], np.diff(values)) if num_frames > 1 else float("nan"),
                "terminal_binned_value": float(data["binned_value"][-1]),
            }
        )

        all_values.append(values)
        all_advantages.append(advantages)
        all_good.append(good)
        all_intervention.append(intervention)
        all_progress.append(progress)
        all_binned_progress.append(binned_progress)
        if num_frames > 1:
            all_local_dvalue.append(np.diff(values))
            all_local_adv.append(advantages[:-1])

    aggregate = {
        "predicted_value": np.concatenate(all_values, axis=0),
        "advantage": np.concatenate(all_advantages, axis=0),
        "is_good_action": np.concatenate(all_good, axis=0),
        "intervention": np.concatenate(all_intervention, axis=0),
        "progress": np.concatenate(all_progress, axis=0),
        "binned_progress": np.concatenate(all_binned_progress, axis=0),
        "local_dvalue": np.concatenate(all_local_dvalue, axis=0) if all_local_dvalue else np.zeros(0, dtype=np.float32),
        "local_advantage": np.concatenate(all_local_adv, axis=0) if all_local_adv else np.zeros(0, dtype=np.float32),
    }
    return episode_metrics, aggregate


def _flatten_metric(results: list[dict[str, Any]], key: str) -> np.ndarray:
    """Extract a single numeric field from a list of result dicts into a flat float32 array."""
    return np.asarray([row[key] for row in results], dtype=np.float32)


def _compute_sequence_metrics(episode_map: dict[int, list[np.ndarray]]) -> tuple[np.ndarray, np.ndarray, dict[int, tuple[np.ndarray, np.ndarray]]]:
    """Compute speed (first-order difference norm) and jerk (second-order) for action sequences per episode."""
    all_speed: list[np.ndarray] = []
    all_jerk: list[np.ndarray] = []
    per_episode: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for episode_index, seq in episode_map.items():
        arr = np.asarray(seq, dtype=np.float32)
        speed = np.linalg.norm(np.diff(arr, axis=0), axis=1).astype(np.float32) if len(arr) > 1 else np.zeros(0, dtype=np.float32)
        jerk = np.linalg.norm(np.diff(arr, n=2, axis=0), axis=1).astype(np.float32) if len(arr) > 2 else np.zeros(0, dtype=np.float32)
        all_speed.append(speed)
        all_jerk.append(jerk)
        per_episode[episode_index] = (speed, jerk)
    return (
        np.concatenate(all_speed, axis=0) if all_speed else np.zeros(0, dtype=np.float32),
        np.concatenate(all_jerk, axis=0) if all_jerk else np.zeros(0, dtype=np.float32),
        per_episode,
    )

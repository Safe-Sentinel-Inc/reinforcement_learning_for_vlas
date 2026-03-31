"""Plotting functions for offline evaluation diagnostics.

Generates multi-panel matplotlib figures covering value-function reliability,
advantage quality, episode summaries, policy conditioning effects, and
representation support analysis.
"""

import csv
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .metrics import (
    _bin_progress_statistics,
    _compute_sequence_metrics,
    _flatten_metric,
    _safe_corr,
    _sample_indices,
)


def _add_caption(fig: plt.Figure, caption: str) -> None:
    """Add a centered explanatory caption at the bottom of a matplotlib figure."""
    fig.subplots_adjust(bottom=0.16)
    fig.text(0.5, 0.03, caption, ha="center", va="bottom", fontsize=10, wrap=True)


def _save_episode_csv(records: list[dict[str, Any]], output_path: Path) -> None:
    """Write a list of episode metric dictionaries to a CSV file."""
    if not records:
        return
    fieldnames = list(records[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _plot_vf_overview(records: list[dict[str, Any]], agg: dict[str, np.ndarray], output_path: Path) -> None:
    """Create a 2x2 panel figure assessing value-function reliability.

    Panels: (A) predicted value histogram, (B) value vs. binned progress scatter,
    (C) mean value and positive fraction over normalized time, (D) per-fold value boxplot.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    axes[0, 0].hist(agg["predicted_value"], bins=80, color="tab:blue", alpha=0.8, edgecolor="white", linewidth=0.3)
    axes[0, 0].set_title("A. Predicted Value Distribution")
    axes[0, 0].set_xlabel("predicted_value")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].grid(True, alpha=0.2)

    sample_idx = _sample_indices(len(agg["predicted_value"]), 12000)
    corr = _safe_corr(agg["predicted_value"], agg["binned_progress"])
    axes[0, 1].scatter(
        agg["binned_progress"][sample_idx],
        agg["predicted_value"][sample_idx],
        s=10,
        alpha=0.35,
        color="tab:purple",
    )
    axes[0, 1].set_title(f"B. Predicted Value vs Progress (corr={corr:.3f})")
    axes[0, 1].set_xlabel("normalized binned_value")
    axes[0, 1].set_ylabel("predicted_value")
    axes[0, 1].grid(True, alpha=0.2)

    centers, mean_value = _bin_progress_statistics(agg["progress"], agg["predicted_value"])
    _, mean_good = _bin_progress_statistics(agg["progress"], agg["is_good_action"])
    axes[1, 0].plot(centers, mean_value, label="mean predicted_value", linewidth=2.0)
    ax_twin = axes[1, 0].twinx()
    ax_twin.plot(centers, mean_good, label="positive fraction", color="tab:green", linewidth=2.0)
    axes[1, 0].set_title("C. Mean Value and Positive Fraction over Normalized Time")
    axes[1, 0].set_xlabel("normalized episode progress")
    axes[1, 0].set_ylabel("predicted_value")
    ax_twin.set_ylabel("positive fraction")
    axes[1, 0].grid(True, alpha=0.2)
    lines, labels = axes[1, 0].get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    axes[1, 0].legend(lines + lines2, labels + labels2, loc="best")

    grouped: dict[int, list[float]] = {}
    for row in records:
        grouped.setdefault(int(row["fold"]), []).append(float(row["mean_predicted_value"]))
    folds = sorted(grouped)
    axes[1, 1].boxplot([grouped[fold] for fold in folds], tick_labels=[f"fold {fold}" for fold in folds], showfliers=False)
    axes[1, 1].set_title("D. Episode Mean Value by Fold")
    axes[1, 1].set_ylabel("mean predicted_value")
    axes[1, 1].grid(True, alpha=0.2, axis="y")

    fig.suptitle("PI06 Offline Evaluation | Value Function Reliability", fontsize=15)
    _add_caption(
        fig,
        "A/B judge whether predicted_value aligns with task progress rather than noise. "
        "C checks if value and positive actions increase as an episode progresses. "
        "D checks whether held-out folds behave consistently instead of one fold dominating the signal.",
    )
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_advantage_quality(records: list[dict[str, Any]], agg: dict[str, np.ndarray], output_path: Path) -> None:
    """Create a 2x2 panel figure assessing advantage label quality.

    Panels: (A) advantage distributions for good/bad actions, (B) positive and intervention
    density over time, (C) local advantage vs. value delta scatter, (D) episode-level
    advantage vs. positive fraction colored by fold.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    good_mask = agg["is_good_action"] > 0.5
    axes[0, 0].hist(agg["advantage"][good_mask], bins=80, alpha=0.65, label="good", color="tab:green", edgecolor="white", linewidth=0.3)
    axes[0, 0].hist(agg["advantage"][~good_mask], bins=80, alpha=0.65, label="bad", color="tab:red", edgecolor="white", linewidth=0.3)
    axes[0, 0].axvline(0.0, color="gray", linestyle="--", linewidth=1.0)
    axes[0, 0].set_title("A. Advantage Distribution for Good/Bad Actions")
    axes[0, 0].set_xlabel("advantage")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].legend(loc="best")
    axes[0, 0].grid(True, alpha=0.2)

    centers, mean_positive = _bin_progress_statistics(agg["progress"], agg["is_good_action"])
    _, mean_intervention = _bin_progress_statistics(agg["progress"], agg["intervention"])
    axes[0, 1].plot(centers, mean_positive, label="positive fraction", linewidth=2.0, color="tab:green")
    axes[0, 1].plot(centers, mean_intervention, label="intervention fraction", linewidth=2.0, color="tab:red")
    axes[0, 1].set_title("B. Positive/Intervention Density over Normalized Time")
    axes[0, 1].set_xlabel("normalized episode progress")
    axes[0, 1].set_ylabel("fraction")
    axes[0, 1].legend(loc="best")
    axes[0, 1].grid(True, alpha=0.2)

    delta_corr = _safe_corr(agg["local_advantage"], agg["local_dvalue"])
    sample_idx = _sample_indices(len(agg["local_advantage"]), 12000)
    axes[1, 0].scatter(
        agg["local_advantage"][sample_idx],
        agg["local_dvalue"][sample_idx],
        s=10,
        alpha=0.35,
        color="tab:orange",
    )
    axes[1, 0].axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    axes[1, 0].axvline(0.0, color="gray", linestyle="--", linewidth=1.0)
    axes[1, 0].set_title(f"C. Local Advantage vs Next-Step Value Delta (corr={delta_corr:.3f})")
    axes[1, 0].set_xlabel("advantage_t")
    axes[1, 0].set_ylabel("predicted_value[t+1] - predicted_value[t]")
    axes[1, 0].grid(True, alpha=0.2)

    x = np.array([row["mean_advantage"] for row in records], dtype=np.float32)
    y = np.array([row["positive_fraction"] for row in records], dtype=np.float32)
    colors = [row["fold"] for row in records]
    scatter = axes[1, 1].scatter(x, y, c=colors, cmap="tab10", s=45)
    axes[1, 1].set_title("D. Episode Mean Advantage vs Positive Fraction")
    axes[1, 1].set_xlabel("mean advantage")
    axes[1, 1].set_ylabel("positive fraction")
    axes[1, 1].grid(True, alpha=0.2)
    legend_handles, _ = scatter.legend_elements()
    axes[1, 1].legend(legend_handles, [f"fold {int(f)}" for f in sorted(set(colors))], loc="best", title="fold")

    fig.suptitle("PI06 Offline Evaluation | Advantage and is_good_action Quality", fontsize=15)
    _add_caption(
        fig,
        "A checks whether advantage meaningfully separates good from bad actions. "
        "B tests whether positive labels cluster in coherent task phases. "
        "C checks whether advantage is locally aligned with value changes. "
        "D summarizes whether episode-level advantage statistics match positive-action density.",
    )
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_episode_summary(records: list[dict[str, Any]], output_path: Path) -> None:
    """Create a 2x2 bar chart summarizing episodes sorted by mean predicted value.

    Panels: (A) mean predicted value, (B) value slope, (C) positive fraction,
    (D) intervention fraction. Bars are colored by fold assignment.
    """
    ordered = sorted(records, key=lambda row: row["mean_predicted_value"], reverse=True)
    labels = [f"ep{row['episode_index']}" for row in ordered]
    x = np.arange(len(ordered))

    fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharex=True)
    panels = [
        ("A. Episode Mean Predicted Value", "mean_predicted_value"),
        ("B. Episode Value Slope", "value_slope"),
        ("C. Positive Fraction", "positive_fraction"),
        ("D. Intervention Fraction", "intervention_fraction"),
    ]
    for ax, (title, key) in zip(axes.ravel(), panels):
        values = [float(row[key]) for row in ordered]
        colors = ["tab:blue" if row["fold"] == 0 else "tab:orange" if row["fold"] == 1 else "tab:green" for row in ordered]
        ax.bar(x, values, color=colors, alpha=0.8, edgecolor="white")
        ax.set_title(title)
        ax.grid(True, alpha=0.2, axis="y")
    for ax in axes[-1]:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")

    fig.suptitle("PI06 Offline Evaluation | Episode-Level Summary", fontsize=15)
    _add_caption(
        fig,
        "Episodes are sorted by mean predicted_value. "
        "This summary is designed for ranking and comparing episodes before using any representation plot.",
    )
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_policy_condition(results: list[dict[str, Any]], aggregates: dict[str, Any], output_path: Path) -> dict[str, float]:
    """Create a 2x2 panel figure comparing policy outputs under positive vs. negative advantage.

    Panels: (A) action gap per sample, (B) action gap vs. predicted value,
    (C) action speed/jerk stability boxplots, (D) prefix gap vs. action gap.
    Returns a dictionary of summary statistics.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    action_gap = _flatten_metric(results, "action_gap")
    cosine = _flatten_metric(results, "action_cosine")
    prefix_gap = _flatten_metric(results, "prefix_condition_gap")
    value = _flatten_metric(results, "predicted_value")
    good = _flatten_metric(results, "is_good_action")
    x = np.arange(len(results))
    colors = np.where(good > 0.5, "tab:green", "tab:red")

    axes[0, 0].scatter(x, action_gap, c=colors, s=28, alpha=0.85)
    axes[0, 0].set_title("A. Action Gap: advantage=True vs advantage=False")
    axes[0, 0].set_xlabel("sample index")
    axes[0, 0].set_ylabel("||a_pos - a_neg||")
    axes[0, 0].grid(True, alpha=0.2)

    axes[0, 1].scatter(value, action_gap, c=cosine, cmap="viridis", s=35, alpha=0.85)
    axes[0, 1].set_title("B. Action Gap vs Predicted Value")
    axes[0, 1].set_xlabel("predicted_value")
    axes[0, 1].set_ylabel("action gap")
    axes[0, 1].grid(True, alpha=0.2)

    pos_speed, pos_jerk, _ = _compute_sequence_metrics(aggregates["episode_pos_actions"])
    neg_speed, neg_jerk, _ = _compute_sequence_metrics(aggregates["episode_neg_actions"])
    axes[1, 0].boxplot(
        [pos_speed, neg_speed, pos_jerk, neg_jerk],
        tick_labels=["speed(+)", "speed(-)", "jerk(+)", "jerk(-)"],
        showfliers=False,
    )
    axes[1, 0].set_title("C. Stability under Positive/Negative Advantage")
    axes[1, 0].set_ylabel("magnitude")
    axes[1, 0].grid(True, alpha=0.2, axis="y")

    axes[1, 1].scatter(prefix_gap, action_gap, c=good, cmap="coolwarm", s=35, alpha=0.85)
    axes[1, 1].set_title("D. Representation Condition Gap vs Action Condition Gap")
    axes[1, 1].set_xlabel("prefix condition gap")
    axes[1, 1].set_ylabel("action gap")
    axes[1, 1].grid(True, alpha=0.2)

    fig.suptitle("PI06 Offline Evaluation | Advantage-Conditioned Policy Comparison", fontsize=15)
    _add_caption(
        fig,
        "This report checks whether the advantage condition actually changes policy outputs, "
        "and whether positive conditioning is associated with smoother action sequences rather than random drift.",
    )
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return {
        "mean_action_gap": float(action_gap.mean()) if len(action_gap) else 0.0,
        "mean_action_cosine": float(cosine.mean()) if len(cosine) else 0.0,
        "mean_prefix_condition_gap": float(prefix_gap.mean()) if len(prefix_gap) else 0.0,
        "mean_positive_speed": float(pos_speed.mean()) if len(pos_speed) else 0.0,
        "mean_negative_speed": float(neg_speed.mean()) if len(neg_speed) else 0.0,
        "mean_positive_jerk": float(pos_jerk.mean()) if len(pos_jerk) else 0.0,
        "mean_negative_jerk": float(neg_jerk.mean()) if len(neg_jerk) else 0.0,
    }


def _plot_feature_support(results: list[dict[str, Any]], aggregates: dict[str, Any], output_path: Path) -> None:
    """Create a 2x2 auxiliary panel figure for representation-level diagnostics.

    Panels: (A) prefix cosine similarity matrix, (B) cross-camera similarity before/after
    fusion, (C) prefix gap vs. action gap, (D) prefix temporal drift histogram.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    prefix_features = np.asarray([row["pos_prefix"] for row in results], dtype=np.float32)
    prefix_features = prefix_features / np.clip(np.linalg.norm(prefix_features, axis=1, keepdims=True), 1e-8, None)
    sim = prefix_features @ prefix_features.T
    im = axes[0, 0].imshow(sim, vmin=-1.0, vmax=1.0, cmap="coolwarm", aspect="auto")
    axes[0, 0].set_title("A. Prefix Similarity Matrix (positive condition)")
    axes[0, 0].set_xlabel("sample index")
    axes[0, 0].set_ylabel("sample index")
    fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)

    pre = _flatten_metric(results, "camera_similarity_prefusion")
    post = _flatten_metric(results, "camera_similarity_postfusion")
    axes[0, 1].plot(np.arange(len(results)), pre, label="before fusion", linewidth=1.8)
    axes[0, 1].plot(np.arange(len(results)), post, label="after fusion", linewidth=1.8)
    axes[0, 1].set_title("B. Same-Frame Cross-Camera Similarity")
    axes[0, 1].set_xlabel("sample index")
    axes[0, 1].set_ylabel("mean cosine similarity")
    axes[0, 1].legend(loc="best")
    axes[0, 1].grid(True, alpha=0.2)

    prefix_gap = _flatten_metric(results, "prefix_condition_gap")
    action_gap = _flatten_metric(results, "action_gap")
    axes[1, 0].scatter(prefix_gap, action_gap, c=post - pre, cmap="plasma", s=35, alpha=0.85)
    axes[1, 0].set_title("C. Prefix Condition Gap vs Action Gap")
    axes[1, 0].set_xlabel("prefix condition gap")
    axes[1, 0].set_ylabel("action gap")
    axes[1, 0].grid(True, alpha=0.2)

    prefix_speed, _, _ = _compute_sequence_metrics(aggregates["episode_prefix"])
    axes[1, 1].hist(prefix_speed, bins=30, color="tab:purple", alpha=0.8, edgecolor="white", linewidth=0.3)
    axes[1, 1].set_title("D. Prefix Temporal Drift Distribution")
    axes[1, 1].set_xlabel("||f_t - f_{t-1}||")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].grid(True, alpha=0.2)

    fig.suptitle("PI06 Offline Evaluation | Representation Support (Auxiliary)", fontsize=15)
    _add_caption(
        fig,
        "These representation plots are auxiliary diagnostics only. "
        "They help explain why a checkpoint looks stable or unstable, but should not override value-function or action-quality evidence.",
    )
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

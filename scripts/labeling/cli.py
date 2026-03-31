"""Command-line interface that dispatches to the three labeling modes."""

import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Label a LeRobot dataset for pi0.6* (Recap) training"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Subcommand for progress-based labeling with binary success/fail annotations
    p1 = subparsers.add_parser(
        "add_labels",
        help="Add binned_value + intervention columns from binary labels",
    )
    p1.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config.py (reads TASK_NAME, SUCCESS_EPISODES, etc.). "
    )
    p1.add_argument("--num-bins", type=int, default=200)
    p1.add_argument("--return-min", type=float, default=0.0)
    p1.add_argument("--return-max", type=float, default=1.0)
    p1.add_argument("--output-dir", type=str, default=None)
    p1.add_argument(
        "--lenient",
        action="store_true",
        default=False,
        help="Pad/truncate on length mismatch instead of failing.",
    )
    p1.add_argument(
        "--num-folds",
        type=int,
        default=0,
        help="Number of folds for K-fold cross-validation. 0 = no folding.",
    )
    p1.add_argument(
        "--fold-seed",
        type=int,
        default=42,
        help="Random seed for fold assignment shuffle.",
    )
    p1.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Override the repo ID from config (default: use TASK_NAME from config.py).",
    )

    # Subcommand for value-function-based advantage labeling
    p2 = subparsers.add_parser(
        "vf_label",
        help="Use trained VF to label is_good_action via advantage threshold",
    )
    p2.add_argument("--repo-id", required=True, help="LeRobot dataset repo ID")
    p2.add_argument("--vf-config", required=True)
    p2.add_argument("--vf-checkpoint-dir", required=True)
    p2.add_argument("--gamma", type=float, default=0.98, help="Discount factor for advantage computation")
    p2.add_argument("--positive-fraction", type=float, default=0.3)
    p2.add_argument("--batch-size", type=int, default=32)
    p2.add_argument("--return-min", type=float, default=0.0)
    p2.add_argument("--return-max", type=float, default=1.0)
    p2.add_argument("--output-dir", type=str, default=None)
    p2.add_argument("--infer-fold", type=int, default=None, help="Only infer on episodes in this fold (reads meta/folds.json)")
    p2.add_argument("--values-dir", type=str, default=None, help="Directory for intermediate per-episode value files")

    # Subcommand for combining per-episode value shards into final labels
    p3 = subparsers.add_parser(
        "vf_merge",
        help="Merge sharded VF values and compute is_good_action labels",
    )
    p3.add_argument("--repo-id", required=True, help="LeRobot dataset repo ID")
    p3.add_argument("--values-dir", required=True, help="Directory with per-episode value .npy files")
    p3.add_argument("--gamma", type=float, default=0.98, help="Discount factor for advantage computation")
    p3.add_argument("--positive-fraction", type=float, default=0.3)
    p3.add_argument("--return-min", type=float, default=0.0)
    p3.add_argument("--return-max", type=float, default=1.0)
    p3.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()

    if args.mode == "add_labels":
        from .progress_labeling import _run_add_labels
        _run_add_labels(args)
    elif args.mode == "vf_label":
        from .vf_inference import _run_vf_label
        _run_vf_label(args)
    elif args.mode == "vf_merge":
        from .advantage_labeling import _run_vf_merge
        _run_vf_merge(args)

#!/usr/bin/env bash
# Generates progress labels (binned_value), intervention flags, and K-fold
# assignments from binary success/failure episode annotations.
# Must be re-run from scratch whenever new episodes are added to the dataset.
set -euo pipefail
cd "$(dirname "$0")/../.."

# ─── CONFIG ────────────────────────────────────────────────────────────────────
CONFIG=mcap_data/fold_clothes/config.py
REPO_ID=Fold_clothes   # Dataset name; leave blank to infer from config.py TASK_NAME
NUM_FOLDS=3
# ───────────────────────────────────────────────────────────────────────────────

mkdir -p logs
LOG_FILE="logs/add_labels_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"

REPO_FLAG=""
if [[ -n "$REPO_ID" ]]; then
    REPO_FLAG="--repo-id ${REPO_ID}"
fi

HF_LEROBOT_HOME=./lerobot_data uv run scripts/add_returns_to_lerobot.py add_labels \
    --config "${CONFIG}" \
    --num-folds "${NUM_FOLDS}" \
    ${REPO_FLAG}

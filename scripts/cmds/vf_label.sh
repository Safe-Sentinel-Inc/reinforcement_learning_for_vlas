#!/usr/bin/env bash
# Scores the entire dataset with a single value function and writes is_good_action labels.
# This is the non-K-fold (legacy) approach; prefer vf_kfold_label.sh for cross-validated labeling.
set -euo pipefail
cd "$(dirname "$0")/../.."

# ─── CONFIG ────────────────────────────────────────────────────────────────────
REPO_ID=Fold_clothes
VF_CONFIG=pi06_rl_vf_airbot_clothes_folding
VF_CHECKPOINT_DIR=checkpoints/pi06_rl_vf_airbot_clothes_folding/vf_v1/20000
GPUS=0,1
POSITIVE_FRACTION=0.3
BATCH_SIZE=32
# ───────────────────────────────────────────────────────────────────────────────

mkdir -p logs
LOG_FILE="logs/vf_label_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"

CUDA_VISIBLE_DEVICES="${GPUS}" HF_LEROBOT_HOME=./lerobot_data \
    uv run scripts/add_returns_to_lerobot.py vf_label \
        --repo-id "${REPO_ID}" \
        --vf-config "${VF_CONFIG}" \
        --vf-checkpoint-dir "${VF_CHECKPOINT_DIR}" \
        --positive-fraction "${POSITIVE_FRACTION}" \
        --batch-size "${BATCH_SIZE}"

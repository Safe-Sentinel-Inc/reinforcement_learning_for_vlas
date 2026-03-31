#!/usr/bin/env bash
# Trains the advantage-conditioned policy using is_good_action labels in the dataset.
# The dataset must already contain these labels (produced by vf_kfold_label.sh or vf_label.sh).
set -euo pipefail
cd "$(dirname "$0")/../.."

# ─── CONFIG ────────────────────────────────────────────────────────────────────
POLICY_CONFIG=pi06_rl_pretrain_airbot_clothes_folding
EXP_NAME=policy_iter0
GPUS=2,3,4,5,6,7
OVERWRITE=true   # true = start fresh; false = resume from last checkpoint
# ───────────────────────────────────────────────────────────────────────────────

mkdir -p logs
LOG_FILE="logs/train_policy_${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"

FLAG="--overwrite"
[[ "$OVERWRITE" == "false" ]] && FLAG="--resume"

CUDA_VISIBLE_DEVICES="${GPUS}" HF_LEROBOT_HOME=./lerobot_data \
    uv run scripts/train.py "${POLICY_CONFIG}" \
        --exp-name "${EXP_NAME}" \
        "${FLAG}"

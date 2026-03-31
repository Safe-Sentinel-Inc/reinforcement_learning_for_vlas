#!/usr/bin/env bash
# Trains a single value function on the full dataset (no cross-validation).
# Useful for quick experiments; for production use vf_kfold_train.sh + vf_kfold_label.sh.
set -euo pipefail
cd "$(dirname "$0")/../.."

# ─── CONFIG ────────────────────────────────────────────────────────────────────
VF_CONFIG=pi06_rl_vf_airbot_clothes_folding
EXP_NAME=vf_v1
GPUS=0,1,2,3
NUM_TRAIN_STEPS=20000
OVERWRITE=true   # true = start fresh; false = resume from last checkpoint
# ───────────────────────────────────────────────────────────────────────────────

mkdir -p logs
LOG_FILE="logs/train_vf_${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"

FLAG="--overwrite"
[[ "$OVERWRITE" == "false" ]] && FLAG="--resume"

CUDA_VISIBLE_DEVICES="${GPUS}" HF_LEROBOT_HOME=./lerobot_data \
    uv run scripts/train.py "${VF_CONFIG}" \
        --exp-name "${EXP_NAME}" \
        --num-train-steps "${NUM_TRAIN_STEPS}" \
        "${FLAG}"

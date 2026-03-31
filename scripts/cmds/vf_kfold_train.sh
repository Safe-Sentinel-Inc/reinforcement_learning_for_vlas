#!/usr/bin/env bash
# K-fold phase 1: trains K separate value functions concurrently, each one
# excluding a different fold so that every fold has a VF that never saw its data.
# Each fold is allocated a contiguous block of GPUs from the pool.
set -euo pipefail
cd "$(dirname "$0")/../.."

# ─── CONFIG ────────────────────────────────────────────────────────────────────
REPO_ID=Fold_clothes
VF_CONFIG=pi06_rl_vf_airbot_clothes_folding
GPUS=(2 3 4 5 6 7)      # Pool of GPU IDs available for training
NUM_FOLDS=3
NUM_TRAIN_STEPS=20000
GPUS_PER_FOLD=2          # Contiguous GPUs allocated to each fold
EXP_PREFIX=kfold
RESUME=false             # true = pick up from existing checkpoints; false = train from scratch
# ───────────────────────────────────────────────────────────────────────────────

mkdir -p logs
LOG_FILE="logs/kfold_train_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"
echo "===== K-fold Phase 1: Training ${NUM_FOLDS} VFs ====="

if [[ "$RESUME" == "true" ]]; then
    RESUME_FLAG="--resume"
else
    RESUME_FLAG="--overwrite"
fi

TRAIN_PIDS=()
for (( fold=0; fold<NUM_FOLDS; fold++ )); do
    gpu_start=$(( fold * GPUS_PER_FOLD ))
    fold_gpus=""
    for (( g=0; g<GPUS_PER_FOLD; g++ )); do
        idx=$(( gpu_start + g ))
        [[ -n "$fold_gpus" ]] && fold_gpus="${fold_gpus},"
        fold_gpus="${fold_gpus}${GPUS[$idx]}"
    done
    exp_name="${EXP_PREFIX}_fold${fold}"

    echo "[Fold ${fold}] GPUs ${fold_gpus}, exp=${exp_name} (${RESUME_FLAG})"

    CUDA_VISIBLE_DEVICES="${fold_gpus}" HF_LEROBOT_HOME=./lerobot_data \
        uv run scripts/train.py "${VF_CONFIG}" \
            --exp-name "${exp_name}" \
            --data.repo-id "${REPO_ID}" \
            --data.exclude-fold "${fold}" \
            --num-train-steps "${NUM_TRAIN_STEPS}" \
            "${RESUME_FLAG}" \
        > "logs/kfold_train_fold${fold}.log" 2>&1 &

    TRAIN_PIDS+=($!)
    echo "  PID: ${TRAIN_PIDS[-1]}, log: logs/kfold_train_fold${fold}.log"
done

echo "Waiting for all ${NUM_FOLDS} training jobs..."
for pid in "${TRAIN_PIDS[@]}"; do
    wait "$pid"
    status=$?
    if [[ $status -ne 0 ]]; then
        echo "ERROR: training PID ${pid} failed (exit ${status})" >&2
        exit 1
    fi
done

echo "===== K-fold Phase 1 COMPLETE ====="

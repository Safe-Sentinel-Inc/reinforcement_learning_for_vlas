#!/usr/bin/env bash
# K-fold phases 2 and 3: runs each value function on its held-out fold in parallel,
# then merges the per-fold predictions and writes the is_good_action column.
# Leave CHECKPOINT_STEP empty to automatically pick the latest checkpoint from fold 0.
set -euo pipefail
cd "$(dirname "$0")/../.."

# ─── CONFIG ────────────────────────────────────────────────────────────────────
REPO_ID=Fold_clothes
VF_CONFIG=pi06_rl_vf_airbot_clothes_folding
GPUS=(2 3 4 5 6 7)      # Pool of GPU IDs available for inference
NUM_FOLDS=3
GPUS_PER_FOLD=2
EXP_PREFIX=kfold
CHECKPOINT_STEP=        # Blank = auto-detect the most recent checkpoint
BATCH_SIZE=48
POSITIVE_FRACTION=0.3
GAMMA=0.98
VALUES_DIR=/tmp/vf_kfold_Fold_clothes
# ───────────────────────────────────────────────────────────────────────────────

mkdir -p logs
LOG_FILE="logs/kfold_eval_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"

# If no checkpoint step was given, find the highest-numbered one from fold 0
if [[ -z "$CHECKPOINT_STEP" ]]; then
    _base="checkpoints/${VF_CONFIG}/${EXP_PREFIX}_fold0"
    if [[ -d "$_base" ]]; then
        CHECKPOINT_STEP=$(ls -1 "$_base" | grep -E '^[0-9]+$' | sort -n | tail -1)
    fi
    [[ -z "$CHECKPOINT_STEP" ]] && { echo "ERROR: cannot detect checkpoint step" >&2; exit 1; }
fi
echo "Using checkpoint step: ${CHECKPOINT_STEP}"

# ── Phase 2: Run VF inference on each fold's held-out data in parallel ───────
echo ""
echo "===== Phase 2: VF inference ====="
mkdir -p "${VALUES_DIR}"

INFER_PIDS=()
for (( fold=0; fold<NUM_FOLDS; fold++ )); do
    gpu_start=$(( fold * GPUS_PER_FOLD ))
    fold_gpus=""
    for (( g=0; g<GPUS_PER_FOLD; g++ )); do
        idx=$(( gpu_start + g ))
        [[ -n "$fold_gpus" ]] && fold_gpus="${fold_gpus},"
        fold_gpus="${fold_gpus}${GPUS[$idx]}"
    done
    exp_name="${EXP_PREFIX}_fold${fold}"
    ckpt_dir="checkpoints/${VF_CONFIG}/${exp_name}/${CHECKPOINT_STEP}"

    echo "[Fold ${fold}] GPUs ${fold_gpus}, ckpt=${ckpt_dir}"

    CUDA_VISIBLE_DEVICES="${fold_gpus}" HF_LEROBOT_HOME=./lerobot_data \
        uv run scripts/add_returns_to_lerobot.py vf_label \
            --repo-id "${REPO_ID}" \
            --vf-config "${VF_CONFIG}" \
            --vf-checkpoint-dir "${ckpt_dir}" \
            --infer-fold "${fold}" \
            --batch-size "${BATCH_SIZE}" \
            --values-dir "${VALUES_DIR}" \
        > "logs/kfold_eval_fold${fold}.log" 2>&1 &

    INFER_PIDS+=($!)
    echo "  PID: ${INFER_PIDS[-1]}, log: logs/kfold_eval_fold${fold}.log"
done

echo "Waiting for all ${NUM_FOLDS} inference jobs..."
for pid in "${INFER_PIDS[@]}"; do
    wait "$pid"
    status=$?
    if [[ $status -ne 0 ]]; then
        echo "ERROR: inference PID ${pid} failed (exit ${status})" >&2
        exit 1
    fi
done
echo "===== Phase 2 COMPLETE ====="

# ── Phase 3: Combine fold predictions and compute advantage-based labels ─────
echo ""
echo "===== Phase 3: Merge values + compute is_good_action ====="

HF_LEROBOT_HOME=./lerobot_data uv run scripts/add_returns_to_lerobot.py vf_merge \
    --repo-id "${REPO_ID}" \
    --values-dir "${VALUES_DIR}" \
    --positive-fraction "${POSITIVE_FRACTION}" \
    --gamma "${GAMMA}"

echo "===== Phase 3 COMPLETE — is_good_action written to dataset ====="

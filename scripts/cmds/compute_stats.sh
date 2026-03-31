#!/usr/bin/env bash
# Calculates normalization statistics (mean, std, etc.) over the dataset
# for use during training and inference.
set -euo pipefail
cd "$(dirname "$0")/../.."

# ─── CONFIG ────────────────────────────────────────────────────────────────────
VF_CONFIG=pi06_rl_vf_airbot_clothes_folding
# ───────────────────────────────────────────────────────────────────────────────

mkdir -p logs
LOG_FILE="logs/compute_stats_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"

HF_LEROBOT_HOME=./lerobot_data uv run scripts/compute_norm_stats.py \
    --config-name "${VF_CONFIG}"

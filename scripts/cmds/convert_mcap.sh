#!/usr/bin/env bash
# Transforms raw MCAP teleoperation recordings into the LeRobot v2 dataset format.
# Supports incremental mode via --resume to add new episodes to an existing dataset.
set -euo pipefail
cd "$(dirname "$0")/../.."

# ─── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR=mcap_data/fold_clothes   # Path to the MCAP source directory (must contain a config.py)
RESUME=false                       # Set to true to incrementally add episodes; false rebuilds from scratch
# ───────────────────────────────────────────────────────────────────────────────

mkdir -p logs
LOG_FILE="logs/convert_mcap_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"

RESUME_FLAG=""
if [[ "$RESUME" == "true" ]]; then
    RESUME_FLAG="--resume"
fi

HF_LEROBOT_HOME=./lerobot_data uv run examples/robot/convert_mcap_to_lerobot.py \
    --data_dir "${DATA_DIR}" \
    ${RESUME_FLAG}

#!/usr/bin/env bash
# Launches a WebSocket server that hosts the trained policy for remote inference.
set -euo pipefail
cd "$(dirname "$0")/../.."

# ─── CONFIG ────────────────────────────────────────────────────────────────────
POLICY_CONFIG=pi06_rl_pretrain_airbot_clothes_folding
CHECKPOINT_DIR=checkpoints/pi06_rl_pretrain_airbot_clothes_folding/policy_iter0/XXXXX
PORT=8000
# ───────────────────────────────────────────────────────────────────────────────

mkdir -p logs
LOG_FILE="logs/serve_policy_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"

uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config "${POLICY_CONFIG}" \
    --policy.dir "${CHECKPOINT_DIR}" \
    --port "${PORT}"

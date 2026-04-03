#!/usr/bin/env bash
# Runs policy inference in synchronous mode, where each action chunk finishes
# executing before the next one is requested from the server.
# Best for debugging and validation; use infer_async.sh for real-time deployment.
set -euo pipefail
cd "$(dirname "$0")/../.."

# ─── CONFIG ────────────────────────────────────────────────────────────────────
HOST=127.0.0.1
PORT=8000
PROMPT="Fold clothes"
CHUNK_SIZE_EXECUTE=25
DAGGER=false            # Enable DAgger-style human intervention collection (needs leader arms)
# ───────────────────────────────────────────────────────────────────────────────

mkdir -p logs
LOG_FILE="logs/infer_sync_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"

cd examples/robot

python inference_sync.py \
    --policy-config.host "${HOST}" \
    --policy-config.port "${PORT}" \
    --prompt "${PROMPT}" \
    --chunk-size-execute "${CHUNK_SIZE_EXECUTE}" \
    --dagger.enable "${DAGGER}"

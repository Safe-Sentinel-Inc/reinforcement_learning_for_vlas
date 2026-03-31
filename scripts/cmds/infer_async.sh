#!/usr/bin/env bash
# Runs policy inference in asynchronous mode where model inference and robot
# execution happen concurrently. Uses Temporal Chunk Smoothing (TCS) to blend
# overlapping action chunks for smooth real-time behavior.
# Recommended for production deployment on real hardware.
set -euo pipefail
cd "$(dirname "$0")/../.."

# ─── CONFIG ────────────────────────────────────────────────────────────────────
HOST=127.0.0.1
PORT=8000
PROMPT="Fold clothes"
CHUNK_SIZE_EXECUTE=25

# Temporal Chunk Smoothing (TCS) settings:
# tcs_drop_max          Upper bound on stale leading steps to discard when a fresh
#                       chunk arrives (actual drop count equals min(latency, this value))
# tcs_min_overlap       Shortest blending window between the outgoing and incoming
#                       chunks; weights ramp linearly from old to new across this span
# initial_action_wait_s Seconds to hold the current pose at episode start while
#                       waiting for the first predicted action chunk
TCS_DROP_MAX=12
TCS_MIN_OVERLAP=8
INITIAL_ACTION_WAIT_S=10.0

RECORD=false            # Enable to write MCAP recordings during inference
RECORD_DIR=./inference_data
DAGGER=false            # Enable DAgger-style human intervention collection (needs leader arms)
# ───────────────────────────────────────────────────────────────────────────────

mkdir -p logs
LOG_FILE="logs/infer_async_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"

cd examples/robot

python inference_async.py \
    --policy-config.host "${HOST}" \
    --policy-config.port "${PORT}" \
    --prompt "${PROMPT}" \
    --chunk-size-execute "${CHUNK_SIZE_EXECUTE}" \
    --tcs-drop-max "${TCS_DROP_MAX}" \
    --tcs-min-overlap "${TCS_MIN_OVERLAP}" \
    --initial-action-wait-s "${INITIAL_ACTION_WAIT_S}" \
    --record.record-data "${RECORD}" \
    --record.save-dir "${RECORD_DIR}" \
    --dagger.enable "${DAGGER}"

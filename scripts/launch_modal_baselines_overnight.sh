#!/usr/bin/env bash
# ==============================================================================
# Parallel Modal Cloud GPU Launcher for Stage 1A Baselines (B1-B4) + --no_vicreg
# ==============================================================================

set -euo pipefail

LOG_DIR="${LOG_DIR:-logs/modal_overnight}"
SKIP_UPLOAD="${SKIP_UPLOAD:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

mkdir -p "$LOG_DIR"

if [[ "$SKIP_UPLOAD" != "1" && -d "data/baked_regions" ]]; then
  echo "Uploading data/baked_regions to Modal volume squared-sense-vol:/data/baked_regions..."
  uv run modal volume put squared-sense-vol data/baked_regions /data/baked_regions
fi

uv run modal run modal_app.py --action train-baselines --model b4_uniform32 --epochs 80 --batch-size 16 > "$LOG_DIR/b4_uniform32.log" 2>&1 &
uv run modal run modal_app.py --action train-baselines --model b1_lod0      --epochs 80 --batch-size 32 > "$LOG_DIR/b1_lod0.log"      2>&1 &
uv run modal run modal_app.py --action train-baselines --model b2_lod2      --epochs 80 --batch-size 32 > "$LOG_DIR/b2_lod2.log"      2>&1 &
uv run modal run modal_app.py --action train-baselines --model b3_mae       --epochs 80 --batch-size 32 > "$LOG_DIR/b3_mae.log"       2>&1 &
uv run modal run modal_app.py --action train-spatial   --epochs 20 --batch-size 32 --no-vicreg         > "$LOG_DIR/no_vicreg.log"    2>&1 &

wait
echo "All Modal GPU jobs finished. Logs in $LOG_DIR/"

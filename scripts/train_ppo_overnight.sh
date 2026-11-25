#!/usr/bin/env bash
# ==============================================================================
# Squared Sense: PPO Navigation Training Script
# Edit the variables below or pass CLI arguments directly to override them:
#   ./scripts/train_ppo_overnight.sh --total_steps 20000 --bc_epochs 10
# ==============================================================================

set -euo pipefail

# Default Configuration Variables
REGION_DIR="${REGION_DIR:-data/baked_regions}"
MODEL_TYPE="${MODEL_TYPE:-m1_full}"
CKPT="${CKPT:-checkpoints/spatial_finetuned_best.pkl}"
TEMPORAL_CKPT="${TEMPORAL_CKPT:-checkpoints/temporal_best.pkl}"
BC_EPOCHS="${BC_EPOCHS:-20}"
BC_STEPS="${BC_STEPS:-600}"
TOTAL_STEPS="${TOTAL_STEPS:-50000}"
SAVE_DIR="${SAVE_DIR:-checkpoints/ppo_nav_m1_overnight}"
DEVICE="${DEVICE:-auto}"

# Resolve repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

echo "======================================================="
echo " Squared Sense: PPO Navigation Training ($MODEL_TYPE)"
echo "======================================================="
echo "  Region Dir:    $REGION_DIR"
echo "  Model Type:    $MODEL_TYPE"
echo "  Spatial Ckpt:  $CKPT"
echo "  Temporal Ckpt: $TEMPORAL_CKPT"
echo "  BC Warm-Start: $BC_EPOCHS epochs ($BC_STEPS demo steps)"
echo "  Total Steps:   $TOTAL_STEPS"
echo "  Save Dir:      $SAVE_DIR"
echo "  Device:        $DEVICE"
if [ "$#" -gt 0 ]; then
  echo "  Extra Args:    $*"
fi
echo "======================================================="

exec uv run python -m eval.train_ppo_nav \
  --region_dir "$REGION_DIR" \
  --model_type "$MODEL_TYPE" \
  --ckpt "$CKPT" \
  --temporal_ckpt "$TEMPORAL_CKPT" \
  --bc_epochs "$BC_EPOCHS" \
  --bc_steps "$BC_STEPS" \
  --total_steps "$TOTAL_STEPS" \
  --save_dir "$SAVE_DIR" \
  --device "$DEVICE" \
  "$@"

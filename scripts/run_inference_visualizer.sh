#!/usr/bin/env bash
# ==============================================================================
# Squared Sense: 3D Visualizer Real-Time Inference Script
# Edit the variables below or pass CLI arguments directly to override them:
#   ./scripts/run_inference_visualizer.sh --episodes 20 --fps 15
# ==============================================================================

set -euo pipefail

REGION_DIR="${REGION_DIR:-data/baked_regions}"
MODEL_TYPE="${MODEL_TYPE:-m1_full}"
CKPT="${CKPT:-checkpoints/spatial_finetuned_best.pkl}"
TEMPORAL_CKPT="${TEMPORAL_CKPT:-checkpoints/temporal_best.pkl}"
POLICY_CKPT="${POLICY_CKPT:-checkpoints/ppo_nav_m1_overnight/best_ppo.pkl}"
EPISODES="${EPISODES:-10}"
FPS="${FPS:-10.0}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-auto}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

if [ ! -f "$POLICY_CKPT" ] && [ -f "checkpoints/ppo_nav/policy_best.pkl" ]; then
  echo "Warning: $POLICY_CKPT not found, falling back to checkpoints/ppo_nav/policy_best.pkl"
  POLICY_CKPT="checkpoints/ppo_nav/policy_best.pkl"
fi

echo "======================================================="
echo " Squared Sense: 3D Visualizer Real-Time Inference"
echo "======================================================="
echo "  Region Dir:    $REGION_DIR"
echo "  Model Type:    $MODEL_TYPE"
echo "  Policy Ckpt:   $POLICY_CKPT"
echo "  Spatial Ckpt:  $CKPT"
echo "  Temporal Ckpt: $TEMPORAL_CKPT"
echo "  Episodes:      $EPISODES"
echo "  Target FPS:    $FPS"
echo "  Device:        $DEVICE"
if [ "$#" -gt 0 ]; then
  echo "  Extra Args:    $*"
fi
echo "======================================================="

exec uv run python -m eval.run_inference \
  --region_dir "$REGION_DIR" \
  --model_type "$MODEL_TYPE" \
  --policy_ckpt "$POLICY_CKPT" \
  --ckpt "$CKPT" \
  --temporal_ckpt "$TEMPORAL_CKPT" \
  --episodes "$EPISODES" \
  --fps "$FPS" \
  --seed "$SEED" \
  --device "$DEVICE" \
  "$@"

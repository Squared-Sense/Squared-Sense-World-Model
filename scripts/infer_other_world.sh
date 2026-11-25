#!/usr/bin/env bash
# ==============================================================================
# Real-time 3D WebGL Visualizer Inference on OtherWorlds/<world_name>
# Usage:
#   ./scripts/infer_other_world.sh stony_peaks
#   MODEL_TYPE=m2_static TASK=rl2 ./scripts/infer_other_world.sh forest_biome
# ==============================================================================

set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "Usage: $0 <world_name>"
  exit 1
fi

WORLD_NAME="$1"
shift
MODEL_TYPE="${MODEL_TYPE:-m1_full}"
TASK="${TASK:-rl1}"
CKPT="${CKPT:-}"
TEMPORAL_CKPT="${TEMPORAL_CKPT:-}"
POLICY_CKPT="${POLICY_CKPT:-}"
EPISODES="${EPISODES:-10}"
FPS="${FPS:-10.0}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-auto}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

REGION_DIR="$REPO_ROOT/OtherWorlds/$WORLD_NAME/baked_regions"
if [[ ! -d "$REGION_DIR" ]]; then
  echo "Error: Baked regions not found at $REGION_DIR. Run ./OtherWorlds/bake_world.sh $WORLD_NAME first."
  exit 1
fi

if [[ -z "$CKPT" ]]; then
  if [[ "$MODEL_TYPE" == "m2_static" && -f "checkpoints/m2_spatial_jepa/best.pkl" ]]; then
    CKPT="checkpoints/m2_spatial_jepa/best.pkl"
  else
    CKPT="checkpoints/spatial_finetuned_best.pkl"
  fi
fi
if [[ "$MODEL_TYPE" == "m1_full" && -z "$TEMPORAL_CKPT" ]]; then
  TEMPORAL_CKPT="checkpoints/temporal_best.pkl"
fi
if [[ -z "$POLICY_CKPT" ]]; then
  WORLD_CKPT="checkpoints/other_worlds/${WORLD_NAME}_${MODEL_TYPE}_${TASK}/best_ppo.pkl"
  if [[ -f "$WORLD_CKPT" ]]; then
    POLICY_CKPT="$WORLD_CKPT"
  elif [[ "$MODEL_TYPE" == "m1_full" ]]; then
    POLICY_CKPT=$([[ "$TASK" == "rl2" ]] && echo "checkpoints/ppo_nav_m1_rl2/latest_ppo.pkl" || echo "checkpoints/ppo_nav_m1_overnight/best_ppo.pkl")
  else
    POLICY_CKPT=$([[ "$TASK" == "rl2" ]] && echo "checkpoints/ppo_nav_m2_rl2/best_ppo.pkl" || echo "checkpoints/ppo_nav_m2_overnight/best_ppo.pkl")
  fi
fi

CMD=(
  uv run python -m eval.run_inference
  --region_dir "$REGION_DIR"
  --model_type "$MODEL_TYPE"
  --task "$TASK"
  --policy_ckpt "$POLICY_CKPT"
  --ckpt "$CKPT"
  --episodes "$EPISODES"
  --fps "$FPS"
  --seed "$SEED"
  --device "$DEVICE"
)
[[ "$MODEL_TYPE" == "m1_full" && -n "$TEMPORAL_CKPT" ]] && CMD+=(--temporal_ckpt "$TEMPORAL_CKPT")

exec "${CMD[@]}" "$@"

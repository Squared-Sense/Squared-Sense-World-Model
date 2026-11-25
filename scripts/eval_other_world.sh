#!/usr/bin/env bash
# ==============================================================================
# Zero-Shot Eval or Frozen-Backbone PPO Fine-Tuning on OtherWorlds/<world_name>
# Usage:
#   ./scripts/eval_other_world.sh stony_peaks
#   MODE=both MODEL_TYPE=m2_static TASK=rl2 ./scripts/eval_other_world.sh forest_biome
# ==============================================================================

set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "Usage: $0 <world_name>"
  exit 1
fi

WORLD_NAME="$1"
MODE="${MODE:-eval}"
MODEL_TYPE="${MODEL_TYPE:-m1_full}"
TASK="${TASK:-rl1}"
BACKBONE_CKPT="${BACKBONE_CKPT:-}"
TEMPORAL_CKPT="${TEMPORAL_CKPT:-}"
POLICY_CKPT="${POLICY_CKPT:-}"
TOTAL_STEPS="${TOTAL_STEPS:-25000}"
NUM_EPISODES="${NUM_EPISODES:-20}"
SEED="${SEED:-1000}"
DEVICE="${DEVICE:-cpu}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

BAKED_DIR="$REPO_ROOT/OtherWorlds/$WORLD_NAME/baked_regions"
if [[ ! -d "$BAKED_DIR" ]]; then
  echo "Error: Baked regions not found at $BAKED_DIR. Run ./OtherWorlds/bake_world.sh $WORLD_NAME first."
  exit 1
fi

if [[ -z "$BACKBONE_CKPT" ]]; then
  if [[ "$MODEL_TYPE" == "m2_static" && -f "checkpoints/m2_spatial_jepa/best.pkl" ]]; then
    BACKBONE_CKPT="checkpoints/m2_spatial_jepa/best.pkl"
  else
    BACKBONE_CKPT="checkpoints/spatial_finetuned_best.pkl"
  fi
fi
if [[ "$MODEL_TYPE" == "m1_full" && -z "$TEMPORAL_CKPT" ]]; then
  TEMPORAL_CKPT="checkpoints/temporal_best.pkl"
fi
if [[ -z "$POLICY_CKPT" ]]; then
  if [[ "$MODEL_TYPE" == "m1_full" ]]; then
    POLICY_CKPT=$([[ "$TASK" == "rl2" ]] && echo "checkpoints/ppo_nav_m1_rl2/latest_ppo.pkl" || echo "checkpoints/ppo_nav_m1_overnight/best_ppo.pkl")
  else
    POLICY_CKPT=$([[ "$TASK" == "rl2" ]] && echo "checkpoints/ppo_nav_m2_rl2/best_ppo.pkl" || echo "checkpoints/ppo_nav_m2_overnight/best_ppo.pkl")
  fi
fi

TRAIN_SAVE_DIR="checkpoints/other_worlds/${WORLD_NAME}_${MODEL_TYPE}_${TASK}"
OUT_JSON="results/other_worlds/${WORLD_NAME}_${MODEL_TYPE}_${TASK}.json"
mkdir -p "results/other_worlds"

if [[ "$MODE" == "train" || "$MODE" == "both" ]]; then
  CMD=(uv run python -m eval.train_ppo_nav --region_dir "$BAKED_DIR" --model_type "$MODEL_TYPE" --ckpt "$BACKBONE_CKPT" --task "$TASK" --lr 1e-4 --bc_epochs 0 --total_steps "$TOTAL_STEPS" --save_dir "$TRAIN_SAVE_DIR" --device "$DEVICE")
  [[ "$MODEL_TYPE" == "m1_full" && -n "$TEMPORAL_CKPT" ]] && CMD+=(--temporal_ckpt "$TEMPORAL_CKPT")
  [[ -n "$POLICY_CKPT" && -f "$POLICY_CKPT" ]] && CMD+=(--init_policy "$POLICY_CKPT")
  "${CMD[@]}"
  POLICY_CKPT="$TRAIN_SAVE_DIR/best_ppo.pkl"
fi

if [[ "$MODE" == "eval" || "$MODE" == "both" ]]; then
  CMD=(uv run python -m eval.eval_navigation --region_dir "$BAKED_DIR" --model_type "$MODEL_TYPE" --backbone_ckpt "$BACKBONE_CKPT" --policy_ckpt "$POLICY_CKPT" --task "$TASK" --num_episodes "$NUM_EPISODES" --seed "$SEED" --out_json "$OUT_JSON")
  [[ "$MODEL_TYPE" == "m1_full" && -n "$TEMPORAL_CKPT" ]] && CMD+=(--temporal_ckpt "$TEMPORAL_CKPT")
  "${CMD[@]}"
fi

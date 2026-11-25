#!/usr/bin/env bash
# ==============================================================================
# Squared Sense: RL2 (5-Target Chained + Burst Occlusion) PPO Training Script
# ==============================================================================

set -euo pipefail

REGION_DIR="${REGION_DIR:-data/baked_regions}"
MODEL_TYPE="${MODEL_TYPE:-m1_full}"
CKPT="${CKPT:-checkpoints/spatial_finetuned_best.pkl}"
TEMPORAL_CKPT="${TEMPORAL_CKPT:-checkpoints/temporal_best.pkl}"
INIT_POLICY="${INIT_POLICY:-checkpoints/ppo_nav_m1_overnight/best_ppo.pkl}"
NUM_TARGETS="${NUM_TARGETS:-5}"
COMPASS_OCCLUSION="${COMPASS_OCCLUSION:-0.5}"
OCCLUSION_BURST="${OCCLUSION_BURST:-6}"
LR="${LR:-1e-4}"
BC_EPOCHS="${BC_EPOCHS:-0}"
TOTAL_STEPS="${TOTAL_STEPS:-25000}"
SAVE_DIR="${SAVE_DIR:-checkpoints/ppo_nav_m1_rl2}"
DEVICE="${DEVICE:-auto}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

CMD=(
  uv run python -m eval.train_ppo_nav
  --region_dir "$REGION_DIR"
  --model_type "$MODEL_TYPE"
  --ckpt "$CKPT"
  --task rl2
  --num_targets "$NUM_TARGETS"
  --compass_occlusion "$COMPASS_OCCLUSION"
  --occlusion_burst "$OCCLUSION_BURST"
  --lr "$LR"
  --bc_epochs "$BC_EPOCHS"
  --total_steps "$TOTAL_STEPS"
  --save_dir "$SAVE_DIR"
  --device "$DEVICE"
)

if [[ "$MODEL_TYPE" == "m1_full" && -n "$TEMPORAL_CKPT" ]]; then
  CMD+=(--temporal_ckpt "$TEMPORAL_CKPT")
fi
if [[ -n "$INIT_POLICY" && -f "$INIT_POLICY" ]]; then
  CMD+=(--init_policy "$INIT_POLICY")
fi

exec "${CMD[@]}" "$@"

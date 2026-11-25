#!/usr/bin/env bash
# ==============================================================================
# Overnight IRIS 27 Evaluation & RL2 Pipeline (Bash)
# Usage:
#   ./scripts/run_iris_overnight.sh
#   RUN_RL2=1 ./scripts/run_iris_overnight.sh
# ==============================================================================

set -euo pipefail

REGION_DIR="${REGION_DIR:-data/baked_regions}"
RESULTS_DIR="${RESULTS_DIR:-results}"
LOG_FILE="${LOG_FILE:-logs/iris_overnight.log}"
RUN_RL2="${RUN_RL2:-0}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
SEED="${SEED:-1000}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

mkdir -p "$RESULTS_DIR" "$(dirname "$LOG_FILE")"

run_step() {
  local name="$1"
  shift
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] === START: $name ===" | tee -a "$LOG_FILE"
  "$@" 2>&1 | tee -a "$LOG_FILE"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] === DONE: $name ===" | tee -a "$LOG_FILE"
}

# 1. Temporal Dynamics & Pure Ego-Shift Ablation (M1)
if [[ -f "checkpoints/spatial_finetuned_best.pkl" && -f "checkpoints/temporal_best.pkl" ]]; then
  run_step "Temporal Dynamics & Ego-Shift Ablation (M1)" \
    uv run python -m eval.eval_temporal_dynamics \
      --spatial_ckpt checkpoints/spatial_finetuned_best.pkl \
      --temporal_ckpt checkpoints/temporal_best.pkl \
      --region_dir "$REGION_DIR" \
      --out_json "$RESULTS_DIR/temporal_dynamics_ablation.json"
fi

# 2. Event-Driven Shell Gating Pareto Sweep (M1 RL1)
if [[ -f "checkpoints/ppo_nav_m1_overnight/best_ppo.pkl" ]]; then
  run_step "Event-Driven Shell Gating Pareto Sweep (M1)" \
    uv run python -m eval.eval_gating_pareto \
      --policy_ckpt checkpoints/ppo_nav_m1_overnight/best_ppo.pkl \
      --backbone_ckpt checkpoints/spatial_finetuned_best.pkl \
      --temporal_ckpt checkpoints/temporal_best.pkl \
      --region_dir "$REGION_DIR" \
      --num_episodes "$EVAL_EPISODES" \
      --seed "$SEED" \
      --out_json "$RESULTS_DIR/gating_pareto.json"
fi

# 3. Few-Shot Probe Sweep across available checkpoints
run_probe_if_exists() {
  local mtype="$1" ckpt="$2" temp="${3:-}"
  if [[ -f "$ckpt" ]]; then
    local cmd=(uv run python -m eval.probe_suite --model_type "$mtype" --ckpt "$ckpt" --region_dir "$REGION_DIR" --train_fractions 0.1 0.25 1.0 --out_json "$RESULTS_DIR/probe_${mtype}_fewshot.json")
    if [[ -n "$temp" && -f "$temp" ]]; then
      cmd+=(--temporal_ckpt "$temp")
    fi
    run_step "Few-Shot Probes ($mtype)" "${cmd[@]}"
  fi
}

run_probe_if_exists "m1_full" "checkpoints/spatial_finetuned_best.pkl" "checkpoints/temporal_best.pkl"
run_probe_if_exists "m2_static" "checkpoints/m2_spatial_jepa/best.pkl"
run_probe_if_exists "b1_lod0" "checkpoints/baselines/b1_lod0/best.pkl"
run_probe_if_exists "b2_lod2" "checkpoints/baselines/b2_lod2/best.pkl"
run_probe_if_exists "b3_mae" "checkpoints/baselines/b3_mae/best.pkl"
run_probe_if_exists "b4_uniform32" "checkpoints/baselines/b4_uniform32/best.pkl"

# 4. Optional RL2 Training + Zero-Shot Evaluation
if [[ "$RUN_RL2" == "1" ]]; then
  if [[ -f "checkpoints/ppo_nav_m1_overnight/best_ppo.pkl" ]]; then
    REGION_DIR="$REGION_DIR" MODEL_TYPE="m1_full" INIT_POLICY="checkpoints/ppo_nav_m1_overnight/best_ppo.pkl" SAVE_DIR="checkpoints/ppo_nav_m1_rl2" \
      "$SCRIPT_DIR/train_ppo_rl2.sh"
    run_step "RL2 Zero-Shot Eval (M1)" \
      uv run python -m eval.eval_navigation \
        --model_type m1_full \
        --backbone_ckpt checkpoints/spatial_finetuned_best.pkl \
        --temporal_ckpt checkpoints/temporal_best.pkl \
        --policy_ckpt checkpoints/ppo_nav_m1_rl2/best_ppo.pkl \
        --region_dir "$REGION_DIR" \
        --task rl2 \
        --num_episodes "$EVAL_EPISODES" \
        --seed "$SEED" \
        --out_json "$RESULTS_DIR/nav_m1_rl2.json"
  fi
  if [[ -f "checkpoints/ppo_nav_m2_overnight/best_ppo.pkl" && -f "checkpoints/m2_spatial_jepa/best.pkl" ]]; then
    REGION_DIR="$REGION_DIR" MODEL_TYPE="m2_static" CKPT="checkpoints/m2_spatial_jepa/best.pkl" INIT_POLICY="checkpoints/ppo_nav_m2_overnight/best_ppo.pkl" SAVE_DIR="checkpoints/ppo_nav_m2_rl2" \
      "$SCRIPT_DIR/train_ppo_rl2.sh"
    run_step "RL2 Zero-Shot Eval (M2)" \
      uv run python -m eval.eval_navigation \
        --model_type m2_static \
        --backbone_ckpt checkpoints/m2_spatial_jepa/best.pkl \
        --policy_ckpt checkpoints/ppo_nav_m2_rl2/best_ppo.pkl \
        --region_dir "$REGION_DIR" \
        --task rl2 \
        --num_episodes "$EVAL_EPISODES" \
        --seed "$SEED" \
        --out_json "$RESULTS_DIR/nav_m2_rl2.json"
  fi
fi

echo "Overnight IRIS 27 pipeline complete! Outputs in $RESULTS_DIR"

#!/usr/bin/env bash
# ==============================================================================
# CPU Parallel Overnight Runner: M1 RL2, M2 RL1->RL2, and Zero-Retraining Evals
# ==============================================================================

set -euo pipefail

REGION_DIR="${REGION_DIR:-data/baked_regions}"
RESULTS_DIR="${RESULTS_DIR:-results}"
LOG_DIR="${LOG_DIR:-logs/cpu_overnight}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
SEED="${SEED:-1000}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

# Chain A: M1 RL2 (25k steps)
M1_INIT=()
if [[ -f "checkpoints/ppo_nav_m1_overnight/best_ppo.pkl" ]]; then
  M1_INIT=(--init_policy checkpoints/ppo_nav_m1_overnight/best_ppo.pkl)
fi
uv run python -m eval.train_ppo_nav \
  --region_dir "$REGION_DIR" \
  --model_type m1_full \
  --ckpt checkpoints/spatial_finetuned_best.pkl \
  --temporal_ckpt checkpoints/temporal_best.pkl \
  "${M1_INIT[@]}" \
  --task rl2 --num_targets 5 --compass_occlusion 0.5 --occlusion_burst 6 \
  --lr 1e-4 --bc_epochs 0 --total_steps 25000 \
  --save_dir checkpoints/ppo_nav_m1_rl2 --device cpu > "$LOG_DIR/m1_rl2.log" 2>&1 &
PID_M1=$!

# Chain B: M2 RL1 (50k steps) -> M2 RL2 (25k steps)
(
  uv run python -m eval.train_ppo_nav \
    --region_dir "$REGION_DIR" \
    --model_type m2_static \
    --ckpt checkpoints/m2_spatial_jepa/best.pkl \
    --task rl1 --bc_epochs 20 --bc_steps 600 --total_steps 50000 \
    --save_dir checkpoints/ppo_nav_m2_overnight --device cpu
  uv run python -m eval.train_ppo_nav \
    --region_dir "$REGION_DIR" \
    --model_type m2_static \
    --ckpt checkpoints/m2_spatial_jepa/best.pkl \
    --init_policy checkpoints/ppo_nav_m2_overnight/best_ppo.pkl \
    --task rl2 --num_targets 5 --compass_occlusion 0.5 --occlusion_burst 6 \
    --lr 1e-4 --bc_epochs 0 --total_steps 25000 \
    --save_dir checkpoints/ppo_nav_m2_rl2 --device cpu
) > "$LOG_DIR/m2_chain.log" 2>&1 &
PID_M2=$!

_pct() {
  local f="$1" tgt="$2"
  if [[ ! -f "$f" ]]; then echo 0; return; fi
  local n; n=$(wc -l < "$f" 2>/dev/null || echo 0)
  echo $(( n * 100 / tgt > 100 ? 100 : n * 100 / tgt ))
}

_bar() {
  local pct="$1" w=18
  local fill=$(( pct * w / 100 ))
  printf '%*s' "$fill" '' | tr ' ' '='
  printf '%*s' "$(( w - fill ))" '' | tr ' ' '-'
}

T_START=$(date +%s)
while kill -0 "$PID_M1" 2>/dev/null || kill -0 "$PID_M2" 2>/dev/null; do
  P1=$(kill -0 "$PID_M1" 2>/dev/null && _pct checkpoints/ppo_nav_m1_rl2/ppo_metrics.jsonl 195 || echo 100)
  P2_1=$(_pct checkpoints/ppo_nav_m2_overnight/ppo_metrics.jsonl 390)
  P2_2=$(_pct checkpoints/ppo_nav_m2_rl2/ppo_metrics.jsonl 195)
  if ! kill -0 "$PID_M2" 2>/dev/null; then P2=100; elif (( P2_2 > 0 )); then P2=$(( 67 + P2_2 / 3 )); else P2=$(( P2_1 * 2 / 3 )); fi
  ALL=$(( (P1 + P2) / 2 ))
  ELAP=$(( $(date +%s) - T_START ))
  COLS=$(tput cols 2>/dev/null || echo 100)
  LINE=$(printf "[%02d:%02d:%02d] [%s] %3d%% | M1-RL2:%3d%% | M2:%3d%%" \
    $((ELAP/3600)) $(((ELAP%3600)/60)) $((ELAP%60)) "$(_bar "$ALL")" "$ALL" "$P1" "$P2")
  printf "\r\033[K%s" "${LINE:0:$((COLS-1))}"
  sleep 2
done
printf "\r\033[K"
wait $PID_M1 $PID_M2

# Phase 2: All Zero-Retraining Evaluations + RL2 Zero-Shot Nav Eval
REGION_DIR="$REGION_DIR" RESULTS_DIR="$RESULTS_DIR" LOG_FILE="$LOG_DIR/evals.log" EVAL_EPISODES="$EVAL_EPISODES" SEED="$SEED" \
  "$SCRIPT_DIR/run_iris_overnight.sh"

if [[ -f "checkpoints/ppo_nav_m1_rl2/best_ppo.pkl" ]]; then
  uv run python -m eval.eval_navigation \
    --model_type m1_full \
    --backbone_ckpt checkpoints/spatial_finetuned_best.pkl \
    --temporal_ckpt checkpoints/temporal_best.pkl \
    --policy_ckpt checkpoints/ppo_nav_m1_rl2/best_ppo.pkl \
    --region_dir "$REGION_DIR" --task rl2 --num_episodes "$EVAL_EPISODES" --seed "$SEED" \
    --out_json "$RESULTS_DIR/nav_m1_rl2.json" | tee -a "$LOG_DIR/evals.log"
fi

if [[ -f "checkpoints/ppo_nav_m2_rl2/best_ppo.pkl" ]]; then
  uv run python -m eval.eval_navigation \
    --model_type m2_static \
    --backbone_ckpt checkpoints/m2_spatial_jepa/best.pkl \
    --policy_ckpt checkpoints/ppo_nav_m2_rl2/best_ppo.pkl \
    --region_dir "$REGION_DIR" --task rl2 --num_episodes "$EVAL_EPISODES" --seed "$SEED" \
    --out_json "$RESULTS_DIR/nav_m2_rl2.json" | tee -a "$LOG_DIR/evals.log"
fi

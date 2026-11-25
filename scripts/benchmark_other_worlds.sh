#!/usr/bin/env bash
# ==============================================================================
# Full OtherWorlds Cross-Biome Benchmarking Pipeline + Figure Generation
# ==============================================================================

set -euo pipefail

NUM_EPISODES="${NUM_EPISODES:-20}"
SEED="${SEED:-1000}"
LOG_FILE="${LOG_FILE:-logs/other_worlds_benchmark.log}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

mkdir -p "$(dirname "$LOG_FILE")"

for baked in OtherWorlds/*/baked_regions; do
  [[ -d "$baked" ]] || continue
  world="$(basename "$(dirname "$baked")")"
  for model in m1_full m2_static; do
    for task in rl1 rl2; do
      echo "=== Benchmarking $world | $model | $task ===" | tee -a "$LOG_FILE"
      MODEL_TYPE="$model" TASK="$task" NUM_EPISODES="$NUM_EPISODES" SEED="$SEED" \
        "$SCRIPT_DIR/eval_other_world.sh" "$world" 2>&1 | tee -a "$LOG_FILE"
    done
  done
done

uv run python -m eval.summarize_other_worlds 2>&1 | tee -a "$LOG_FILE"

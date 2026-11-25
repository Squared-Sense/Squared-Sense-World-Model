#!/usr/bin/env bash
# ==============================================================================
# Squared Sense World Model — All-in-One GPU Training, Benchmarking & Archival
# Designed for Kaggle (2x T4 GPU) or any Linux CUDA machine.
#
# Logs all training metrics (*.jsonl, train.log) and benchmark results (*.json)
# into checkpoints/ and results/, then packages everything into a single .zip.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

REGION_DIR="${BAKED_DIR:-${REGION_DIR:-data/baked_regions}}"
EPISODES_DIR="${EPISODES_DIR:-data/episodes}"
RESULTS_DIR="${RESULTS_DIR:-results}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
SEED="${SEED:-1000}"
PPO_STEPS_RL1="${PPO_STEPS_RL1:-50000}"
PPO_STEPS_RL2_4="${PPO_STEPS_RL2_4:-25000}"
OUT_ZIP="${OUT_ZIP:-/kaggle/working/squared_sense_master_artifacts}"

mkdir -p "$RESULTS_DIR" checkpoints

echo "======================================================="
echo " Squared Sense: GPU All-in-One Training & Benchmarks"
echo "======================================================="
uv run python -c "import jax; print('JAX backend:', jax.default_backend(), '| devices:', jax.devices())"
echo "Region Dir:   $REGION_DIR"
echo "Episodes Dir: $EPISODES_DIR"
echo "Results Dir:  $RESULTS_DIR"
echo "======================================================="

# ------------------------------------------------------------------------------
# 1. Pre-trained Backbone Checkpoints (auto-detect or train if missing)
# ------------------------------------------------------------------------------
M2_CKPT="checkpoints/m2_spatial_jepa/best.pkl"
M1_SPATIAL_CKPT="checkpoints/m1_full_temporal/spatial_finetuned_best.pkl"
M1_TEMPORAL_CKPT="checkpoints/m1_full_temporal/temporal_best.pkl"

if [[ ! -f "$M2_CKPT" && -f "checkpoints/spatial_finetuned_best.pkl" ]]; then
    M2_CKPT="checkpoints/spatial_finetuned_best.pkl"
fi
if [[ ! -f "$M1_SPATIAL_CKPT" && -f "checkpoints/spatial_finetuned_best.pkl" ]]; then
    M1_SPATIAL_CKPT="checkpoints/spatial_finetuned_best.pkl"
fi
if [[ ! -f "$M1_TEMPORAL_CKPT" && -f "checkpoints/temporal_best.pkl" ]]; then
    M1_TEMPORAL_CKPT="checkpoints/temporal_best.pkl"
fi

# ------------------------------------------------------------------------------
# 2. Train Navigation Policies (B0, M1, M2 across RL1, RL2, RL3, RL4)
# ------------------------------------------------------------------------------
echo "[1/4] Training B0 (b0_blind) on RL1 ($PPO_STEPS_RL1 steps)..."
uv run python -m eval.train_ppo_nav \
    --model_type b0_blind \
    --region_dir "$REGION_DIR" \
    --task rl1 \
    --bc_epochs 20 \
    --total_steps "$PPO_STEPS_RL1" \
    --save_dir checkpoints/ppo_nav_b0_rl1 \
    --device gpu

if [[ -f "$M1_SPATIAL_CKPT" && -f "$M1_TEMPORAL_CKPT" ]]; then
    echo "[1/4] Training M1 (m1_full) RL1 -> RL2 -> RL3 -> RL4..."
    uv run python -m eval.train_ppo_nav \
        --model_type m1_full \
        --ckpt "$M1_SPATIAL_CKPT" \
        --temporal_ckpt "$M1_TEMPORAL_CKPT" \
        --region_dir "$REGION_DIR" \
        --task rl1 \
        --bc_epochs 20 \
        --total_steps "$PPO_STEPS_RL1" \
        --save_dir checkpoints/ppo_nav_m1_rl1 \
        --device gpu

    for t in rl2 rl3_beacon rl3_flash rl4_beacon rl4_flash; do
        uv run python -m eval.train_ppo_nav \
            --model_type m1_full \
            --ckpt "$M1_SPATIAL_CKPT" \
            --temporal_ckpt "$M1_TEMPORAL_CKPT" \
            --init_policy checkpoints/ppo_nav_m1_rl1/best_ppo.pkl \
            --region_dir "$REGION_DIR" \
            --task "$t" \
            --lr 1e-4 \
            --bc_epochs 0 \
            --total_steps "$PPO_STEPS_RL2_4" \
            --save_dir "checkpoints/ppo_nav_m1_${t}" \
            --device gpu
    done
fi

if [[ -f "$M2_CKPT" ]]; then
    echo "[1/4] Training M2 (m2_static) RL1 -> RL2 -> RL3 -> RL4..."
    uv run python -m eval.train_ppo_nav \
        --model_type m2_static \
        --ckpt "$M2_CKPT" \
        --region_dir "$REGION_DIR" \
        --task rl1 \
        --bc_epochs 20 \
        --total_steps "$PPO_STEPS_RL1" \
        --save_dir checkpoints/ppo_nav_m2_rl1 \
        --device gpu

    for t in rl2 rl3_beacon rl3_flash rl4_beacon rl4_flash; do
        uv run python -m eval.train_ppo_nav \
            --model_type m2_static \
            --ckpt "$M2_CKPT" \
            --init_policy checkpoints/ppo_nav_m2_rl1/best_ppo.pkl \
            --region_dir "$REGION_DIR" \
            --task "$t" \
            --lr 1e-4 \
            --bc_epochs 0 \
            --total_steps "$PPO_STEPS_RL2_4" \
            --save_dir "checkpoints/ppo_nav_m2_${t}" \
            --device gpu
    done
fi

# ------------------------------------------------------------------------------
# 3. Frozen Probe Suite & Gating Pareto Benchmarks -> JSON
# ------------------------------------------------------------------------------
echo "[2/4] Running Probe Suite & Gating Pareto..."
if [[ -f "$M1_SPATIAL_CKPT" && -f "$M1_TEMPORAL_CKPT" ]]; then
    uv run python -m eval.probe_suite \
        --model_type m1_full \
        --ckpt "$M1_SPATIAL_CKPT" \
        --temporal_ckpt "$M1_TEMPORAL_CKPT" \
        --data_dir "$EPISODES_DIR" \
        --region_dir "$REGION_DIR" \
        --out_json "$RESULTS_DIR/probe_m1_full.json"

    uv run python -m eval.eval_gating_pareto \
        --model_type m1_full \
        --backbone_ckpt "$M1_SPATIAL_CKPT" \
        --temporal_ckpt "$M1_TEMPORAL_CKPT" \
        --region_dir "$REGION_DIR" \
        --out_json "$RESULTS_DIR/gating_pareto.json"
fi

if [[ -f "$M2_CKPT" ]]; then
    uv run python -m eval.probe_suite \
        --model_type m2_static \
        --ckpt "$M2_CKPT" \
        --data_dir "$EPISODES_DIR" \
        --region_dir "$REGION_DIR" \
        --out_json "$RESULTS_DIR/probe_m2_static.json"
fi

for b in b1_lod0 b2_lod2 b3_mae; do
    b_ckpt="checkpoints/baselines/${b}/best.pkl"
    if [[ -f "$b_ckpt" ]]; then
        uv run python -m eval.probe_suite \
            --model_type "$b" \
            --ckpt "$b_ckpt" \
            --data_dir "$EPISODES_DIR" \
            --region_dir "$REGION_DIR" \
            --out_json "$RESULTS_DIR/probe_${b}.json"
    fi
done

# ------------------------------------------------------------------------------
# 4. Navigation Benchmarks (RL1..RL4) -> JSON + Unified JSONL Manifest
# ------------------------------------------------------------------------------
echo "[3/4] Running RL1-RL4 Navigation Benchmarks..."
ALL_TASKS=(rl1 rl2 rl3_beacon rl3_flash rl4_beacon rl4_flash)

for t in "${ALL_TASKS[@]}"; do
    # B0 (always evaluated from its RL1 policy)
    if [[ -f "checkpoints/ppo_nav_b0_rl1/best_ppo.pkl" ]]; then
        uv run python -m eval.eval_navigation \
            --model_type b0_blind \
            --policy_ckpt checkpoints/ppo_nav_b0_rl1/best_ppo.pkl \
            --region_dir "$REGION_DIR" \
            --task "$t" \
            --num_episodes "$EVAL_EPISODES" \
            --seed "$SEED" \
            --out_json "$RESULTS_DIR/nav_b0_${t}.json"
    fi

    # M1
    m1_pol="checkpoints/ppo_nav_m1_${t}/best_ppo.pkl"
    [[ ! -f "$m1_pol" ]] && m1_pol="checkpoints/ppo_nav_m1_rl1/best_ppo.pkl"
    if [[ -f "$m1_pol" && -f "$M1_SPATIAL_CKPT" && -f "$M1_TEMPORAL_CKPT" ]]; then
        uv run python -m eval.eval_navigation \
            --model_type m1_full \
            --backbone_ckpt "$M1_SPATIAL_CKPT" \
            --temporal_ckpt "$M1_TEMPORAL_CKPT" \
            --policy_ckpt "$m1_pol" \
            --region_dir "$REGION_DIR" \
            --task "$t" \
            --num_episodes "$EVAL_EPISODES" \
            --seed "$SEED" \
            --out_json "$RESULTS_DIR/nav_m1_${t}.json"
    fi

    # M2
    m2_pol="checkpoints/ppo_nav_m2_${t}/best_ppo.pkl"
    [[ ! -f "$m2_pol" ]] && m2_pol="checkpoints/ppo_nav_m2_rl1/best_ppo.pkl"
    if [[ -f "$m2_pol" && -f "$M2_CKPT" ]]; then
        uv run python -m eval.eval_navigation \
            --model_type m2_static \
            --backbone_ckpt "$M2_CKPT" \
            --policy_ckpt "$m2_pol" \
            --region_dir "$REGION_DIR" \
            --task "$t" \
            --num_episodes "$EVAL_EPISODES" \
            --seed "$SEED" \
            --out_json "$RESULTS_DIR/nav_m2_${t}.json"
    fi
done

# Aggregate all eval JSONs into a single JSONL log
uv run python - <<'PY'
import glob, json, os
out_jsonl = os.path.join(os.environ.get("RESULTS_DIR", "results"), "all_benchmarks.jsonl")
files = sorted(glob.glob(os.path.join(os.environ.get("RESULTS_DIR", "results"), "*.json")))
with open(out_jsonl, "w", encoding="utf-8") as fout:
    for fp in files:
        with open(fp, "r", encoding="utf-8") as fin:
            data = json.load(fin)
        data["_file"] = os.path.basename(fp)
        fout.write(json.dumps(data) + "\n")
print(f"Aggregated {len(files)} JSON result files into {out_jsonl}")
PY

# ------------------------------------------------------------------------------
# 5. Package Checkpoints + Results + Figures into Master Zip
# ------------------------------------------------------------------------------
echo "[4/4] Packaging weights, JSON/JSONL logs, and figures into ${OUT_ZIP}.zip..."
uv run python - <<'PY'
import os, shutil
export_dir = "/tmp/squared_sense_export_staging"
os.makedirs(export_dir, exist_ok=True)
for d in ["checkpoints", "results", "paper_figs", "eval"]:
    if os.path.exists(d):
        shutil.copytree(d, os.path.join(export_dir, d), dirs_exist_ok=True)
out_base = os.environ.get("OUT_ZIP", "/kaggle/working/squared_sense_master_artifacts")
os.makedirs(os.path.dirname(out_base) or ".", exist_ok=True)
zip_path = shutil.make_archive(out_base, "zip", export_dir)
print(f"Created archive: {zip_path} ({os.path.getsize(zip_path) / 1e6:.2f} MB)")
PY

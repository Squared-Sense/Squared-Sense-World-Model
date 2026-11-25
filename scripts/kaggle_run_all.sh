#!/usr/bin/env bash
# ==============================================================================
# Squared Sense World Model — All Baselines (B0, B1, B2, B3, B4, no_vicreg)
# GPU Training, Probing, Navigation Benchmarks & Zip Archival for Kaggle
#
# Logs all training metrics (*.jsonl, train.log) and benchmark results (*.json,
# results/all_baselines.jsonl) into checkpoints/ and results/, then packages
# everything into /kaggle/working/squared_sense_master_artifacts.zip.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

REGION_DIR="${BAKED_DIR:-${REGION_DIR:-data/baked_regions}}"
EPISODES_DIR="${EPISODES_DIR:-data/episodes}"
RESULTS_DIR="${RESULTS_DIR:-results}"
EPOCHS="${EPOCHS:-80}"
NO_VICREG_EPOCHS="${NO_VICREG_EPOCHS:-20}"
PPO_STEPS="${PPO_STEPS:-25000}"
B0_PPO_STEPS="${B0_PPO_STEPS:-50000}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
SEED="${SEED:-35724444}"
OUT_ZIP="${OUT_ZIP:-/kaggle/working/squared_sense_master_artifacts}"

mkdir -p "$RESULTS_DIR" checkpoints/baselines

echo "======================================================="
echo " Squared Sense: All Baselines (B0-B4 + no_vicreg)"
echo "======================================================="
uv run python -c "import jax; print('JAX backend:', jax.default_backend(), '| devices:', jax.devices())"
echo "Region Dir:   $REGION_DIR"
echo "Episodes Dir: $EPISODES_DIR"
echo "Results Dir:  $RESULTS_DIR"
echo "Epochs:       $EPOCHS (no_vicreg: $NO_VICREG_EPOCHS)"
echo "PPO Steps:    B0=$B0_PPO_STEPS | B1-B3=$PPO_STEPS"
echo "Seed:         $SEED"
echo "======================================================="

# ------------------------------------------------------------------------------
# 1. Train Representation Baselines (B4, B1, B2, B3, no_vicreg) on GPU
# ------------------------------------------------------------------------------
echo "[1/4] Training B4 (b4_uniform32, batch_size=16, $EPOCHS epochs)..."
uv run python -m train.train_baselines \
    --model b4_uniform32 \
    --data_dir "$EPISODES_DIR" \
    --region_dir "$REGION_DIR" \
    --save_dir checkpoints/baselines/b4_uniform32 \
    --epochs "$EPOCHS" \
    --batch_size 16

echo "[1/4] Training B1 (b1_lod0, batch_size=32, $EPOCHS epochs)..."
uv run python -m train.train_baselines \
    --model b1_lod0 \
    --data_dir "$EPISODES_DIR" \
    --region_dir "$REGION_DIR" \
    --save_dir checkpoints/baselines/b1_lod0 \
    --epochs "$EPOCHS" \
    --batch_size 32

echo "[1/4] Training B2 (b2_lod2, batch_size=32, $EPOCHS epochs)..."
uv run python -m train.train_baselines \
    --model b2_lod2 \
    --data_dir "$EPISODES_DIR" \
    --region_dir "$REGION_DIR" \
    --save_dir checkpoints/baselines/b2_lod2 \
    --epochs "$EPOCHS" \
    --batch_size 32

echo "[1/4] Training B3 (b3_mae, batch_size=32, $EPOCHS epochs)..."
uv run python -m train.train_baselines \
    --model b3_mae \
    --data_dir "$EPISODES_DIR" \
    --region_dir "$REGION_DIR" \
    --save_dir checkpoints/baselines/b3_mae \
    --epochs "$EPOCHS" \
    --batch_size 32

echo "[1/4] Training Collapse Control (no_vicreg, $NO_VICREG_EPOCHS epochs)..."
uv run python -m train.train_spatial_jepa \
    --data_dir "$EPISODES_DIR" \
    --region_dir "$REGION_DIR" \
    --save_dir checkpoints/baselines/no_vicreg \
    --epochs "$NO_VICREG_EPOCHS" \
    --batch_size 32 \
    --no_vicreg

# ------------------------------------------------------------------------------
# 2. Frozen Probe Suite for B1, B2, B3, B4, and no_vicreg -> JSON
# ------------------------------------------------------------------------------
echo "[2/4] Running Probe Suite on B1-B4 + no_vicreg..."
for b in b1_lod0 b2_lod2 b3_mae b4_uniform32; do
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

if [[ -f "checkpoints/baselines/no_vicreg/best.pkl" ]]; then
    uv run python -m eval.probe_suite \
        --model_type m2_static \
        --ckpt checkpoints/baselines/no_vicreg/best.pkl \
        --data_dir "$EPISODES_DIR" \
        --region_dir "$REGION_DIR" \
        --out_json "$RESULTS_DIR/probe_no_vicreg.json"
fi

# ------------------------------------------------------------------------------
# 3. Downstream Navigation Baselines (B0, B1, B2, B3) -> JSON + JSONL
# ------------------------------------------------------------------------------
echo "[3/4] Training & Evaluating B0 (b0_blind)..."
uv run python -m eval.train_ppo_nav \
    --model_type b0_blind \
    --region_dir "$REGION_DIR" \
    --task rl1 \
    --bc_epochs 20 \
    --total_steps "$B0_PPO_STEPS" \
    --save_dir checkpoints/ppo_nav_b0_rl1 \
    --device gpu

for t in rl1 rl2 rl3_beacon rl3_flash rl4_beacon rl4_flash; do
    uv run python -m eval.eval_navigation \
        --model_type b0_blind \
        --policy_ckpt checkpoints/ppo_nav_b0_rl1/best_ppo.pkl \
        --region_dir "$REGION_DIR" \
        --task "$t" \
        --num_episodes "$EVAL_EPISODES" \
        --seed "$SEED" \
        --out_json "$RESULTS_DIR/nav_b0_${t}.json"
done

for b in b1_lod0 b2_lod2 b3_mae; do
    b_ckpt="checkpoints/baselines/${b}/best.pkl"
    if [[ -f "$b_ckpt" ]]; then
        echo "[3/4] Training & Evaluating PPO Navigation for $b..."
        uv run python -m eval.train_ppo_nav \
            --model_type "$b" \
            --ckpt "$b_ckpt" \
            --region_dir "$REGION_DIR" \
            --task rl1 \
            --bc_epochs 20 \
            --total_steps "$PPO_STEPS" \
            --save_dir "checkpoints/ppo_nav_${b}_rl1" \
            --device gpu

        b_pol="checkpoints/ppo_nav_${b}_rl1/best_ppo.pkl"
        [[ ! -f "$b_pol" ]] && b_pol="checkpoints/ppo_nav_${b}_rl1/latest_ppo.pkl"
        uv run python -m eval.eval_navigation \
            --model_type "$b" \
            --backbone_ckpt "$b_ckpt" \
            --policy_ckpt "$b_pol" \
            --region_dir "$REGION_DIR" \
            --task rl1 \
            --num_episodes "$EVAL_EPISODES" \
            --seed "$SEED" \
            --out_json "$RESULTS_DIR/nav_${b}_rl1.json"
    fi
done

# Aggregate all JSON results into a single JSONL manifest
uv run python - <<'PY'
import glob, json, os
res_dir = os.environ.get("RESULTS_DIR", "results")
out_jsonl = os.path.join(res_dir, "all_baselines.jsonl")
files = sorted(glob.glob(os.path.join(res_dir, "*.json")))
with open(out_jsonl, "w", encoding="utf-8") as fout:
    for fp in files:
        with open(fp, "r", encoding="utf-8") as fin:
            data = json.load(fin)
        data["_file"] = os.path.basename(fp)
        fout.write(json.dumps(data) + "\n")
print(f"Aggregated {len(files)} JSON result files into {out_jsonl}")
PY

# ------------------------------------------------------------------------------
# 4. Package Baseline Weights + JSON/JSONL Logs into Master Zip
# ------------------------------------------------------------------------------
echo "[4/4] Packaging baseline weights and JSON/JSONL logs into ${OUT_ZIP}.zip..."
uv run python - <<'PY'
import os, shutil
export_dir = "/tmp/squared_sense_export_staging"
if os.path.exists(export_dir):
    shutil.rmtree(export_dir)
os.makedirs(export_dir, exist_ok=True)
for d in ["checkpoints/baselines", "checkpoints/ppo_nav_b0_rl1", "checkpoints/ppo_nav_b1_lod0_rl1", "checkpoints/ppo_nav_b2_lod2_rl1", "checkpoints/ppo_nav_b3_mae_rl1", "results"]:
    if os.path.exists(d):
        shutil.copytree(d, os.path.join(export_dir, d), dirs_exist_ok=True)
out_base = os.environ.get("OUT_ZIP", "/kaggle/working/squared_sense_master_artifacts")
os.makedirs(os.path.dirname(out_base) or ".", exist_ok=True)
zip_path = shutil.make_archive(out_base, "zip", export_dir)
print(f"Created archive: {zip_path} ({os.path.getsize(zip_path) / 1e6:.2f} MB)")
PY

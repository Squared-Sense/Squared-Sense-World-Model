# Modal Cloud GPU Training & Evaluation Pipeline Guide

This guide details how to train, evaluate, and generate publication figures for the **Squared Sense World Model** on **Modal** (`modal.com`) using cloud GPUs (NVIDIA L4 24GB, A10G, or A100).

---

## 1. Architecture Overview

```
+----------------------------------------------------------------------------+
|                                MODAL CLOUD                                 |
|                                                                            |
|  +----------------------------------------------------------------------+  |
|  | Persistent Volume: 'squared-sense-vol' mounted at /vol               |  |
|  |   /vol/data/episodes/*.npz       (Voxel episodes)                     |  |
|  |   /vol/data/courses/*.npz        (Navigation courses)                 |  |
|  |   /vol/checkpoints/              (Spatial, Temporal & Baseline ckpts) |  |
|  |   /vol/results/*.json            (Probe and navigation metrics)       |  |
|  |   /vol/paper_figs/*.pdf, *.png   (Springer LNEE vector figures)       |  |
|  +----------------------------------------------------------------------+  |
|                                     ^                                      |
|                                     | Auto-commit via vol.commit()         |
|  +----------------------------------------------------------------------+  |
|  | Container Image: Debian Slim (Python 3.12)                           |  |
|  |   - CUDA 12 + JAX 0.12.0.1 + Flax NNX 0.12.12 + Optax 0.2.9          |  |
|  |   - Repository attached at /root/squared_sense                       |  |
|  |   - XLA_PYTHON_CLIENT_PREALLOCATE=false                              |  |
|  |   - Fast container builds (strictly ignores data/, .venv/, etc.)      |  |
|  +----------------------------------------------------------------------+  |
|                                     |                                      |
|  +----------------------------------------------------------------------+  |
|  | Remote GPU Functions (NVIDIA L4 24GB VRAM, 16GB Host RAM)             |  |
|  |   - verify_gpu       : Hardware & volume verification                |  |
|  |   - train_spatial    : Stage 1A Spatial JEPA (M2)                    |  |
|  |   - train_baselines  : Baselines B1 (LOD-0), B2 (LOD-2), B3 (3D-MAE) |  |
|  |   - train_temporal   : Stage 1B Dual-LSTM Temporal JEPA (M1)         |  |
|  |   - run_eval         : Frozen Probe Suite across all models          |  |
|  |   - train_ppo        : Frozen-backbone ActorCriticHead PPO Policy    |  |
|  |   - eval_nav         : Zero-shot course evaluation & gating profile  |  |
|  |   - generate_figures : Springer Nature LNEE vector figures           |  |
|  |   - run_all          : Full end-to-end automated pipeline            |  |
|  +----------------------------------------------------------------------+  |
+----------------------------------------------------------------------------+
```

---

## 2. One-Time Local Setup & Authentication

Run these commands in your local terminal:

```bash
# 1. Install or sync dependencies with uv
uv sync

# 2. Authenticate with your Modal account (opens browser)
uv run modal setup
```

Once authenticated, your credentials are saved to `~/.modal.toml`.

---

## 3. Upload Datasets to the Persistent Volume

Before launching training runs, upload your local `data/` folder to the persistent volume `squared-sense-vol`:

```bash
# Upload all episodes and courses to the volume under /data
uv run modal volume put squared-sense-vol data /data

# (Optional) Verify remote volume files
uv run modal volume ls squared-sense-vol /data
uv run modal volume ls squared-sense-vol /data/episodes
uv run modal volume ls squared-sense-vol /data/courses
```

> [!NOTE]
> The container automatically mounts this volume at `/vol`, so files will be accessible at `/vol/data/episodes` and `/vol/data/courses`.

---

## 4. Execution Commands

### Step 0: Verify Cloud GPU & Volume Connectivity
Confirms `nvidia-smi`, `jax.devices()`, CUDA backend, and directory structure on `/vol`:

```bash
uv run modal run modal_app.py --action verify
```

---

### Step 1: Run Fast Cloud Smoke Test (< 2 Minutes)
Verifies the entire pipeline end-to-end on synthetic data inside the GPU container:

```bash
uv run modal run modal_app.py --action smoke-test
```

---

### Step 2: Stage 1A Spatial JEPA Training (`M2`)
Trains `MultiLODBackbone` context encoder and `JEPAPredictor` via 3D spatial masking and stop-gradient EMA targets with VICReg regularization:

```bash
uv run modal run modal_app.py --action train-spatial --epochs 80 --batch-size 32
```

Outputs committed to `/vol/checkpoints/m2_spatial_jepa/`:
- `best.pkl`
- `last.pkl`
- `metrics.jsonl`
- `train.log`

---

### Step 3: Train Parameter-Matched Baselines (`B1`, `B2`, `B3`)
Trains all three comparison baselines sequentially:

```bash
# Train all 3 baselines:
uv run modal run modal_app.py --action train-baselines --model all

# Or train an individual baseline:
uv run modal run modal_app.py --action train-baselines --model b1_lod0
uv run modal run modal_app.py --action train-baselines --model b2_lod2
uv run modal run modal_app.py --action train-baselines --model b3_mae
```

---

### Step 4: Stage 1B Dual-LSTM Temporal JEPA (`M1`)
Trains the `DualLSTMTemporalIntegrator` across Phase 1 (frozen spatial backbone) and Phase 2 (anchored top-2 block fine-tuning):

```bash
uv run modal run modal_app.py --action train-temporal
```

Outputs committed to `/vol/checkpoints/m1_full_temporal/`:
- `temporal_best.pkl`
- `spatial_finetuned_best.pkl`
- `metrics.jsonl`
- `train.log`

---

### Step 5: Representation Probes Suite
Runs frozen linear and non-linear probes (distant goal regression, local hazard classification, kinematic velocity, occlusion memory, and multi-step rollout):

```bash
uv run modal run modal_app.py --action eval-probes
```

Outputs committed to `/vol/results/`:
- `probe_m2_static.json`
- `probe_m1_full.json`
- `probe_b1_lod0.json`
- `probe_b2_lod2.json`
- `probe_b3_mae.json`

---

### Step 6: PPO Navigation Policy Training & Course Evaluation
Trains the ~50K parameter `ActorCriticHead` on frozen representations, then evaluates zero-shot performance and gating efficiency:

```bash
# Train PPO policy on debug and baked courses:
uv run modal run modal_app.py --action train-ppo

# Evaluate navigation success rate, SPL, and empirical MFLOPs/step:
uv run modal run modal_app.py --action eval-nav
```

Outputs committed to `/vol/results/nav_m2.json`.

---

### Step 7: Springer Nature LNEE Vector Figures Generation
Generates all 4 publication vector figures (`.pdf` + 600 DPI `.png`) from cloud checkpoints and evaluation results:

```bash
uv run modal run modal_app.py --action make-figures
```

Outputs committed to `/vol/paper_figs/`:
- `fig1_training_and_rankme.pdf` / `.png`
- `fig2_3d_patch_similarity.pdf` / `.png`
- `fig3_gating_efficiency.pdf` / `.png`
- `fig4_navigation_trajectories.pdf` / `.png`

---

### Step 8: Full End-to-End Orchestrator (`run-all`)
Runs the complete sequence automatically in a single cloud invocation:

```bash
uv run modal run modal_app.py --action run-all
```

---

## 5. Download Results & Checkpoints to Local Machine

Download trained checkpoints, evaluation JSONs, and publication figures from the persistent Modal volume:

```bash
# Download all checkpoints to ./checkpoints_cloud/
uv run modal volume get squared-sense-vol /checkpoints ./checkpoints_cloud

# Download evaluation JSON results to ./results_cloud/
uv run modal volume get squared-sense-vol /results ./results_cloud

# Download publication vector figures to ./paper_figs_cloud/
uv run modal volume get squared-sense-vol /paper_figs ./paper_figs_cloud
```

---

## 6. GPU Hardware Selection & Cost Guide

The default GPU is configured as `DEFAULT_GPU = "L4"` at the top of `modal_app.py`. You can adjust it based on your needs:

| GPU | VRAM | Speed vs CPU | Typical Modal Price | Recommendation |
| :--- | :--- | :--- | :--- | :--- |
| **NVIDIA L4** | 24 GB | ~15-20× faster | ~$0.80 / hour | **Default (Recommended)**: Optimal cost-to-performance for 3D ConvNeXt + Transformer |
| **NVIDIA A10G** | 24 GB | ~18-22× faster | ~$1.10 / hour | Fast alternative with 24 GB VRAM |
| **NVIDIA A100 (40GB)** | 40 GB | ~30-40× faster | ~$3.67 / hour | Best for ultra-fast full 80-epoch sweeps with large batch sizes (e.g. batch_size=64) |

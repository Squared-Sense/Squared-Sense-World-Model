# Squared Sense World Model

A multi-LOD 3D Joint-Embedding Predictive Architecture (JEPA) and ego-motion-compensated Dual-LSTM world model for 3D voxel navigation in Minecraft, implemented in **JAX**, **Flax NNX**, and **Optax**.

## Architecture & Model Variants

All models operate strictly **channel-last** `(..., X, Y, Z, C)` on 3 concentric $16 \times 16 \times 16 \times 16$ voxel shells (LOD-0 at $1\times$, LOD-1 at $2\times$, LOD-2 at $4\times$) with a 16-channel voxel schema (`data/schema.py`).

| Model | Description | Trainable Params | Inference Params | Dense MFLOPs/step |
| :--- | :--- | :---: | :---: | :---: |
| **`M1` (`m1_full`)** | Multi-LOD Spatial JEPA + Dual-LSTM Temporal Integrator (`448`-D feat) | 5,457,132 | 4,455,212 | 1,791.3 |
| **`M2` (`m2_static`)** | Multi-LOD Spatial JEPA (`MultiLODBackbone` + `JEPAPredictor`, `256`-D feat) | 4,649,568 | 3,647,648 | 1,770.1 |
| **`B1` (`b1_lod0`)** | Single-LOD JEPA on LOD-0 ($1\times$, 64 tokens, `256`-D feat) | 4,648,672 | 3,647,136 | 555.2 |
| **`B2` (`b2_lod2`)** | Single-LOD JEPA on LOD-2 ($4\times$, 64 tokens, `256`-D feat) | 4,648,672 | 3,647,136 | 555.2 |
| **`B3` (`b3_mae`)** | Generative 3D-MAE (Multi-LOD encoder + 3D patch decoder, `256`-D feat) | 4,797,792 | 3,647,648 | 1,770.1 |
| **`B0` (`b0_blind`)** | Blind Compass-Only Baseline (zero voxel perception, `goal_vec` only) | 0 (18,701 head) | 0 (18,701 head) | 0.0 (0.037 MLP) |

- **Event-Driven Shell Gating (`models/gating.py`)**: Host-side stateful cache (`EventDrivenShellGate`, `tau_gate=0.05`) that skips unchanged shell CNNs and Transformer passes (`1743.6` nominal dense MFLOPs/step).
- **Frozen-Backbone Policy (`eval/train_ppo_nav.py`)**: Lightweight `ActorCriticHead` (`18,701` params for `b0_blind`, `51,469` params for `256`-D, `76,045` params for `448`-D `M1`) trained with optional Behavioral Cloning (BC) warm-start and clipped PPO.

## Quickstart

### 1. Run Unit & Integration Tests
```bash
uv run pytest -v
```

### 2. Pre-Training (Stage 1A, Baselines, Stage 1B)
```bash
# Stage 1A: Multi-LOD Spatial JEPA (M2)
uv run python -m train.train_spatial_jepa --region_dir data/baked_regions --save_dir checkpoints/m2_spatial_jepa

# Baselines: B1 (LOD-0), B2 (LOD-2), B3 (3D-MAE)
uv run python -m train.train_baselines --model b1_lod0 --region_dir data/baked_regions
uv run python -m train.train_baselines --model b2_lod2 --region_dir data/baked_regions
uv run python -m train.train_baselines --model b3_mae --region_dir data/baked_regions

# Stage 1B: Dual-LSTM Temporal JEPA (M1)
uv run python -m train.train_temporal_jepa --spatial_ckpt checkpoints/m2_spatial_jepa/best.pkl --region_dir data/baked_regions --save_dir checkpoints/m1_full_temporal
```

### 3. Frozen Representation Probe Suite
```bash
uv run python -m eval.probe_suite --model_type m1_full --ckpt checkpoints/spatial_finetuned_best.pkl --temporal_ckpt checkpoints/temporal_best.pkl --region_dir data/baked_regions --out_json results/probe_m1.json
```

### 4. RL1–RL4 Navigation Training & Evaluation (`MinecraftNavEnv` / `VoxelNavEnv`)
```bash
# B0: Blind compass-only baseline (RL1)
uv run python -m eval.train_ppo_nav --model_type b0_blind --region_dir data/baked_regions --task rl1 --bc_epochs 20 --total_steps 50000 --save_dir checkpoints/ppo_nav_b0

# RL1: Single-target curriculum navigation
uv run python -m eval.train_ppo_nav --model_type m1_full --region_dir data/baked_regions --task rl1 --bc_epochs 20 --total_steps 50000 --save_dir checkpoints/ppo_nav_m1

# RL2: 5-target chained navigation with 50% 6-tick burst compass occlusion
uv run python -m eval.train_ppo_nav --model_type m1_full --region_dir data/baked_regions --task rl2 --init_policy checkpoints/ppo_nav_m1/best_ppo.pkl --lr 1e-4 --total_steps 25000 --save_dir checkpoints/ppo_nav_m1_rl2

# Zero-shot evaluation (RL1, RL2, RL3/RL4 beacon & flash dead-reckoning)
uv run python -m eval.eval_navigation --model_type m1_full --policy_ckpt checkpoints/ppo_nav_m1_rl2/best_ppo.pkl --region_dir data/baked_regions --task rl4_flash --num_episodes 20 --use_gating --out_json eval/results_m1_rl4_flash.json
```

### 5. Publication Vector Figures
```bash
uv run python -m paper_figs.make_figures --out_dir paper_figs/final --metrics_dir checkpoints
```

## Documentation
- [docs/CONTRACT.md](docs/CONTRACT.md) — 16-channel voxel schema, LOD geometry, and data formats
- [docs/MODEL_CONTRACT.md](docs/MODEL_CONTRACT.md) — Tensor shapes and exact module parameter counts
- [docs/TRAIN_CONTRACT.md](docs/TRAIN_CONTRACT.md) — Stage 1A, Stage 1B, and baseline training specs
- [docs/EVAL_CONTRACT.md](docs/EVAL_CONTRACT.md) — Probe suite, RL1/RL2 navigation, and figure specs
- [docs/RL_BC_NAVIGATION_GUIDE.md](docs/RL_BC_NAVIGATION_GUIDE.md) — Detailed RL1/RL2 & BC warm-start guide
- [docs/MODAL_GUIDE.md](docs/MODAL_GUIDE.md) — Cloud GPU execution on Modal

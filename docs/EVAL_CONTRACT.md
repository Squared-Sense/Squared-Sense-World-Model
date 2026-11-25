# Squared Sense — Evaluation, RL Navigation & Figures Contract

## 1. Frozen Feature Extraction Interface
- All models (`M1`, `M2`, `B1`, `B2`, `B3`) expose frozen inference embeddings:
  - Static models (`M2`, `B1`, `B2`, `B3`): mean-pool `tokens (B, N_tok, 256)` -> `z (B, 256)`.
  - Full temporal model (`M1`): concatenate mean-pooled `z_t (B, 256)` and Dual-LSTM fused
    context `c_t (B, 192)` -> `feat (B, 448)` (or `c_t (B, 192)` alone for pure-memory probes).

## 2. Option B Frozen Probe Suite (`eval/probe_suite.py`, CPU-friendly scikit-learn)
Run across 3 random seeds (report mean ± std) with `StandardScaler` fit strictly on `X_train` and per-episode Dual-LSTM hidden state resets:
1. Distant Goal Direction & Distance (30-48 blocks): `RidgeCV` regression on `meta[:, 3:6]` (R² & Cosine Sim).
2. 1-Block Local Hazard Detection: `LogisticRegression(class_weight="balanced")` binary classification (F1 & AUROC) for lava/drop within 4 blocks ahead.
3. Kinematic Velocity & Occlusion Memory (TI-JEPA Kill Test):
   - Velocity R² (`RidgeCV`) regressing `state[:, 0:3]` (`vx, vy, vz`).
   - Occlusion recall F1 (`LogisticRegression(class_weight="balanced")`) predicting whether a hazard was visible 5 steps earlier after a turn.
4. Representation Health & Multi-Step Rollout: Uncentered RankMe + k-step (k=1,2,4,8) latent cosine similarity & L1 error.

## 3. Option A 3D Voxel Navigation Simulator (`eval/sim_env.py`)
- Arena volume: `(16, 128, 64, 128)` uint8 (C, X, Y, Z) + ordered waypoints `[(x, y, z), ...]`.
- Discrete Action Space (`num_actions = 12`):
  0: Forward, 1: Forward-Left, 2: Forward-Right, 3: Strafe-Left, 4: Strafe-Right,
  5: Backward, 6: Turn-Left (-45°), 7: Turn-Right (+45°), 8: Step-Up-Forward,
  9: Pitch-Up (+15°), 10: Pitch-Down (-15°), 11: Stay / Idle.
- Physics: 1-block step-up allowed; drop <=2 blocks safe; drop >=3 blocks = fall penalty;
  stepping on `ch 2 >= 128` (lava/hazard) = terminate with penalty.
- Potential-Based RL Reward Function:
  - Reach active waypoint (dist <= 1.5 blocks) : +10.0 (advance to next waypoint; episode succeeds on final waypoint)
  - Potential-based distance shaping           : +0.25 * (d_{t-1} - d_t)
  - Step into hazard (`ch 2 >= 128`)           : -5.0 (terminate, collision=True)
  - Fall >= 3 blocks (`dy <= -3`)              : -3.0 (terminate, collision=True)
  - Wall bump / invalid move                   : -0.1
  - Per-step time penalty                      : -0.02

## 4. Frozen-Backbone PPO Policy (`eval/train_ppo_nav.py`, RTX 3050 Mobile friendly)
- Backbone (`SpatialJEPA` / `SingleLODJEPA` / `Generative3DMAE` + optional `DualLSTM`) is 100% FROZEN (`stop_gradient`), or omitted for **`b0_blind`** (`feat_dim=0`, `18,701` head params, `0.0` perception MFLOPs/step).
- Actor-Critic MLP: input `[frozen_feat, goal_dir_3d]` -> `Linear(in, 128) -> GELU -> Linear(128, 128) -> GELU`
  -> Actor head `Linear(128, 12)` + Critic head `Linear(128, 1)`.
- Supports optional Behavioral Cloning (BC) warm-start from A* demos before PPO fine-tuning, as well as `--init_policy` fine-tuning across the **RL1–RL4** task ladder (`rl1`, `rl2`, `rl3_beacon`, `rl3_flash`, `rl4_beacon`, `rl4_flash`, plus `--flash_compass_ticks`).
- Zero-Shot Evaluation Metrics (`eval/eval_navigation.py`):
  Success Rate (`success_rate_pct` / Full5%), `spl` / `spl_pct`, `chns` (dataset-level Complexity- & Horizon-Weighted Navigation Score in `0..100`), `compass_visible_pct` (% of acted-on steps with visible compass), `waypoint_completion_pct`, `mean_targets_reached`, `hazard_collision_rate_pct`, `empirical_mflops_per_step`, and `latency_ms_per_step`.

## 5. Publication Figures (`paper_figs/make_figures.py`, Springer LNEE Compliance)
- Output format: Vector `.pdf` (and 600 DPI `.png` preview).
- Grayscale-safe colorblind palette + distinct line styles/markers.
- ZERO raw Minecraft screenshots (all 3D voxel plots rendered directly from NumPy arrays).
# Reinforcement Learning & Behavioral Cloning Navigation Guide

Comprehensive reference for training and evaluating navigation policies on frozen perception representations in **Squared Sense World Model**.

---

## 1. System Architecture Overview

The Squared Sense navigation agent operates on a **strictly frozen perception representation** paired with a lightweight, trainable **ActorCriticHead** MLP.

```
                          ┌────────────────────────┐
                          │   Agent Observation    │
                          │ (Concentric 3D Shells) │
                          └───────────┬────────────┘
                                      │
                                      ▼
                   ╔══════════════════════════════════════╗
                   ║      STRICTLY FROZEN BACKBONE        ║
                   ║                                      ║
                   ║  [M2: Multi-LOD Spatial JEPA (3D)]   ║ ──> z_static (256-dim)
                   ║  [M1: Dual-LSTM Temporal Integrator] ║ ──> c_t      (192-dim)
                   ║                                      ║
                   ║       (0 Gradients / Inference Only) ║
                   ╚══════════════════════════════════════╝
                                      │
                                      ▼
                        Fused Feature [z_static, c_t] (448-dim)
                                      +
                        Normalized Goal Direction (3-dim)
                                      │
                                      ▼
                   ╔══════════════════════════════════════╗
                   ║       TRAINABLE ACTOR-CRITIC         ║
                   ║         (~51,469 Parameters)         ║
                   ║                                      ║
                   ║  Linear(in_dim, 128) -> GELU         ║
                   ║  Linear(128, 128)    -> GELU         ║
                   ║  ├─ Actor:  Linear(128, 12) -> logits║
                   ║  └─ Critic: Linear(128, 1)  -> value ║
                   ╚══════════════════════════════════════╝
```

### Key Architectural Contracts
1. **Perception Weights are Bit-Identical**: Backpropagation stops strictly before the frozen perception models via `jax.lax.stop_gradient` and separate optimization scopes. At no point are gradients computed or backpropagated into the Spatial JEPA or Dual-LSTM weights.
2. **Compact Policy Space**: The `ActorCriticHead` contains only ~51,469 parameters. This enables rapid optimization and minimal memory consumption during both CPU and GPU execution.
3. **Observation Space**: Concentric multi-LOD voxel shells (LOD-0: $16^3$, LOD-1: $16^3$, LOD-2: $16^3$) centered on the agent with cardinal yaw snapping (hysteresis deadband $\pm 10^\circ$).

---

## 2. Behavioral Cloning (BC) Warm-Start

### Purpose
When initializing reinforcement learning from scratch in high-dimensional 3D voxel space, uniform random exploration frequently causes the agent to fall off edges or walk into hazards (e.g. lava trenches) before discovering distant waypoints ("the random walk into lava problem"). 

Behavioral Cloning warm-starts the policy weights by supervising action choices on expert or heuristic demonstrations.

### Mathematical Formulation
Given demonstration tuples $\mathcal{D}_{demo} = \{ (s_i, a_i^*) \}_{i=1}^N$ containing frozen perception features $s_i = [\mathbf{z}_i, \mathbf{g}_i]$ and expert discrete actions $a_i^* \in \{0, \dots, 11\}$:

$$\mathcal{L}_{BC}(\theta) = -\frac{1}{N} \sum_{i=1}^N \log \pi_\theta(a_i^* \mid s_i) = \frac{1}{N} \sum_{i=1}^N \text{CrossEntropy}\left( \text{softmax}(f_\theta(s_i)), \, a_i^* \right)$$

### Optimization Protocol
- **Optimizer**: AdamW (`lr = 3e-4`, `weight_decay = 1e-4`, `clip_by_global_norm = 0.5`).
- **Duration**: **3 to 5 epochs**. 
- **Goal**: Bias initial action distribution logits toward directional movement toward the active waypoint without collapsing policy entropy.

---

## 3. Proximal Policy Optimization (PPO) Training Loop

After optional BC warm-start, the policy enters online interaction with `VoxelNavEnv`.

### PPO Algorithmic Workflow

```
   ┌────────────────────────────────────────────────────────┐
   │               1. Rollout Collection Phase              │
   │  For t = 1 ... rollout_steps (e.g., 128 steps):       │
   │   - Extract frozen features z_t, c_t (JIT compiled)    │
   │   - Compute logits & value V(s_t) via ActorCriticHead  │
   │   - Sample action a_t ~ Categorical(pi(.|s_t))         │
   │   - Step VoxelNavEnv -> next_obs, r_t, done_t          │
   │   - Store (s_t, a_t, log_pi_old, r_t, V_t, done_t)     │
   └───────────────────────────┬────────────────────────────┘
                               │
                               ▼
   ┌────────────────────────────────────────────────────────┐
   │           2. Advantage Estimation Phase (GAE)          │
   │   - Compute temporal difference errors:                │
   │       delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)      │
   │   - Compute smoothed GAE advantages:                   │
   │       A_t^GAE = sum_{l=0}^inf (gamma * lambda)^l delta │
   │   - Standardize advantages: (A - mean(A)) / std(A)     │
   │   - Compute discounted return targets: R_t = A_t + V_t │
   └───────────────────────────┬────────────────────────────┘
                               │
                               ▼
   ┌────────────────────────────────────────────────────────┐
   │             3. Mini-Batch Policy Update Phase          │
   │   For epoch = 1 ... ppo_epochs (e.g., 4 epochs):       │
   │     Shuffle buffer into mini-batches (e.g., size 32):  │
   │       - Ratio: r_t(theta) = pi_theta(a_t) / pi_old(a_t)│
   │       - Clipped Surrogate Objective:                   │
   │           L_CLIP = min(r_t * A_t, clip(r_t, 1+-eps)*A_t│
   │       - Value Regression Loss:                         │
   │           L_VF = 0.5 * (V_theta(s_t) - R_t)^2          │
   │       - Entropy Bonus:                                 │
   │           S[pi_theta] = -sum pi * log(pi)              │
   │       - Total Loss: L_CLIP + c1 * L_VF - c2 * Entropy  │
   │       - Gradient step via Optax AdamW                  │
   └───────────────────────────┬────────────────────────────┘
                               │
                               ▼
   ┌────────────────────────────────────────────────────────┐
   │               4. Checkpointing & Logging               │
   │   - Append metrics to ppo_metrics.jsonl & train.log    │
   │   - Update terminal progress bar with dynamic ETA      │
   │   - Save policy_best.pkl if mean reward improves       │
   └────────────────────────────────────────────────────────┘
```

### Reward Function Formulation (`MinecraftNavEnv` & `VoxelNavEnv`)
At each step $t$, the scalar environment reward $r_t$ is computed via potential-based shaping:

$$r_t = R_{\text{dist}} + R_{\text{heading}} + R_{\text{waypoint}} + R_{\text{step}} + R_{\text{wall}} + R_{\text{hazard}}$$

Where:
- **$R_{\text{dist}} = +0.25 \cdot (d_{t-1} - d_t)$**: Potential-based Euclidean distance shaping toward the active target $w^*$.
- **$R_{\text{heading}} = +0.05 \cdot (\cos\theta_t - \cos\theta_{t-1})$**: Horizontal heading-alignment shaping bonus (`MinecraftNavEnv`).
- **$R_{\text{waypoint}} = +10.0$**: Bonus awarded when the agent arrives within reaching radius ($d \le 1.5$ blocks) of the active target.
- **$R_{\text{step}} = -0.02$**: Constant time-penalty encouraging minimal-path navigation.
- **$R_{\text{wall}} = -0.15$** (`MinecraftNavEnv`) / **$-0.10$** (`VoxelNavEnv`): Penalty when blocked by solid obstacle blocks.
- **$R_{\text{hazard}} = -5.0$** (lava/hazard/void) or **$-3.0$** (fatal fall): Catastrophic termination penalty.

---

## 4. Overfitting Dynamics & Risks

### Is Training Prone to Overfitting?

| Component | Overfitting Risk | Primary Driver | Mitigations Built Into Codebase |
| :--- | :--- | :--- | :--- |
| **Perception Backbone** (`M1`, `M2`) | **0% (None)** | Strictly frozen | `jax.lax.stop_gradient`, 0 gradient operations. Bit-identical before and after training. |
| **Behavioral Cloning** (BC) | **Moderate if overtrained** | Memorizing trajectory coordinates | Cap BC at **3–5 epochs**. Use weight decay (`1e-4`), gradient clipping (`0.5`). Treat BC purely as a directional prior. |
| **PPO Policy Parameters** | **Low** | Small MLP (~51K params) | Clipped surrogate ratio ($\epsilon = 0.2$), entropy regularization ($\beta = 0.01$), GAE variance reduction ($\lambda = 0.95$). |
| **Environment Topology** | **Moderate on single course** | Memorizing a single bridge seed | Train across procedural seeds or multiple course files (`--courses_dir`). Test zero-shot on held-out courses. |

### Practical Recommendations
1. **Never run BC for $>10$ epochs**: More than 10 epochs causes the policy distribution to become overconfident (entropy collapses below 0.5), leading to severe covariate shift when the agent encounters unseen states during online RL.
2. **Keep PPO Entropy $\ge 1.0$**: The terminal progress bar monitors `Entropy` (initial $\approx 2.48$). If entropy drops below $0.5$ early in training, the policy has collapsed to a single deterministic action.

---

## 5. Execution Reference & Commands

### Prerequisites & Checkpoint Auto-Resolution
The scripts automatically detect checkpoints from either `checkpoints/` or `checkpoints/m1_full_temporal/`:
- Backbone weights: `spatial_finetuned_best.pkl`
- Temporal weights: `temporal_best.pkl`
- Policy weights: `checkpoints/ppo_nav/policy_best.pkl`

### Command 1: Run BC Warm-Start + PPO Training on CPU
```bash
uv run python -m eval.train_ppo_nav \
  --model_type m1_full \
  --bc_epochs 5 \
  --total_steps 3000 \
  --device cpu \
  --save_dir checkpoints/ppo_nav
```

### Command 2: Run Fast Smoke Test (CPU)
```bash
uv run python -m eval.train_ppo_nav \
  --model_type m1_full \
  --smoke_test \
  --device cpu
```

### Command 3: Run Zero-Shot Course Evaluation
```bash
uv run python -m eval.eval_navigation \
  --model_type m1_full \
  --policy_ckpt checkpoints/ppo_nav/policy_best.pkl \
  --num_episodes 10 \
  --out_json results/nav_eval_m1_full.json
```

---

## 6. Training Time & Throughput Benchmarks

| Device | BC Warm-Start (5 Epochs) | PPO (3,000 Steps / 23 Updates) | PPO (5,000 Steps / 39 Updates) | Zero-Shot Eval (10 Episodes) |
| :--- | :--- | :--- | :--- | :--- |
| **Local CPU** (Modern x86_64) | **~2–5 seconds** | **~75–90 seconds** (~1.5 min) | **~2.5–3.0 minutes** | **~15–20 seconds** |
| **NVIDIA GPU** (RTX / T4 / V100) | **~1–2 seconds** | **~15–20 seconds** | **~30–45 seconds** | **~4–6 seconds** |

### Output Files & Telemetry Logs
- **`checkpoints/ppo_nav/train.log`**: Human-readable timestamped log lines formatted as `[YYYY-MM-DD HH:MM:SS] [PID ...] [PPO Train | Step XX/YY]`.
- **`checkpoints/ppo_nav/ppo_metrics.jsonl`**: Machine-readable JSON records per update with `update`, `total_steps`, `loss`, `policy_loss`, `value_loss`, `entropy`, `mean_step_reward`, `rolling_ep_return`, `success_rate`, `mean_targets_reached`, `num_targets`, `full5_rate`, `turn_pct`, and `blocked_pct`.
- **`checkpoints/ppo_nav/policy_best.pkl`** / **`best_ppo.pkl`** / **`latest_ppo.pkl`**: Serialized `ActorCriticHead` weights (`best_ppo.pkl` and `policy_best.pkl` saved on peak rolling return; `latest_ppo.pkl` saved every 10 updates).

---

## 7. RL2–RL4 & `B0` Compass-Ablation Ladder

RL2–RL4 extend `MinecraftNavEnv` ([env/minecraft_env.py](file:///c:/Squared%20Sense/Squared%20Sense%20World%20Model/env/minecraft_env.py)) and `train_ppo_nav.py` ([eval/train_ppo_nav.py](file:///c:/Squared%20Sense/Squared%20Sense%20World%20Model/eval/train_ppo_nav.py)) to measure **long-horizon spatial memory, temporal persistence, and visual beacon / dead-reckoning reliance** (`M1` Dual-LSTM vs. `M2` static vs. `B0` blind compass-only).

### Task Ladder Specification (`--task`)

| Task | Targets (`--num_targets`) | Max Steps | `--compass_occlusion` | `--flash_compass_ticks` | Purpose |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`rl1`** (default) | `1` | `250` | `0.0` (100% visible) | `0` | Single-target curriculum navigation |
| **`rl2`** | `5` (chained) | `600` | `0.5` (6-tick bursts) | `0` | Multi-target memory under intermittent compass blackout |
| **`rl3_beacon`** | `1` | `250` | `1.0` (0% compass) | `0` | Single-target pure Channel-8 voxel beacon navigation |
| **`rl3_flash`** | `1` | `250` | `0.0` | `3` (ticks `0,1,2`) | Single-target brief compass flash + dead-reckoning |
| **`rl4_beacon`** | `5` (chained) | `600` | `1.0` (0% compass) | `0` | 5-target chain with 0% compass (Channel-8 beacons only) |
| **`rl4_flash`** | `5` (chained) | `600` | `0.0` | `3` per leg | 5-target chain with 3-tick compass flash at start of each leg |

- **`B0` (`b0_blind`) Blind Compass-Only Baseline**: Uses `ActorCriticHead(feat_dim=0)` (`18,701` params, `0.0` perception MFLOPs/step, `skip_shells=True`), bypassing all 3D voxel perception so `--ckpt` is not required.
- **`--flash_compass_ticks N`**: Overrides the task default (`3` for `rl3_flash`/`rl4_flash`, `0` otherwise). When `flash_compass_ticks > 0`, `MinecraftNavEnv` tracks `_leg_step_count`, keeps `compass_occluded = False` for ticks `0..N-1`, locks `compass_occluded = True` (`goal_vec = [0, 0, 0]`) for `_leg_step_count >= N`, and resets `_leg_step_count = 0` on each mid-chain target arrival.

### Key RL2 Mechanics
1. **5-Target Reachable Chain (`reset`)**: Samples a traversable spawn via `SpawnSampler`, then chains 5 distinct reachable targets using `BoundedBfsTargetGenerator` (`10–24` blocks from each previous anchor, retrying alternative cardinal orientations if needed to avoid dead-ends).
2. **Multi-LOD Waypoint Map Injection (`Channel.WAYPOINT`)**: Projects all remaining unreached targets (`0.5`) and the active target (`1.0`) into Channel 8 across LOD-0, LOD-1, and LOD-2 shells.
3. **50% Contiguous Burst Compass Occlusion**: Every `occlusion_burst_len` (6) ticks (with a randomized initial phase per episode), `compass_occluded` toggles. While occluded, `obs["goal_vec"]` is zeroed (`np.zeros(3, dtype=np.float32)`), preserving the `(3,)` observation contract so RL1 `ActorCriticHead` checkpoints load directly.
4. **Reward Safeguards**: Distance shaping (`+0.25 * (d_prev - d_curr)`) and heading alignment (`+0.05 * (cos_curr - cos_prev)`) always compute against the true active `self.target_pos`, even when `goal_vec` is zeroed. Switching `self.target_pos` at the end of a tick on target arrival (`+10.0`) ensures the next tick computes `d_prev` and `cos_prev` relative to the new target without spurious negative shaping spikes.
5. **Continuous Temporal State**: Reaching targets `1..4` keeps `done = False`, so `DualLSTMTemporalIntegrator` (`M1`) retains its spatial memory state across the entire 5-target chain and resets only when `done = True` (all 5/5 reached, fatal hazard/fall, or `max_steps`).

### RL2 Fine-Tuning Commands (from RL1 Checkpoints)
Passing `--init_policy` loads RL1 `ActorCriticHead` weights and skips BC warm-start by default (`bc_epochs = 0` unless explicitly set).

#### Fine-Tune `M1` (`m1_full`) on RL2
```bash
uv run python -m eval.train_ppo_nav \
  --model_type m1_full \
  --ckpt checkpoints/spatial_finetuned_best.pkl \
  --temporal_ckpt checkpoints/temporal_best.pkl \
  --init_policy checkpoints/ppo_nav_m1_overnight/latest_ppo.pkl \
  --region_dir data/baked_regions \
  --task rl2 \
  --num_targets 5 \
  --compass_occlusion 0.5 \
  --occlusion_burst 6 \
  --lr 1e-4 \
  --bc_epochs 0 \
  --total_steps 25000 \
  --save_dir checkpoints/ppo_nav_m1_rl2
```

#### Fine-Tune `M2` (`m2_static`) on RL2
```bash
uv run python -m eval.train_ppo_nav \
  --model_type m2_static \
  --ckpt checkpoints/m2_spatial_jepa/best.pkl \
  --init_policy checkpoints/ppo_nav_m2_overnight/latest_ppo.pkl \
  --region_dir data/baked_regions \
  --task rl2 \
  --num_targets 5 \
  --compass_occlusion 0.5 \
  --occlusion_burst 6 \
  --lr 1e-4 \
  --bc_epochs 0 \
  --total_steps 25000 \
  --save_dir checkpoints/ppo_nav_m2_rl2
```

### RL2 Zero-Shot Evaluation on Held-Out Seeds
`eval_navigation.py` logs `mean_targets_reached` (out of 5), `waypoint_completion_pct`, and `success_rate_pct` (Full 5/5%):

```bash
uv run python -m eval.eval_navigation \
  --model_type m1_full \
  --backbone_ckpt checkpoints/spatial_finetuned_best.pkl \
  --temporal_ckpt checkpoints/temporal_best.pkl \
  --policy_ckpt checkpoints/ppo_nav_m1_rl2/best_ppo.pkl \
  --region_dir data/baked_regions \
  --task rl2 \
  --num_episodes 20 \
  --seed 1000 \
  --out_json eval/results_m1_rl2.json
```

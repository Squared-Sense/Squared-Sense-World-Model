# Squared Sense — Phase 1 Data & Interface Contract (IMMUTABLE)

## 1. Repository Layout
- `mod_export/`   : Java mod patch (`/sqsense export <name>`)
- `data/`         : `schema.py`, `shell_extractor.py`, `astar_oracle.py`, `dataset.py`
- `models/`       : Phase 2-3 neural modules
- `train/`        : Phase 2-4 training & diagnostics
- `eval/`         : Phase 5 probes & simulator
- `paper_figs/`   : Vector PDF/SVG plots (NO Minecraft screenshots)
- `tests/`        : Pytest unit tests for every module

## 2. 16-Channel Voxel Schema (`uint8` [0..255] on disk -> `float32` [0.0..1.0] in model)
- ch 0  : Solid opaque walkable terrain             [AVG pool]
- ch 1  : Passable air / empty space                [AVG pool]
- ch 2  : Hazard (lava, fire, magma)                [MAX pool - salience]
- ch 3  : Liquid (water)                            [AVG pool]
- ch 4  : Climbable / vertical step (stairs/ladders)[MAX pool - salience]
- ch 5  : Foliage / fragile / breakable             [AVG pool]
- ch 6  : World Y Level (norm [-64..320] -> [0..255])[AVG pool]
- ch 7  : Collision height (0=air, 128=slab, 255=full/fence) [MAX pool - salience]
- ch 8  : Waypoint / goal beacon marker             [MAX pool - salience]
- ch 9  : Resource / interactive block              [MAX pool - salience]
- ch 10 : Walkable top surface (solid + >=2 air)    [MAX pool - salience]
- ch 11 : Combined light level (max(sky, block)/15) [AVG pool]
- ch 12 : Lethal drop / cliff indicator (>=3 air below) [MAX pool - salience]
- ch 13 : Surface shell flag (solid bordering air)  [MAX pool - salience]
- ch 14 : Normalized block hardness / diggability   [AVG pool]
- ch 15 : Out-of-bounds / void padding mask         [MAX pool]

MAX_POOL_CHANNELS = [2, 4, 7, 8, 9, 10, 12, 13, 15]
AVG_POOL_CHANNELS = [0, 1, 3, 5, 6, 11, 14]

## 3. Concentric Multi-LOD Shell Geometry (All output shapes: `(16, 16, 16, 16)` = C, X, Y, Z)
- LOD-0 (1x): span 16x16x16 | forward +13 / back -2  | lateral ±8  | base Y -6 / +9 (at default foot index gy=6)
- LOD-1 (2x): span 32x32x32 | forward +26 / back -4  | lateral ±16 | base Y -12 / +18
- LOD-2 (4x): span 64x64x64 | forward +52 / back -8  | lateral ±32 | base Y -24 / +36
- Player Foot Anchor: In local grid coordinates, player feet sit at (gx=8, gy=6, gz=2), standardized with the Fabric mod (ObservationConfig).
- Pitch vertical shift: `shift_y = round(4 * sin(pitch_rad)) * scale` where `scale in {1, 2, 4}` dynamically shifts vertical foot index gy in [1..12].
- Cardinal snapping: snap yaw to nearest 90° cardinal axis with ±10° rotational hysteresis deadband (snap departs when departing by > 45° + deadband = 55°).

## 4. Option B Episode Format (5 Hz trajectories)
- `shells`: `(T, 3, 16, 16, 16, 16)` uint8   # (Time, LOD, C, X, Y, Z)
- `state` : `(T, 12)` float32                # [vx, vy, vz, dx, dy, dz, sin_yaw, cos_yaw, pitch, health, food, grounded]
- `action`: `(T,)` int64                     # discrete movement primitive
- `meta`  : `(T, 6)` float32                 # [world_x, world_y, world_z, goal_dx, goal_dy, goal_dz]
- Splits  : 70% train / 15% val / 15% test strictly by episode ID (NEVER frame shuffle).

## 5. Option A Full-Course Volume Format
- Binary/NPZ volume: `(16, 128, 64, 128)` uint8 (C, X, Y, Z)
  centered around export origin: x in [cx-64, cx+63], y in [cy-16, cy+47], z in [cz-64, cz+63]
- Waypoints JSON: `courses/<course_name>_waypoints.json` containing ordered `[x, y, z]` local coordinates.

# Squared Sense — Model Architecture Contract (JAX + Flax NNX, IMMUTABLE)

## Framework & Conventions
- Framework : `jax`, `jax.numpy as jnp`, `flax.nnx as nnx`, `optax`
- Layout    : STRICTLY CHANNEL-LAST `(..., X, Y, Z, C)` everywhere.
- PRNG      : Pass `rngs: nnx.Rngs` into all `nnx.Module.__init__` constructors.

## Input / Output Tensor Shapes
- Voxel shells input : `x` shape `(B, 3, 16, 16, 16, 16)` float32 in `[0.0, 1.0]`
                       (Batch, LOD=3, X=16, Y=16, Z=16, Channels=16)
- Single shell input : `shell` shape `(N, 16, 16, 16, 16)` float32 (N, X, Y, Z, C)
- Proprioception     : `state` shape `(B, 12)` float32
- Discrete action    : `action` shape `(B,)` int32 (`num_actions = 12` default)
- Spatial tokens     : `tokens` shape `(B, 192, 256)` (3 LODs * 4*4*4 = 192 tokens, D=256)
- Coarse memory slots: `slots` shape `(B, 24, 256)` (3 LODs * 2*2*2 = 24 slots, D=256)

## Module Parameter Budgets (Exact Implemented Counts)
1. `models/cnn3d.py`         : Siamese 3D ConvNeXt + SE + CoordConv  -> **470,944** params (0.471M)
2. `models/transformer.py`   : Pos+Scale Emb (17,152) + 4L Enc (3,159,552) + 2L Pred (1,001,920)
   - Core Backbone Subtotal  : **3,647,648** params (3.648M)
   - Stage 1A Spatial Total  : **4,649,568** params (4.650M)
3. `models/gating.py`        : Event-Driven Shell Gate (0 learnable params; stateful host cache)
4. `models/temporal_lstm.py` : Dual-LSTM + Query Cross-Attention + IDM -> **807,564** params (0.808M)
   - Full Online Training    : **5,457,132** params (5.457M)
   - Full Inference Only     : **4,455,212** params (4.455M)
5. `models/baselines.py`     :
   - Baseline A (Single-LOD) : **4,648,672** trainable / **3,647,136** inference (1 shell, 64 tokens)
   - Baseline B (3D-MAE)     : **4,797,792** trainable (3,647,648 shared encoder + 1,150,144 voxel decoder)
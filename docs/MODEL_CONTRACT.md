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
1. `models/cnn3d.py`         : Siamese 3D ConvNeXt + SE + CoordConv  -> **470,944** params (0.471M; target ~470,560)
2. `models/transformer.py`   : Pos+Scale Emb (17,152) + 4L Enc (3,159,552) + 2L Pred (1,001,920)
   - Core Backbone (`context_encoder`) : **3,647,648** params (3.648M; target ~3,647,264)
   - Stage 1A Spatial Trainable (`context_encoder` + `predictor`) : **4,649,568** params (4.650M; target ~4,649,184)
3. `models/gating.py`        : Event-Driven Shell Gate (0 learnable params; stateful host cache; `DENSE_MFLOPS_PER_STEP = 1743.6`)
4. `models/temporal_lstm.py` : Dual-LSTM + Query Cross-Attention + IDM -> **807,564** params (0.808M; 773,952 excl. IDM head)
   - Full Online Training (`M1` ctx+pred+temporal) : **5,457,132** params (5.457M)
   - Full Inference Only (`M1` ctx+temporal)       : **4,455,212** params (4.455M)
5. `models/baselines.py`     :
   - Baseline A (`B1`/`B2` `SingleLODJEPA`, 1 shell, 64 tokens): **4,648,672** trainable / **3,647,136** inference
   - Baseline B (`B3` `Generative3DMAE`, 3 shells, 192 tokens): **4,797,792** trainable (3,647,648 enc + 1,150,144 dec) / **3,647,648** inference
   - Baseline 0 (`B0` `b0_blind`, compass-only): **0** perception params / **0.0** perception MFLOPs/step (**0.037** MFLOPs/step for 3-input `ActorCriticHead`)
6. `eval/train_ppo_nav.py`   : `ActorCriticHead` -> **18,701** params (`feat_dim=0`, `b0_blind`), **51,469** params (`feat_dim=256`), **76,045** params (`feat_dim=448`)


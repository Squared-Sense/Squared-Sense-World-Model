"""Stage 1B Dual-LSTM Temporal JEPA Training Pipeline.

Implements two-phase anchored optimization:
  - Phase 1 (Frozen Spatial Backbone): Trains DualLSTMTemporalIntegrator on stop-gradient
    spatial representations with K-step rollout discounted L1 loss and auxiliary IDM.
  - Phase 2 (Unfreeze Top 2 Transformer Blocks + Spatial Anchor): Finetunes top 2 Transformer
    blocks and final norm of SpatialJEPA context encoder while training Dual-LSTM with
    a spatial anchor loss on random frames.

Strictly follows docs/TRAIN_CONTRACT.md and docs/MODEL_CONTRACT.md.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from train.gpu_env import auto_configure_batching, setup_gpu_environment

# Ensure non-greedy allocation and VRAM limits are set prior to XLA initialization
setup_gpu_environment()

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from data.dataset import extract_temporal_clips, load_episode_splits
from models.temporal_lstm import DualLSTMTemporalIntegrator
from models.transformer import SpatialJEPA, sample_3d_jepa_masks
from train.checkpoint import load_checkpoint, save_checkpoint
from train.diagnostics import ProgressBar, append_metrics_jsonl, compute_rankme
from train.losses import spatial_jepa_loss, temporal_jepa_loss


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for Stage 1B Temporal JEPA training."""
    parser = argparse.ArgumentParser(
        description="Stage 1B Dual-LSTM Temporal JEPA Training for Squared Sense World Model"
    )
    parser.add_argument(
        "--spatial_ckpt",
        type=str,
        default=None,
        help="Path to Stage 1A pretrained spatial checkpoint (best.pkl). Optional in smoke test.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to dataset directory containing trajectory clips.",
    )
    parser.add_argument(
        "--region_dir",
        type=str,
        default=None,
        help="Path to directory containing baked Minecraft regions (.mmap). If provided and --data_dir is omitted, episodes are loaded or generated directly from baked terrain.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints/m1_full_temporal",
        help="Directory to save checkpoints and metrics.jsonl.",
    )
    parser.add_argument(
        "--phase1_epochs",
        type=int,
        default=15,
        help="Number of epochs for Phase 1 (frozen spatial backbone, default: 15).",
    )
    parser.add_argument(
        "--phase2_epochs",
        type=int,
        default=25,
        help="Number of epochs for Phase 2 (unfreeze top 2 blocks + spatial anchor, default: 25).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for trajectory clips (default: 8).",
    )
    parser.add_argument(
        "--clip_len",
        type=int,
        default=16,
        help="Length of trajectory clip in steps (default: 16).",
    )
    parser.add_argument(
        "--k_steps",
        type=int,
        default=4,
        help="Forward autoregressive rollout steps (default: 4).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Base learning rate for temporal optimizer (default: 3e-4).",
    )
    parser.add_argument(
        "--omega",
        type=float,
        default=0.2,
        help="Weight for auxiliary Inverse Dynamics Model (IDM) cross-entropy (default: 0.2).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Weight for VICReg variance regularizer (default: 0.1).",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.04,
        help="Weight for VICReg covariance regularizer (default: 0.04).",
    )
    parser.add_argument(
        "--anchor_weight",
        type=float,
        default=0.1,
        help="Weight for Stage 1A spatial anchor loss in Phase 2 (default: 0.1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for PRNG initialization (default: 42).",
    )
    parser.add_argument(
        "--as_uint8",
        action="store_true",
        default=True,
        help="Store voxel shells in RAM as uint8 and normalize per batch inside train_step (default: True).",
    )
    parser.add_argument(
        "--no_as_uint8",
        action="store_false",
        dest="as_uint8",
        help="Disable uint8 storage and eagerly normalize to float32 in RAM.",
    )
    parser.add_argument(
        "--drop_last",
        action="store_true",
        default=True,
        help="Enforce static batch sizes during training to prevent recompilation (default: True).",
    )
    parser.add_argument(
        "--no_drop_last",
        action="store_false",
        dest="drop_last",
        help="Allow variable-sized last batch (may cause JIT recompilation).",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=25,
        help="Step interval for mini-logs and progress bar updates (default: 25).",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Run fast smoke test (1 epoch Phase 1 + 1 epoch Phase 2 on synthetic clips, batch_size=2).",
    )

    parsed = parser.parse_args(args)
    if parsed.smoke_test:
        parsed.phase1_epochs = 1
        parsed.phase2_epochs = 1
        parsed.batch_size = 2
        parsed.clip_len = 16
        if parsed.log_every == 5:
            parsed.log_every = 2

    return parsed


@nnx.jit(static_argnames=("k_steps",))
def train_step_phase1(
    temporal_model: DualLSTMTemporalIntegrator,
    spatial_model: SpatialJEPA,
    opt_temp: nnx.Optimizer,
    shells: jax.Array,
    states: jax.Array,
    actions: jax.Array,
    k_steps: int = 4,
    omega: float = 0.2,
    alpha: float = 0.1,
    beta: float = 0.04,
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """Phase 1: Optimize DualLSTMTemporalIntegrator while keeping spatial_model completely frozen.

    Args:
        temporal_model: Trainable DualLSTMTemporalIntegrator module.
        spatial_model: Frozen pretrained SpatialJEPA module.
        opt_temp: nnx.Optimizer for temporal_model.
        shells: (B, T, 3, 16, 16, 16, 16) voxel input clips.
        states: (B, T, 12) proprioceptive states.
        actions: (B, T) discrete actions.
        k_steps: Autoregressive rollout horizon.
        omega: Weight for IDM cross-entropy loss.
        alpha: Weight for VICReg variance penalty.
        beta: Weight for VICReg covariance penalty.

    Returns:
        (loss, metrics_dict) tuple.
    """
    if shells.dtype == jnp.uint8:
        shells = shells.astype(jnp.float32) / 255.0
    b, T = shells.shape[0], shells.shape[1]

    # Extract spatial tokens with stop_gradient
    flat_shells = shells.reshape((b * T, 3, 16, 16, 16, 16))
    tokens = jax.lax.stop_gradient(spatial_model.context_encoder(flat_shells, visible_idx=None))
    tokens_seq = tokens.reshape((b, T, 192, 256))

    # Extract normalized EMA targets with stop_gradient
    target_tokens = jax.lax.stop_gradient(spatial_model.target_encoder(flat_shells, visible_idx=None))
    mean_pooled = jnp.mean(target_tokens, axis=1)  # (b * T, 256)
    mean = jnp.mean(mean_pooled, axis=-1, keepdims=True)
    var = jnp.var(mean_pooled, axis=-1, keepdims=True)
    norm_pooled = (mean_pooled - mean) / jnp.sqrt(var + 1e-6)
    ema_targets_seq = norm_pooled.reshape((b, T, 256))

    T_valid = T - k_steps + 1
    slices = [ema_targets_seq[:, t : t + k_steps] for t in range(T_valid)]
    ema_future_targets = jnp.stack(slices, axis=1)

    def loss_fn(t_m: DualLSTMTemporalIntegrator):
        pred_future, idm_logits = t_m.rollout(tokens_seq, states, actions, k_steps=k_steps)
        t_loss, t_metrics = temporal_jepa_loss(
            pred_future=pred_future,
            ema_future_targets=ema_future_targets,
            idm_logits=idm_logits,
            actions=actions,
            omega=omega,
            alpha=alpha,
            beta=beta,
        )
        return t_loss, t_metrics

    (loss, metrics), grads_temp = nnx.value_and_grad(loss_fn, has_aux=True)(temporal_model)
    opt_temp.update(temporal_model, grads_temp)
    return loss, metrics


@nnx.jit(static_argnames=("k_steps",))
def train_step_phase2(
    temporal_model: DualLSTMTemporalIntegrator,
    spatial_model: SpatialJEPA,
    opt_temp: nnx.Optimizer,
    opt_spat: nnx.Optimizer,
    shells: jax.Array,
    states: jax.Array,
    actions: jax.Array,
    rng_key: jax.Array,
    k_steps: int = 4,
    omega: float = 0.2,
    alpha: float = 0.1,
    beta: float = 0.04,
    anchor_weight: float = 0.1,
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """Phase 2: Jointly optimize temporal model and top 2 Transformer blocks + final norm of context encoder.

    Args:
        temporal_model: DualLSTMTemporalIntegrator module.
        spatial_model: SpatialJEPA module (blocks[-2:] and norm unfrozen).
        opt_temp: nnx.Optimizer for temporal_model.
        opt_spat: nnx.Optimizer targeting ONLY blocks[-2:] and norm.
        shells: (B, T, 3, 16, 16, 16, 16) voxel input clips.
        states: (B, T, 12) proprioceptive states.
        actions: (B, T) discrete actions.
        rng_key: JAX PRNG key for anchor frame and mask sampling.
        k_steps: Autoregressive rollout horizon.
        omega: Weight for IDM cross-entropy loss.
        alpha: Weight for VICReg variance penalty.
        beta: Weight for VICReg covariance penalty.
        anchor_weight: Weight for Stage 1A spatial anchor loss.

    Returns:
        (total_loss, metrics_dict) tuple.
    """
    if shells.dtype == jnp.uint8:
        shells = shells.astype(jnp.float32) / 255.0
    b, T = shells.shape[0], shells.shape[1]
    k_frame, k_mask = jax.random.split(rng_key)
    frame_idx = jax.random.randint(k_frame, (), 0, T)
    x_anchor = jnp.take(shells, frame_idx, axis=1)
    vis_idx, tgt_idx = sample_3d_jepa_masks(k_mask, b, num_shells=spatial_model.num_shells, cross_scale_prob=0.3)

    def loss_fn(t_m: DualLSTMTemporalIntegrator, s_m: SpatialJEPA):
        # 1. Temporal rollout loss
        flat_shells = shells.reshape((b * T, 3, 16, 16, 16, 16))
        tokens = s_m.context_encoder(flat_shells, visible_idx=None)
        tokens_seq = tokens.reshape((b, T, 192, 256))

        target_tokens = jax.lax.stop_gradient(s_m.target_encoder(flat_shells, visible_idx=None))
        mean_pooled = jnp.mean(target_tokens, axis=1)
        mean = jnp.mean(mean_pooled, axis=-1, keepdims=True)
        var = jnp.var(mean_pooled, axis=-1, keepdims=True)
        norm_pooled = (mean_pooled - mean) / jnp.sqrt(var + 1e-6)
        ema_targets_seq = norm_pooled.reshape((b, T, 256))

        T_valid = T - k_steps + 1
        slices = [ema_targets_seq[:, t : t + k_steps] for t in range(T_valid)]
        ema_future_targets = jnp.stack(slices, axis=1)

        pred_future, idm_logits = t_m.rollout(tokens_seq, states, actions, k_steps=k_steps)
        t_loss, t_metrics = temporal_jepa_loss(
            pred_future=pred_future,
            ema_future_targets=ema_future_targets,
            idm_logits=idm_logits,
            actions=actions,
            omega=omega,
            alpha=alpha,
            beta=beta,
        )

        # 2. Stage 1A Spatial anchor loss on random frame
        pred_tgt, norm_ema, ctx_tok = s_m(x_anchor, vis_idx, tgt_idx)
        spat_loss, spat_metrics = spatial_jepa_loss(pred_tgt, norm_ema, ctx_tok, alpha=alpha, beta=beta)

        total_loss = t_loss + anchor_weight * spat_loss
        metrics = {
            "loss": total_loss,
            "temporal_loss": t_loss,
            "spatial_anchor_loss": spat_loss,
            "rollout_l1": t_metrics["rollout_l1"],
            "idm_loss": t_metrics["idm_loss"],
        }
        return total_loss, metrics

    (loss, metrics), (grads_temp, grads_spat) = nnx.value_and_grad(
        loss_fn, argnums=(0, 1), has_aux=True
    )(temporal_model, spatial_model)

    # Update temporal model
    opt_temp.update(temporal_model, grads_temp)

    # Strictly update ONLY top 2 Transformer blocks and final norm of context encoder
    target_top = (
        spatial_model.context_encoder.blocks[2],
        spatial_model.context_encoder.blocks[3],
        spatial_model.context_encoder.norm,
    )
    grads_top = (
        grads_spat.context_encoder.blocks[2],
        grads_spat.context_encoder.blocks[3],
        grads_spat.context_encoder.norm,
    )
    opt_spat.update(target_top, grads_top)

    # Spatial EMA target update
    spatial_model.update_ema(0.999)

    return loss, metrics


@nnx.jit
def extract_context_states(
    temporal_model: DualLSTMTemporalIntegrator,
    tokens_seq: jax.Array,
    state_seq: jax.Array,
    action_seq: jax.Array,
) -> jax.Array:
    """Extract fused context representations c_t across trajectory steps.

    Args:
        temporal_model: DualLSTMTemporalIntegrator module.
        tokens_seq: (B, T, 192, 256) sequence of voxel tokens.
        state_seq: (B, T, 12) sequence of states.
        action_seq: (B, T) sequence of actions.

    Returns:
        (B, T, 192) fused context representations.
    """
    b, T = tokens_seq.shape[:2]
    hidden = temporal_model.init_hidden(b)
    c_list = []
    for t in range(T):
        _, c_t, hidden = temporal_model.step(
            tokens_seq[:, t], state_seq[:, t], action_seq[:, t], hidden
        )
        c_list.append(c_t)
    return jnp.stack(c_list, axis=1)


@nnx.jit(static_argnames=("k_steps",))
def val_step(
    temporal_model: DualLSTMTemporalIntegrator,
    spatial_model: SpatialJEPA,
    shells: jax.Array,
    states: jax.Array,
    actions: jax.Array,
    k_steps: int = 4,
    omega: float = 0.2,
    alpha: float = 0.1,
    beta: float = 0.04,
) -> Tuple[jax.Array, Dict[str, jax.Array], jax.Array]:
    """Validation evaluation step computing temporal loss metrics and context states.

    Returns:
        (val_loss, metrics_dict, c_seq) tuple.
    """
    if shells.dtype == jnp.uint8:
        shells = shells.astype(jnp.float32) / 255.0
    b, T = shells.shape[0], shells.shape[1]
    flat_shells = shells.reshape((b * T, 3, 16, 16, 16, 16))
    tokens = spatial_model.context_encoder(flat_shells, visible_idx=None)
    tokens_seq = tokens.reshape((b, T, 192, 256))

    target_tokens = spatial_model.target_encoder(flat_shells, visible_idx=None)
    mean_pooled = jnp.mean(target_tokens, axis=1)
    mean = jnp.mean(mean_pooled, axis=-1, keepdims=True)
    var = jnp.var(mean_pooled, axis=-1, keepdims=True)
    norm_pooled = (mean_pooled - mean) / jnp.sqrt(var + 1e-6)
    ema_targets_seq = norm_pooled.reshape((b, T, 256))

    T_valid = T - k_steps + 1
    slices = [ema_targets_seq[:, t : t + k_steps] for t in range(T_valid)]
    ema_future_targets = jnp.stack(slices, axis=1)

    pred_future, idm_logits = temporal_model.rollout(tokens_seq, states, actions, k_steps=k_steps)
    t_loss, metrics = temporal_jepa_loss(
        pred_future=pred_future,
        ema_future_targets=ema_future_targets,
        idm_logits=idm_logits,
        actions=actions,
        omega=omega,
        alpha=alpha,
        beta=beta,
    )

    c_seq = extract_context_states(temporal_model, tokens_seq, states, actions)
    return t_loss, metrics, c_seq


def load_temporal_dataset(
    data_dir: Optional[str],
    smoke_test: bool,
    seed: int,
    clip_len: int = 16,
    as_uint8: bool = True,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Load or synthetically generate trajectory clips.

    Returns:
        Tuple of (train_dict, val_dict) with keys 'shells', 'states', 'actions'.
    """
    if smoke_test or data_dir is None:
        rng = np.random.default_rng(seed)
        n_train = 4
        n_val = 2
        if as_uint8:
            train_shells = rng.integers(0, 256, size=(n_train, clip_len, 3, 16, 16, 16, 16), dtype=np.uint8)
            val_shells = rng.integers(0, 256, size=(n_val, clip_len, 3, 16, 16, 16, 16), dtype=np.uint8)
        else:
            train_shells = rng.uniform(0.0, 1.0, size=(n_train, clip_len, 3, 16, 16, 16, 16)).astype(np.float32)
            val_shells = rng.uniform(0.0, 1.0, size=(n_val, clip_len, 3, 16, 16, 16, 16)).astype(np.float32)
        train_data = {
            "shells": train_shells,
            "states": rng.normal(0.0, 1.0, size=(n_train, clip_len, 12)).astype(np.float32),
            "actions": rng.integers(0, 12, size=(n_train, clip_len), dtype=np.int32),
        }
        val_data = {
            "shells": val_shells,
            "states": rng.normal(0.0, 1.0, size=(n_val, clip_len, 12)).astype(np.float32),
            "actions": rng.integers(0, 12, size=(n_val, clip_len), dtype=np.int32),
        }
        return train_data, val_data

    splits = load_episode_splits(data_dir, seed=seed, as_uint8=as_uint8)
    train_data = extract_temporal_clips(splits["train"], clip_len=clip_len, stride=4)
    val_data = extract_temporal_clips(splits["val"], clip_len=clip_len, stride=4)
    return train_data, val_data


def train_temporal_jepa(args: argparse.Namespace) -> Tuple[DualLSTMTemporalIntegrator, SpatialJEPA]:
    """Run full two-phase Stage 1B Temporal JEPA training pipeline."""
    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(args.save_dir, "train.log")
    metrics_path = os.path.join(args.save_dir, "metrics.jsonl")
    temp_best_ckpt = os.path.join(args.save_dir, "temporal_best.pkl")
    spat_best_ckpt = os.path.join(args.save_dir, "spatial_finetuned_best.pkl")

    backend = jax.default_backend()
    if backend == "gpu":
        from train.gpu_env import query_gpu_vram
        vram = query_gpu_vram()
        if vram and (vram["total_mb"] <= 4600 or vram["free_mb"] < 3000):
            if args.batch_size > 4 and not args.smoke_test:
                print(
                    f"[VRAM Auto-Config] Detected {vram['total_mb']} MiB GPU ({vram['free_mb']} MiB free). "
                    f"Adjusting temporal batch_size from {args.batch_size} to 4 for sequence memory safety."
                )
                args.batch_size = 4

    root_key = jax.random.key(args.seed)
    k_spat, k_temp, train_key = jax.random.split(root_key, 3)

    # 1. Initialize models
    spatial_model = SpatialJEPA(num_shells=3, rngs=nnx.Rngs(int(jax.random.bits(k_spat))))
    if args.spatial_ckpt and os.path.isfile(args.spatial_ckpt):
        print(f"Loading Stage 1A pretrained spatial checkpoint: {args.spatial_ckpt}")
        load_checkpoint(args.spatial_ckpt, spatial_model)

    temporal_model = DualLSTMTemporalIntegrator(
        token_dim=256,
        spatial_hidden=192,
        state_dim=12,
        state_hidden=64,
        num_actions=12,
        num_heads=6,
        rngs=nnx.Rngs(int(jax.random.bits(k_temp))),
    )

    # 2. Load dataset
    effective_data_dir = args.data_dir or args.region_dir
    as_uint8 = getattr(args, "as_uint8", True)
    train_dict, val_dict = load_temporal_dataset(
        effective_data_dir, args.smoke_test, args.seed, args.clip_len, as_uint8=as_uint8
    )
    num_train = len(train_dict["shells"])
    num_val = len(val_dict["shells"])
    drop_last = getattr(args, "drop_last", True)
    if drop_last and num_train >= args.batch_size:
        steps_per_epoch = max(1, num_train // args.batch_size)
    else:
        steps_per_epoch = max(1, math.ceil(num_train / args.batch_size))
    if drop_last and num_val >= args.batch_size:
        val_steps = num_val // args.batch_size
    else:
        val_steps = max(1, math.ceil(num_val / args.batch_size))

    # 3. Setup Optimizers
    # Temporal optimizer
    tx_temp = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=args.lr, weight_decay=0.05),
    )
    opt_temp = nnx.Optimizer(temporal_model, tx_temp, wrt=nnx.Param)

    # Phase 2 Spatial optimizer: 0.1 * lr (3e-5) targeting ONLY blocks[-2:] and norm
    top_blocks = (
        spatial_model.context_encoder.blocks[2],
        spatial_model.context_encoder.blocks[3],
        spatial_model.context_encoder.norm,
    )
    tx_spat = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=args.lr * 0.1, weight_decay=0.05),
    )
    opt_spat = nnx.Optimizer(top_blocks, tx_spat, wrt=nnx.Param)

    best_val_loss = float("inf")
    global_step = 0

    print(
        f"Starting Stage 1B Temporal JEPA Training:\n"
        f"  Phase 1 Epochs: {args.phase1_epochs} | Phase 2 Epochs: {args.phase2_epochs}\n"
        f"  Batch Size: {args.batch_size} | Clip Len: {args.clip_len} | K-Steps: {args.k_steps}\n"
        f"  Base LR: {args.lr} | Phase 2 Spatial LR: {args.lr * 0.1}\n"
        f"  Train Clips: {num_train} | Val Clips: {num_val} | Dtype: {train_dict['shells'].dtype} | Drop Last: {drop_last}\n"
        f"  Save Dir: {args.save_dir}\n",
        flush=True,
    )

    total_phases = [(1, args.phase1_epochs), (2, args.phase2_epochs)]
    for phase_num, num_epochs in total_phases:
        print(f"\n--- Entering Phase {phase_num} ({num_epochs} Epochs) ---", flush=True)
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            epoch_rng = np.random.default_rng(args.seed + phase_num * 1000 + epoch)
            perm = epoch_rng.permutation(num_train)

            train_loss_accum = 0.0
            rollout_l1_accum = 0.0
            idm_loss_accum = 0.0

            pbar = ProgressBar(
                total=steps_per_epoch,
                prefix=f"P{phase_num} Ep {epoch:02d}/{num_epochs:02d}",
                log_every=args.log_every,
                log_file=log_path,
                process_tag=f"TemporalJEPA_P{phase_num}",
            )

            for s in range(steps_per_epoch):
                idx = perm[s * args.batch_size : (s + 1) * args.batch_size]
                b_shells = jnp.asarray(train_dict["shells"][idx])
                b_states = jnp.asarray(train_dict["states"][idx])
                b_actions = jnp.asarray(train_dict["actions"][idx])
                step_key = jax.random.fold_in(train_key, global_step)

                if phase_num == 1:
                    loss, metrics = train_step_phase1(
                        temporal_model=temporal_model,
                        spatial_model=spatial_model,
                        opt_temp=opt_temp,
                        shells=b_shells,
                        states=b_states,
                        actions=b_actions,
                        k_steps=args.k_steps,
                        omega=args.omega,
                        alpha=args.alpha,
                        beta=args.beta,
                    )
                else:
                    loss, metrics = train_step_phase2(
                        temporal_model=temporal_model,
                        spatial_model=spatial_model,
                        opt_temp=opt_temp,
                        opt_spat=opt_spat,
                        shells=b_shells,
                        states=b_states,
                        actions=b_actions,
                        rng_key=step_key,
                        k_steps=args.k_steps,
                        omega=args.omega,
                        alpha=args.alpha,
                        beta=args.beta,
                        anchor_weight=args.anchor_weight,
                    )

                pbar.update(
                    s + 1,
                    {
                        "Loss": loss,
                        "RolloutL1": metrics["rollout_l1"],
                        "IDM": metrics["idm_loss"],
                    },
                )

                train_loss_accum += float(loss)
                rollout_l1_accum += float(metrics["rollout_l1"])
                idm_loss_accum += float(metrics["idm_loss"])
                global_step += 1

            avg_train_loss = train_loss_accum / steps_per_epoch
            avg_rollout_l1 = rollout_l1_accum / steps_per_epoch
            avg_idm_loss = idm_loss_accum / steps_per_epoch

            # Validation Loop
            val_loss_accum = 0.0
            val_c_list = []
            val_pbar = ProgressBar(
                total=val_steps,
                prefix=f"P{phase_num} Val {epoch:02d}/{num_epochs:02d}",
                log_every=args.log_every,
                log_file=log_path,
                process_tag=f"TemporalJEPA_P{phase_num}",
            )
            for vs in range(val_steps):
                v_idx = np.arange(vs * args.batch_size, min((vs + 1) * args.batch_size, num_val))
                v_shells = jnp.asarray(val_dict["shells"][v_idx])
                v_states = jnp.asarray(val_dict["states"][v_idx])
                v_actions = jnp.asarray(val_dict["actions"][v_idx])

                v_loss, v_metrics, c_seq = val_step(
                    temporal_model=temporal_model,
                    spatial_model=spatial_model,
                    shells=v_shells,
                    states=v_states,
                    actions=v_actions,
                    k_steps=args.k_steps,
                    omega=args.omega,
                    alpha=args.alpha,
                    beta=args.beta,
                )
                val_pbar.update(
                    vs + 1,
                    {
                        "ValLoss": v_loss,
                        "RolloutL1": v_metrics["rollout_l1"],
                        "IDM": v_metrics["idm_loss"],
                    },
                )
                val_loss_accum += float(v_loss)
                val_c_list.append(np.asarray(c_seq))

            avg_val_loss = val_loss_accum / val_steps

            # RankMe on validation fused context representations c_t
            all_val_c = np.concatenate(val_c_list, axis=0)  # (N_val, T, 192)
            rankme_val = float(compute_rankme(all_val_c))

            # Record metrics
            record = {
                "phase": phase_num,
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "rollout_l1": avg_rollout_l1,
                "idm_loss": avg_idm_loss,
                "rankme": rankme_val,
                "epoch_sec": time.time() - epoch_start,
            }
            append_metrics_jsonl(metrics_path, record)

            # Checkpoint saving
            metadata = {
                "phase": phase_num,
                "epoch": epoch,
                "global_step": global_step,
                "val_loss": avg_val_loss,
                "rankme": rankme_val,
            }

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_checkpoint(temp_best_ckpt, temporal_model, metadata=metadata)
                if phase_num == 2:
                    save_checkpoint(spat_best_ckpt, spatial_model, metadata=metadata)

            # Console output
            now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            epoch_line = (
                f"[{now_ts}] [TemporalJEPA_P{phase_num} PID {os.getpid()}] "
                f"P{phase_num} Epoch [{epoch:02d}/{num_epochs:02d}] "
                f"Step {global_step:04d} | "
                f"Train Loss: {avg_train_loss:6.4f} | "
                f"Val Loss: {avg_val_loss:6.4f} | "
                f"Rollout L1: {avg_rollout_l1:5.3f} | "
                f"IDM Loss: {avg_idm_loss:5.3f} | "
                f"RankMe: {rankme_val:6.2f}"
            )
            print(epoch_line, flush=True)
            if log_path:
                try:
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(epoch_line + "\n")
                except Exception:
                    pass

    # Final checkpoints
    save_checkpoint(os.path.join(args.save_dir, "temporal_last.pkl"), temporal_model, metadata=metadata)
    save_checkpoint(os.path.join(args.save_dir, "spatial_last.pkl"), spatial_model, metadata=metadata)
    if not os.path.exists(spat_best_ckpt):
        save_checkpoint(spat_best_ckpt, spatial_model, metadata=metadata)

    print(f"\nStage 1B Training Completed! Best Val Loss: {best_val_loss:6.4f}", flush=True)
    return temporal_model, spatial_model


def main():
    """CLI entrypoint."""
    args = parse_args()
    train_temporal_jepa(args)


if __name__ == "__main__":
    main()

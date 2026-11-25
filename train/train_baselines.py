"""Unified Training Pipeline for Parameter-Matched Baselines.

Supports:
  - b1_lod0: SingleLODJEPA (LOD-0 shell only, 64 tokens, parameter-matched to Stage 1A)
  - b2_lod2: SingleLODJEPA (LOD-2 shell only, 64 tokens, parameter-matched to Stage 1A)
  - b3_mae : Generative3DMAE (Multi-LOD encoder + lightweight 3D patch decoder)

Strictly follows docs/TRAIN_CONTRACT.md and docs/MODEL_CONTRACT.md.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from train.gpu_env import auto_configure_batching, setup_gpu_environment

# Ensure non-greedy allocation and VRAM limits are set prior to XLA initialization
setup_gpu_environment()

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from models.baselines import (
    Generative3DMAE,
    SingleLODJEPA,
    Uniform32JEPA,
    sample_3d_jepa_masks_32,
)
from models.transformer import sample_3d_jepa_masks
from train.checkpoint import load_checkpoint, save_checkpoint
from train.diagnostics import ProgressBar, append_metrics_jsonl, compute_rankme
from train.losses import cosine_ema_momentum, spatial_jepa_loss
from train.train_spatial_jepa import load_dataset


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for baseline training."""
    parser = argparse.ArgumentParser(
        description="Unified Baselines Training for Squared Sense World Model"
    )
    parser.add_argument(
        "--model",
        "--baseline",
        dest="model",
        type=str,
        required=True,
        choices=["b1_lod0", "b2_lod2", "b3_mae", "b4_uniform32"],
        help="Baseline model architecture variant to train.",
    )
    parser.add_argument(
        "--allow_upsample_fallback",
        action="store_true",
        help="Allow 2x upsampling from LOD-1 for b4_uniform32 when --region_dir is omitted (unit tests only).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to directory containing dataset (.npz files).",
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
        default=None,
        help="Directory to save checkpoints and metrics.jsonl (defaults to checkpoints/baselines/{model}).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=80,
        help="Number of training epochs (default: 80).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training and validation (default: 32).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Peak learning rate for AdamW (default: 3e-4).",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.05,
        help="Weight decay for AdamW (default: 0.05).",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=10,
        help="Number of linear warmup epochs (default: 10).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Weight for VICReg variance regularizer in JEPA baselines (default: 0.1).",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.04,
        help="Weight for VICReg covariance regularizer in JEPA baselines (default: 0.04).",
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
        help="Run fast smoke test (2 epochs on 16 synthetic random frames, batch_size=4).",
    )

    parsed = parser.parse_args(args)
    if parsed.save_dir is None:
        parsed.save_dir = f"checkpoints/baselines/{parsed.model}"

    if parsed.smoke_test:
        parsed.epochs = 2
        parsed.batch_size = 4
        parsed.warmup_epochs = 1
        if parsed.log_every == 5:
            parsed.log_every = 2

    return parsed


# ==============================================================================
# JIT-Compiled Step Functions for SingleLODJEPA
# ==============================================================================


@nnx.jit
def train_step_single_lod(
    model: SingleLODJEPA,
    optimizer: nnx.Optimizer,
    x: jax.Array,
    rng_key: jax.Array,
    ema_momentum: float,
    alpha: float = 0.1,
    beta: float = 0.04,
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """Single optimization step for SingleLODJEPA baseline."""
    if x.dtype == jnp.uint8:
        x = x.astype(jnp.float32) / 255.0
    b = x.shape[0]
    vis_idx, tgt_idx = sample_3d_jepa_masks(rng_key, b, num_shells=1)

    def loss_fn(m: SingleLODJEPA):
        pred_targets, norm_ema, context_tokens = m(x, vis_idx, tgt_idx)
        loss, metrics = spatial_jepa_loss(pred_targets, norm_ema, context_tokens, alpha=alpha, beta=beta)
        return loss, metrics

    (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)

    # Strictly update ONLY context_encoder and predictor parameters
    optimizer.update(
        (model.context_encoder, model.predictor),
        (grads.jepa.context_encoder, grads.jepa.predictor),
    )
    model.update_ema(ema_momentum)

    return loss, metrics


@nnx.jit
def val_step_single_lod(
    model: SingleLODJEPA,
    x: jax.Array,
    rng_key: jax.Array,
    alpha: float = 0.1,
    beta: float = 0.04,
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """Evaluation step for SingleLODJEPA baseline."""
    if x.dtype == jnp.uint8:
        x = x.astype(jnp.float32) / 255.0
    b = x.shape[0]
    vis_idx, tgt_idx = sample_3d_jepa_masks(rng_key, b, num_shells=1)
    pred_targets, norm_ema, context_tokens = model(x, vis_idx, tgt_idx)
    loss, metrics = spatial_jepa_loss(pred_targets, norm_ema, context_tokens, alpha=alpha, beta=beta)
    emb_std = jnp.mean(jnp.std(context_tokens, axis=0))
    metrics["emb_std"] = emb_std
    return loss, metrics


@nnx.jit
def extract_single_lod_embeddings(model: SingleLODJEPA, x: jax.Array) -> jax.Array:
    """Extract unmasked context representations pooled across single LOD tokens."""
    if x.dtype == jnp.uint8:
        x = x.astype(jnp.float32) / 255.0
    if x.ndim == 6:
        x_single = x[:, model.lod_index : model.lod_index + 1]
    elif x.ndim == 5:
        x_single = jnp.expand_dims(x, axis=1)
    else:
        x_single = x
    tokens = model.context_encoder(x_single, visible_idx=None)  # (B, 64, 256)
    return jnp.mean(tokens, axis=1)  # (B, 256)


# ==============================================================================
# JIT-Compiled Step Functions for Generative3DMAE
# ==============================================================================


@nnx.jit
def train_step_mae(
    model: Generative3DMAE,
    optimizer: nnx.Optimizer,
    x: jax.Array,
    rng_key: jax.Array,
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """Single optimization step for Generative3DMAE baseline."""
    if x.dtype == jnp.uint8:
        x = x.astype(jnp.float32) / 255.0
    b = x.shape[0]
    vis_idx, tgt_idx = sample_3d_jepa_masks(rng_key, b, num_shells=3, cross_scale_prob=0.3)

    def loss_fn(m: Generative3DMAE):
        _, _, mse_loss = m(x, vis_idx, tgt_idx)
        return mse_loss, {"loss": mse_loss, "mse_loss": mse_loss}

    (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)

    return loss, metrics


@nnx.jit
def val_step_mae(
    model: Generative3DMAE,
    x: jax.Array,
    rng_key: jax.Array,
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """Evaluation step for Generative3DMAE baseline."""
    if x.dtype == jnp.uint8:
        x = x.astype(jnp.float32) / 255.0
    b = x.shape[0]
    vis_idx, tgt_idx = sample_3d_jepa_masks(rng_key, b, num_shells=3, cross_scale_prob=0.3)
    _, _, mse_loss = model(x, vis_idx, tgt_idx)
    metrics = {"loss": mse_loss, "mse_loss": mse_loss}
    return mse_loss, metrics


@nnx.jit
def extract_mae_embeddings(model: Generative3DMAE, x: jax.Array) -> jax.Array:
    """Extract unmasked representations from MAE encoder pooled across tokens."""
    if x.dtype == jnp.uint8:
        x = x.astype(jnp.float32) / 255.0
    tokens = model.encoder(x, visible_idx=None)  # (B, 192, 256)
    return jnp.mean(tokens, axis=1)  # (B, 256)


# ==============================================================================
# JIT-Compiled Step Functions for Uniform32JEPA (B4)
# ==============================================================================


@nnx.jit
def train_step_uniform32(
    model: Uniform32JEPA,
    optimizer: nnx.Optimizer,
    x: jax.Array,
    rng_key: jax.Array,
    ema_momentum: float,
    alpha: float = 0.1,
    beta: float = 0.04,
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """Single optimization step for Uniform32JEPA baseline."""
    if x.dtype == jnp.uint8:
        x = x.astype(jnp.float32) / 255.0
    b = x.shape[0]
    vis_idx, tgt_idx = sample_3d_jepa_masks_32(rng_key, b)

    def loss_fn(m: Uniform32JEPA):
        pred_targets, norm_ema, context_tokens = m(x, vis_idx, tgt_idx)
        loss, metrics = spatial_jepa_loss(pred_targets, norm_ema, context_tokens, alpha=alpha, beta=beta)
        return loss, metrics

    (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(
        (model.context_encoder, model.predictor),
        (grads.jepa.context_encoder, grads.jepa.predictor),
    )
    model.update_ema(ema_momentum)
    return loss, metrics


@nnx.jit
def val_step_uniform32(
    model: Uniform32JEPA,
    x: jax.Array,
    rng_key: jax.Array,
    alpha: float = 0.1,
    beta: float = 0.04,
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """Evaluation step for Uniform32JEPA baseline."""
    if x.dtype == jnp.uint8:
        x = x.astype(jnp.float32) / 255.0
    b = x.shape[0]
    vis_idx, tgt_idx = sample_3d_jepa_masks_32(rng_key, b)
    pred_targets, norm_ema, context_tokens = model(x, vis_idx, tgt_idx)
    loss, metrics = spatial_jepa_loss(pred_targets, norm_ema, context_tokens, alpha=alpha, beta=beta)
    metrics["emb_std"] = jnp.mean(jnp.std(context_tokens, axis=0))
    return loss, metrics


@nnx.jit
def extract_uniform32_embeddings(model: Uniform32JEPA, x: jax.Array) -> jax.Array:
    """Extract unmasked context representations pooled across 512 tokens."""
    if x.dtype == jnp.uint8:
        x = x.astype(jnp.float32) / 255.0
    if x.ndim == 5:
        x = jnp.expand_dims(x, axis=1)
    tokens = model.context_encoder(x, visible_idx=None)  # (B, 512, 256)
    return jnp.mean(tokens, axis=1)  # (B, 256)


def load_uniform32_dataset(
    data_dir: Optional[str],
    region_dir: Optional[str],
    smoke_test: bool,
    seed: int,
    as_uint8: bool = True,
    allow_upsample_fallback: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load true 1x-resolution (N, 32, 32, 32, 16) volumes for B4 Uniform32JEPA.

    Requires --region_dir containing baked .mmap regions unless --smoke_test or
    --allow_upsample_fallback is explicitly set.
    """
    if smoke_test:
        rng = np.random.default_rng(seed)
        if as_uint8:
            return (
                rng.integers(0, 256, size=(8, 32, 32, 32, 16), dtype=np.uint8),
                rng.integers(0, 256, size=(4, 32, 32, 32, 16), dtype=np.uint8),
            )
        return (
            rng.uniform(0.0, 1.0, size=(8, 32, 32, 32, 16)).astype(np.float32),
            rng.uniform(0.0, 1.0, size=(4, 32, 32, 32, 16)).astype(np.float32),
        )

    if region_dir and os.path.isdir(region_dir):
        from env.lod_extractor import WorldVolume, extract_uniform32_16ch
        from env.palette import get_palette

        palette = get_palette()
        world = WorldVolume(region_dir, palette_obj=palette)

        def _yaw_to_cardinal(sin_y: float, cos_y: float) -> str:
            yaw = (math.degrees(math.atan2(-sin_y, cos_y)) % 360.0 + 360.0) % 360.0
            if 45.0 <= yaw < 135.0:
                return "WEST"
            if 135.0 <= yaw < 225.0:
                return "NORTH"
            if 225.0 <= yaw < 315.0:
                return "EAST"
            return "SOUTH"

        if data_dir and os.path.isdir(data_dir):
            from data.dataset import load_episode_splits
            splits = load_episode_splits(data_dir, seed=seed, as_uint8=as_uint8)
            if len(splits["train"]) > 0:
                def _extract_split(eps):
                    vols = []
                    for ep in eps:
                        meta = ep["meta"]
                        st = ep["state"]
                        for t in range(len(meta)):
                            px, py, pz = int(round(meta[t, 0])), int(round(meta[t, 1])), int(round(meta[t, 2]))
                            card = _yaw_to_cardinal(float(st[t, 6]), float(st[t, 7]))
                            pitch = float(st[t, 8])
                            vols.append(extract_uniform32_16ch(world, palette, (px, py, pz), card, pitch, as_uint8=as_uint8))
                    return np.stack(vols, axis=0)

                train_v = _extract_split(splits["train"])
                val_eps = splits["val"] if len(splits["val"]) > 0 else splits["train"][:1]
                val_v = _extract_split(val_eps)
                return train_v, val_v

        # Fallback to sampling traversable positions directly from baked region spawns
        import glob
        from env.spawn_indexer import SpawnSampler
        spawns_files = glob.glob(os.path.join(region_dir, "*.spawns.npz"))
        if not spawns_files:
            raise FileNotFoundError(f"No .spawns.npz found in region_dir '{region_dir}' for b4_uniform32.")
        sampler = SpawnSampler.from_npz(spawns_files[0])
        rng = np.random.default_rng(seed)
        cards = ["SOUTH", "WEST", "NORTH", "EAST"]
        train_v = np.stack([
            extract_uniform32_16ch(world, palette, sampler.sample(rng=rng), str(rng.choice(cards)), 0.0, as_uint8=as_uint8)
            for _ in range(512)
        ], axis=0)
        val_v = np.stack([
            extract_uniform32_16ch(world, palette, sampler.sample(rng=rng), str(rng.choice(cards)), 0.0, as_uint8=as_uint8)
            for _ in range(128)
        ], axis=0)
        return train_v, val_v

    if allow_upsample_fallback and data_dir:
        tr, va = load_dataset(data_dir, smoke_test=False, seed=seed, as_uint8=as_uint8)
        tr32 = np.repeat(np.repeat(np.repeat(tr[:, 1], 2, axis=1), 2, axis=2), 2, axis=3)
        va32 = np.repeat(np.repeat(np.repeat(va[:, 1], 2, axis=1), 2, axis=2), 2, axis=3)
        return tr32, va32

    raise ValueError(
        "b4_uniform32 requires --region_dir pointing to baked Minecraft regions (.mmap) "
        "for true 1x 32x32x32x16 extraction. Pass --allow_upsample_fallback only for unit tests."
    )


# ==============================================================================
# Unified Baseline Training Pipeline
# ==============================================================================


def train_baseline(args: argparse.Namespace) -> Tuple[Union[SingleLODJEPA, Generative3DMAE, Uniform32JEPA], float]:
    """Execute end-to-end training pipeline for the selected baseline model."""
    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(args.save_dir, "train.log")
    metrics_path = os.path.join(args.save_dir, "metrics.jsonl")
    best_ckpt_path = os.path.join(args.save_dir, "best.pkl")
    last_ckpt_path = os.path.join(args.save_dir, "last.pkl")

    backend = jax.default_backend()
    args.batch_size, args.micro_batch_size = auto_configure_batching(
        batch_size=args.batch_size,
        micro_batch_size=getattr(args, "micro_batch_size", None),
        backend=backend,
    )

    root_key = jax.random.key(args.seed)
    model_key, train_key = jax.random.split(root_key)

    # 1. Instantiate Model & Optimizer
    is_mae = args.model == "b3_mae"
    is_u32 = args.model == "b4_uniform32"
    if args.model == "b1_lod0":
        model: Union[SingleLODJEPA, Generative3DMAE, Uniform32JEPA] = SingleLODJEPA(
            lod_index=0, rngs=nnx.Rngs(int(jax.random.bits(model_key)))
        )
    elif args.model == "b2_lod2":
        model = SingleLODJEPA(
            lod_index=2, rngs=nnx.Rngs(int(jax.random.bits(model_key)))
        )
    elif args.model == "b3_mae":
        model = Generative3DMAE(rngs=nnx.Rngs(int(jax.random.bits(model_key))))
    elif args.model == "b4_uniform32":
        model = Uniform32JEPA(rngs=nnx.Rngs(int(jax.random.bits(model_key))))
    else:
        raise ValueError(f"Unknown baseline model: {args.model}")

    # 2. Load Data
    effective_data_dir = args.data_dir or args.region_dir
    as_uint8 = getattr(args, "as_uint8", True)
    if is_u32:
        train_x, val_x = load_uniform32_dataset(
            data_dir=args.data_dir,
            region_dir=args.region_dir,
            smoke_test=args.smoke_test,
            seed=args.seed,
            as_uint8=as_uint8,
            allow_upsample_fallback=getattr(args, "allow_upsample_fallback", False),
        )
    else:
        train_x, val_x = load_dataset(effective_data_dir, args.smoke_test, args.seed, as_uint8=as_uint8)
    num_train = len(train_x)
    num_val = len(val_x)
    drop_last = getattr(args, "drop_last", True)
    if drop_last and num_train >= args.batch_size:
        steps_per_epoch = max(1, num_train // args.batch_size)
    else:
        steps_per_epoch = max(1, math.ceil(num_train / args.batch_size))
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch

    # 3. Schedule & Optimizer Setup
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=args.lr * 0.01,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=args.weight_decay),
    )

    if is_mae:
        optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    else:
        assert isinstance(model, (SingleLODJEPA, Uniform32JEPA))
        optimizer = nnx.Optimizer((model.context_encoder, model.predictor), tx, wrt=nnx.Param)

    best_val_loss = float("inf")
    global_step = 0

    print(
        f"Starting Baseline Training [{args.model}]:\n"
        f"  Epochs: {args.epochs} | Steps/Epoch: {steps_per_epoch} | Total Steps: {total_steps}\n"
        f"  Batch Size: {args.batch_size} | Peak LR: {args.lr} | Warmup Epochs: {args.warmup_epochs}\n"
        f"  Train Samples: {num_train} | Val Samples: {num_val} | Dtype: {train_x.dtype} | Drop Last: {drop_last}\n"
        f"  Save Dir: {args.save_dir}\n",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        epoch_rng = np.random.default_rng(args.seed + epoch)
        train_indices = epoch_rng.permutation(num_train)

        train_loss_accum = 0.0

        pbar = ProgressBar(
            total=steps_per_epoch,
            prefix=f"Epoch {epoch:02d}/{args.epochs:02d}",
            log_every=args.log_every,
            log_file=log_path,
            process_tag=f"Baseline_{args.model}",
        )

        for step in range(steps_per_epoch):
            batch_idx = train_indices[step * args.batch_size : (step + 1) * args.batch_size]
            batch_x = jnp.asarray(train_x[batch_idx])
            step_key = jax.random.fold_in(train_key, global_step)

            if is_mae:
                assert isinstance(model, Generative3DMAE)
                loss, metrics = train_step_mae(
                    model=model,
                    optimizer=optimizer,
                    x=batch_x,
                    rng_key=step_key,
                )
            elif is_u32:
                assert isinstance(model, Uniform32JEPA)
                ema_m = cosine_ema_momentum(global_step, total_steps, start=0.996, end=1.0)
                loss, metrics = train_step_uniform32(
                    model=model,
                    optimizer=optimizer,
                    x=batch_x,
                    rng_key=step_key,
                    ema_momentum=ema_m,
                    alpha=args.alpha,
                    beta=args.beta,
                )
            else:
                assert isinstance(model, SingleLODJEPA)
                ema_m = cosine_ema_momentum(global_step, total_steps, start=0.996, end=1.0)
                loss, metrics = train_step_single_lod(
                    model=model,
                    optimizer=optimizer,
                    x=batch_x,
                    rng_key=step_key,
                    ema_momentum=ema_m,
                    alpha=args.alpha,
                    beta=args.beta,
                )

            step_lr = float(schedule(global_step))
            pbar_dict = {"Loss": loss, "LR": step_lr}
            if not is_mae:
                pbar_dict["Pred"] = metrics.get("pred_loss", 0.0)
                pbar_dict["Var"] = metrics.get("var_loss", 0.0)
                pbar_dict["Cov"] = metrics.get("cov_loss", 0.0)
                pbar_dict["EMA"] = ema_m
            else:
                pbar_dict["MSE"] = metrics.get("mse_loss", loss)
            pbar.update(step + 1, pbar_dict)

            train_loss_accum += float(loss)
            global_step += 1

        avg_train_loss = train_loss_accum / steps_per_epoch

        # Validation Loop
        if drop_last and num_val >= args.batch_size:
            val_steps = num_val // args.batch_size
        else:
            val_steps = max(1, math.ceil(num_val / args.batch_size))
        val_loss_accum = 0.0

        val_pbar = ProgressBar(
            total=val_steps,
            prefix=f"Val {epoch:02d}/{args.epochs:02d}",
            log_every=args.log_every,
            log_file=log_path,
            process_tag=f"Baseline_{args.model}",
        )

        for v_step in range(val_steps):
            v_idx = np.arange(v_step * args.batch_size, min((v_step + 1) * args.batch_size, num_val))
            v_batch = jnp.asarray(val_x[v_idx])
            v_key = jax.random.fold_in(train_key, 200_000 + epoch * 100 + v_step)

            if is_mae:
                assert isinstance(model, Generative3DMAE)
                v_loss, v_metrics = val_step_mae(model, v_batch, v_key)
                val_pbar.update(v_step + 1, {"ValLoss": v_loss, "MSE": v_metrics.get("mse_loss", v_loss)})
            elif is_u32:
                assert isinstance(model, Uniform32JEPA)
                v_loss, v_metrics = val_step_uniform32(model, v_batch, v_key, alpha=args.alpha, beta=args.beta)
                val_pbar.update(v_step + 1, {"ValLoss": v_loss, "EmbStd": v_metrics.get("emb_std", 0.0)})
            else:
                assert isinstance(model, SingleLODJEPA)
                v_loss, v_metrics = val_step_single_lod(model, v_batch, v_key, alpha=args.alpha, beta=args.beta)
                val_pbar.update(v_step + 1, {"ValLoss": v_loss, "EmbStd": v_metrics.get("emb_std", 0.0)})

            val_loss_accum += float(v_loss)

        avg_val_loss = val_loss_accum / val_steps

        # Compute uncentered RankMe every 5 epochs and on final epoch
        rankme_val: Optional[float] = None
        if (epoch % 5 == 0) or (epoch == args.epochs):
            val_pooled_list = []
            for v_step in range(val_steps):
                v_idx = np.arange(v_step * args.batch_size, min((v_step + 1) * args.batch_size, num_val))
                v_batch = jnp.asarray(val_x[v_idx])
                if is_mae:
                    assert isinstance(model, Generative3DMAE)
                    pooled = extract_mae_embeddings(model, v_batch)
                elif is_u32:
                    assert isinstance(model, Uniform32JEPA)
                    pooled = extract_uniform32_embeddings(model, v_batch)
                else:
                    assert isinstance(model, SingleLODJEPA)
                    pooled = extract_single_lod_embeddings(model, v_batch)
                val_pooled_list.append(np.asarray(pooled))
            all_val_pooled = np.concatenate(val_pooled_list, axis=0)
            rankme_val = float(compute_rankme(all_val_pooled))

        # Checkpoints
        save_checkpoint(
            last_ckpt_path,
            model,
            metadata={
                "model": args.model,
                "epoch": epoch,
                "global_step": global_step,
                "val_loss": avg_val_loss,
                "rankme": rankme_val,
            },
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(
                best_ckpt_path,
                model,
                metadata={
                    "model": args.model,
                    "epoch": epoch,
                    "global_step": global_step,
                    "val_loss": best_val_loss,
                    "rankme": rankme_val,
                },
            )

        # Logging
        record = {
            "model": args.model,
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "lr": float(schedule(global_step)),
            "epoch_sec": time.time() - epoch_start_time,
        }
        if rankme_val is not None:
            record["rankme"] = rankme_val
        if not is_mae:
            record["ema_momentum"] = ema_m

        append_metrics_jsonl(metrics_path, record)

        rankme_str = f"{rankme_val:6.2f}" if rankme_val is not None else "   N/A"
        now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        epoch_line = (
            f"[{now_ts}] [Baseline_{args.model} PID {os.getpid()}] "
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Step {global_step:04d} | "
            f"Train Loss: {avg_train_loss:6.4f} | "
            f"Val Loss: {avg_val_loss:6.4f} | "
            f"RankMe: {rankme_str}"
        )
        print(epoch_line)
        if log_path:
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(epoch_line + "\n")
            except Exception:
                pass

    print(f"\nBaseline [{args.model}] Training Completed! Best Val Loss: {best_val_loss:6.4f}", flush=True)
    return model, best_val_loss


def main():
    """CLI entrypoint."""
    args = parse_args()
    train_baseline(args)


if __name__ == "__main__":
    main()

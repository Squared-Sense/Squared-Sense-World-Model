"""Stage 1A Spatial JEPA Training Pipeline.

Trains MultiLODBackbone context encoder and JEPAPredictor via 3D spatial masking
and stop-gradient EMA targets with VICReg regularization.

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

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from data.dataset import concat_episode_shells, load_episode_splits, load_preallocated_dataset
from models.transformer import SpatialJEPA, sample_3d_jepa_masks
from train.checkpoint import load_checkpoint, save_checkpoint
from train.diagnostics import ProgressBar, append_metrics_jsonl, compute_rankme
from train.losses import cosine_ema_momentum, spatial_jepa_loss


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments for Stage 1A Spatial JEPA training."""
    parser = argparse.ArgumentParser(
        description="Stage 1A Spatial JEPA Training for Squared Sense World Model"
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
        default="checkpoints/m2_spatial_jepa",
        help="Directory to save checkpoints and metrics.jsonl.",
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
        help="Weight for VICReg variance regularizer (default: 0.1).",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.04,
        help="Weight for VICReg covariance regularizer (default: 0.04).",
    )
    parser.add_argument(
        "--no_vicreg",
        action="store_true",
        help="Disable VICReg variance/covariance regularization (sets alpha=0.0, beta=0.0) for collapse control.",
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
        "--micro_batch_size",
        type=int,
        default=None,
        help="Micro-batch size for gradient accumulation to fit within low VRAM devices (default: None).",
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
    if parsed.no_vicreg:
        parsed.alpha = 0.0
        parsed.beta = 0.0
    if parsed.smoke_test:
        parsed.epochs = 2
        parsed.batch_size = 4
        parsed.warmup_epochs = 1
        if parsed.log_every == 5:
            parsed.log_every = 2

    return parsed


@nnx.jit
def micro_train_step(
    model: SpatialJEPA,
    x: jax.Array,
    rng_key: jax.Array,
    alpha: float,
    beta: float,
) -> Tuple[Tuple[jax.Array, Dict[str, jax.Array]], Any]:
    """Single JIT-compiled micro-batch forward and backward pass."""
    if x.dtype == jnp.uint8:
        x = x.astype(jnp.float32) / 255.0
    b = x.shape[0]
    vis_idx, tgt_idx = sample_3d_jepa_masks(rng_key, b, num_shells=model.num_shells, cross_scale_prob=0.3)

    def loss_fn(m: SpatialJEPA):
        pred_targets, norm_ema, context_tokens = m(x, vis_idx, tgt_idx)
        loss, metrics = spatial_jepa_loss(pred_targets, norm_ema, context_tokens, alpha=alpha, beta=beta)
        return loss, metrics

    (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    return (loss, metrics), grads


@nnx.jit
def apply_accumulated_grads(
    model: SpatialJEPA,
    optimizer: nnx.Optimizer,
    grads: Any,
    ema_momentum: float,
) -> None:
    """Apply accumulated gradients to parameters and update EMA target encoder."""
    optimizer.update(
        (model.context_encoder, model.predictor),
        (grads.context_encoder, grads.predictor),
    )
    model.update_ema(ema_momentum)


def train_step(
    model: SpatialJEPA,
    optimizer: nnx.Optimizer,
    x: jax.Array,
    rng_key: jax.Array,
    ema_momentum: float,
    alpha: float,
    beta: float,
    micro_batch_size: Optional[int] = None,
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """Single optimization step for Stage 1A Spatial JEPA with optional gradient accumulation.

    Performs on-device uint8 -> float32 normalization. If micro_batch_size is
    specified and smaller than batch size, evaluates micro-batches sequentially to
    minimize peak activation memory inside small VRAM (e.g., 4 GB).

    Args:
        model: SpatialJEPA module.
        optimizer: nnx.Optimizer managing context_encoder and predictor parameters.
        x: (B, 3, 16, 16, 16, 16) voxel input batch (uint8 or float32).
        rng_key: JAX PRNG key for 3D mask sampling.
        ema_momentum: Scalar momentum for updating target encoder.
        alpha: VICReg variance loss weight.
        beta: VICReg covariance loss weight.
        micro_batch_size: Optional micro-batch size for gradient accumulation.

    Returns:
        (loss, metrics_dict) tuple of JAX scalars.
    """
    b = x.shape[0]
    if micro_batch_size is None or micro_batch_size >= b or (b % micro_batch_size != 0):
        (loss, metrics), grads = micro_train_step(model, x, rng_key, alpha=alpha, beta=beta)
        apply_accumulated_grads(model, optimizer, grads, ema_momentum)
        return loss, metrics

    num_micro = b // micro_batch_size
    keys = jax.random.split(rng_key, num_micro)

    total_loss = 0.0
    accum_metrics = {}
    accum_grads = None

    for i in range(num_micro):
        x_micro = x[i * micro_batch_size : (i + 1) * micro_batch_size]
        (loss_i, metrics_i), grads_i = micro_train_step(model, x_micro, keys[i], alpha=alpha, beta=beta)
        total_loss = total_loss + loss_i / num_micro

        for k, v in metrics_i.items():
            accum_metrics[k] = accum_metrics.get(k, 0.0) + v / num_micro

        if accum_grads is None:
            accum_grads = jax.tree.map(lambda g: g / num_micro, grads_i)
        else:
            accum_grads = jax.tree.map(lambda acc, g: acc + g / num_micro, accum_grads, grads_i)

    apply_accumulated_grads(model, optimizer, accum_grads, ema_momentum)
    return total_loss, accum_metrics


@nnx.jit
def val_step(
    model: SpatialJEPA,
    x: jax.Array,
    rng_key: jax.Array,
    alpha: float,
    beta: float,
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """Single JIT-compiled evaluation step for validation loss and metrics.

    Args:
        model: SpatialJEPA module.
        x: (B, 3, 16, 16, 16, 16) voxel input batch.
        rng_key: JAX PRNG key for 3D mask sampling.
        alpha: VICReg variance loss weight.
        beta: VICReg covariance loss weight.

    Returns:
        (loss, metrics_dict) tuple of JAX scalars.
    """
    if x.dtype == jnp.uint8:
        x = x.astype(jnp.float32) / 255.0
    b = x.shape[0]
    vis_idx, tgt_idx = sample_3d_jepa_masks(rng_key, b, num_shells=model.num_shells, cross_scale_prob=0.3)
    pred_targets, norm_ema, context_tokens = model(x, vis_idx, tgt_idx)
    loss, metrics = spatial_jepa_loss(pred_targets, norm_ema, context_tokens, alpha=alpha, beta=beta)

    # Measure embedding standard deviation across batch/token dimensions
    emb_std = jnp.mean(jnp.std(context_tokens, axis=0))
    metrics["emb_std"] = emb_std
    return loss, metrics


@nnx.jit
def extract_unmasked_embeddings(model: SpatialJEPA, x: jax.Array) -> jax.Array:
    """Extract unmasked context representations pooled across spatial tokens.

    Args:
        model: SpatialJEPA module.
        x: (B, 3, 16, 16, 16, 16) voxel input batch.

    Returns:
        (B, 256) mean-pooled embedding vectors.
    """
    if x.dtype == jnp.uint8:
        x = x.astype(jnp.float32) / 255.0
    tokens = model.context_encoder(x, visible_idx=None)  # (B, 192, 256)
    return jnp.mean(tokens, axis=1)  # (B, 256)


def load_dataset(
    data_dir: Optional[str],
    smoke_test: bool,
    seed: int,
    as_uint8: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load or synthetically generate train and validation voxel data.

    Returns:
        Tuple of (train_data, val_data) arrays of shape (N, 3, 16, 16, 16, 16).
    """
    if smoke_test or data_dir is None:
        rng = np.random.default_rng(seed)
        if as_uint8:
            train_x = rng.integers(0, 256, size=(16, 3, 16, 16, 16, 16), dtype=np.uint8)
            val_x = rng.integers(0, 256, size=(4, 3, 16, 16, 16, 16), dtype=np.uint8)
        else:
            train_x = rng.uniform(0.0, 1.0, size=(16, 3, 16, 16, 16, 16)).astype(np.float32)
            val_x = rng.uniform(0.0, 1.0, size=(4, 3, 16, 16, 16, 16)).astype(np.float32)
        return train_x, val_x

    return load_preallocated_dataset(data_dir, seed=seed, as_uint8=as_uint8)


def train_spatial_jepa(args: argparse.Namespace) -> Tuple[SpatialJEPA, float]:
    """Run full training loop for Stage 1A Spatial JEPA."""
    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(args.save_dir, "train.log")
    metrics_path = os.path.join(args.save_dir, "metrics.jsonl")
    best_ckpt_path = os.path.join(args.save_dir, "best.pkl")
    last_ckpt_path = os.path.join(args.save_dir, "last.pkl")

    backend = jax.default_backend()
    devices = jax.devices()
    args.batch_size, args.micro_batch_size = auto_configure_batching(
        batch_size=args.batch_size,
        micro_batch_size=getattr(args, "micro_batch_size", None),
        backend=backend,
    )

    # 1. Initialize PRNG and Model
    root_key = jax.random.key(args.seed)
    model_key, train_key = jax.random.split(root_key)
    model = SpatialJEPA(num_shells=3, rngs=nnx.Rngs(int(jax.random.bits(model_key))))

    # 2. Load Data
    effective_data_dir = args.data_dir or args.region_dir
    as_uint8 = getattr(args, "as_uint8", True)
    train_x, val_x = load_dataset(effective_data_dir, args.smoke_test, args.seed, as_uint8=as_uint8)
    num_train = len(train_x)
    num_val = len(val_x)
    drop_last = getattr(args, "drop_last", True)
    if drop_last and num_train >= args.batch_size:
        steps_per_epoch = max(1, num_train // args.batch_size)
    else:
        steps_per_epoch = max(1, math.ceil(num_train / args.batch_size))
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = min(args.warmup_epochs * steps_per_epoch, max(0, total_steps - 1))
    decay_steps = max(total_steps, warmup_steps + 1)

    # 3. Configure Optimizer and Schedule
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.lr,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        end_value=args.lr * 0.01,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=args.weight_decay),
    )
    # Target ONLY context_encoder and predictor
    optimizer = nnx.Optimizer((model.context_encoder, model.predictor), tx, wrt=nnx.Param)

    best_val_loss = float("inf")
    global_step = 0

    micro_str = f" | Micro-Batch: {args.micro_batch_size}" if getattr(args, "micro_batch_size", None) else ""
    print(
        f"Starting Stage 1A Spatial JEPA Training:\n"
        f"  Epochs: {args.epochs} | Steps/Epoch: {steps_per_epoch} | Total Steps: {total_steps}\n"
        f"  Batch Size: {args.batch_size}{micro_str} | Peak LR: {args.lr} | Warmup Epochs: {args.warmup_epochs}\n"
        f"  Train Samples: {num_train} | Val Samples: {num_val} | Dtype: {train_x.dtype} | Drop Last: {drop_last}\n"
        f"  Save Dir: {args.save_dir}\n"
    )

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()

        # Shuffle training indices per epoch
        epoch_rng = np.random.default_rng(args.seed + epoch)
        train_indices = epoch_rng.permutation(num_train)

        train_loss_accum = 0.0
        train_pred_loss_accum = 0.0
        train_var_loss_accum = 0.0
        train_cov_loss_accum = 0.0

        pbar = ProgressBar(
            total=steps_per_epoch,
            prefix=f"Epoch {epoch:02d}/{args.epochs:02d}",
            log_every=args.log_every,
            log_file=log_path,
            process_tag="SpatialJEPA",
        )

        for step in range(steps_per_epoch):
            batch_idx = train_indices[step * args.batch_size : (step + 1) * args.batch_size]
            batch_x = jnp.asarray(train_x[batch_idx])

            ema_m = cosine_ema_momentum(global_step, total_steps, start=0.996, end=1.0)
            step_key = jax.random.fold_in(train_key, global_step)

            loss, metrics = train_step(
                model=model,
                optimizer=optimizer,
                x=batch_x,
                rng_key=step_key,
                ema_momentum=ema_m,
                alpha=args.alpha,
                beta=args.beta,
                micro_batch_size=getattr(args, "micro_batch_size", None),
            )

            step_lr = float(schedule(global_step))
            pbar.update(
                step + 1,
                {
                    "Loss": loss,
                    "Pred": metrics["pred_loss"],
                    "Var": metrics["var_loss"],
                    "Cov": metrics["cov_loss"],
                    "EMA": ema_m,
                    "LR": step_lr,
                },
            )

            train_loss_accum = train_loss_accum + loss
            train_pred_loss_accum = train_pred_loss_accum + metrics["pred_loss"]
            train_var_loss_accum = train_var_loss_accum + metrics["var_loss"]
            train_cov_loss_accum = train_cov_loss_accum + metrics["cov_loss"]
            global_step += 1

        avg_train_loss = float(train_loss_accum) / steps_per_epoch
        avg_train_pred_loss = float(train_pred_loss_accum) / steps_per_epoch

        # Validation Loop
        if drop_last and num_val >= args.batch_size:
            val_steps = num_val // args.batch_size
        else:
            val_steps = max(1, math.ceil(num_val / args.batch_size))
        val_loss_accum = 0.0
        val_emb_std_accum = 0.0

        val_pbar = ProgressBar(
            total=val_steps,
            prefix=f"Val {epoch:02d}/{args.epochs:02d}",
            log_every=args.log_every,
            log_file=log_path,
            process_tag="SpatialJEPA",
        )

        for v_step in range(val_steps):
            v_idx = np.arange(v_step * args.batch_size, min((v_step + 1) * args.batch_size, num_val))
            v_batch = jnp.asarray(val_x[v_idx])
            v_key = jax.random.fold_in(train_key, 100_000 + epoch * 100 + v_step)

            v_loss, v_metrics = val_step(
                model=model,
                x=v_batch,
                rng_key=v_key,
                alpha=args.alpha,
                beta=args.beta,
            )
            val_pbar.update(
                v_step + 1,
                {
                    "ValLoss": v_loss,
                    "EmbStd": v_metrics["emb_std"],
                },
            )
            val_loss_accum = val_loss_accum + v_loss
            val_emb_std_accum = val_emb_std_accum + v_metrics["emb_std"]

        avg_val_loss = float(val_loss_accum) / val_steps
        avg_emb_std = float(val_emb_std_accum) / val_steps

        # Compute uncentered RankMe every 5 epochs and on the final epoch
        rankme_val: Optional[float] = None
        if (epoch % 5 == 0) or (epoch == args.epochs):
            val_pooled_list = []
            for v_step in range(val_steps):
                v_idx = np.arange(v_step * args.batch_size, min((v_step + 1) * args.batch_size, num_val))
                v_batch = jnp.asarray(val_x[v_idx])
                pooled = extract_unmasked_embeddings(model, v_batch)
                val_pooled_list.append(np.asarray(pooled))
            all_val_pooled = np.concatenate(val_pooled_list, axis=0)
            rankme_val = float(compute_rankme(all_val_pooled))

        # Checkpoint: save last.pkl
        save_checkpoint(
            last_ckpt_path,
            model,
            metadata={
                "epoch": epoch,
                "global_step": global_step,
                "val_loss": avg_val_loss,
                "rankme": rankme_val,
            },
        )

        # Checkpoint: save best.pkl if lowest val loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(
                best_ckpt_path,
                model,
                metadata={
                    "epoch": epoch,
                    "global_step": global_step,
                    "val_loss": best_val_loss,
                    "rankme": rankme_val,
                },
            )

        # Metrics logging
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": avg_train_loss,
            "train_pred_loss": avg_train_pred_loss,
            "val_loss": avg_val_loss,
            "emb_std": avg_emb_std,
            "ema_momentum": ema_m,
            "lr": float(schedule(global_step)),
            "epoch_sec": time.time() - epoch_start_time,
        }
        if rankme_val is not None:
            record["rankme"] = rankme_val

        append_metrics_jsonl(metrics_path, record)

        # Compact console progress line
        rankme_str = f"{rankme_val:6.2f}" if rankme_val is not None else "   N/A"
        now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        epoch_line = (
            f"[{now_ts}] [SpatialJEPA PID {os.getpid()}] "
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Step {global_step:04d} | "
            f"Train Loss: {avg_train_loss:6.4f} | "
            f"Val Loss: {avg_val_loss:6.4f} | "
            f"Emb Std: {avg_emb_std:5.3f} | "
            f"RankMe: {rankme_str} | "
            f"EMA: {ema_m:6.4f}"
        )
        print(epoch_line)
        if log_path:
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(epoch_line + "\n")
            except Exception:
                pass

    print(f"\nTraining completed! Best Val Loss: {best_val_loss:6.4f}")
    return model, best_val_loss


def main():
    """CLI entrypoint."""
    args = parse_args()
    train_spatial_jepa(args)


if __name__ == "__main__":
    main()

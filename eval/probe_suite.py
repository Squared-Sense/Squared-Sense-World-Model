"""Frozen representation and dynamics probe suite for Squared Sense World Model.

Implements lightweight CPU-friendly scikit-learn probes across 3 random seeds:
  1. Distant Goal Direction & Distance regression (Ridge, R^2 & Cosine Sim)
  2. Local Hazard Detection classification (LogisticRegression, F1 & AUROC)
  3. Kinematic Velocity & Occlusion Memory (Ridge & LogisticRegression)
  4. Representation Health & Multi-Step Rollout (RankMe & k-step Cosine Sim / L1)

Strictly follows docs/EVAL_CONTRACT.md and docs/MODEL_CONTRACT.md.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from train.gpu_env import setup_gpu_environment

# Ensure non-greedy allocation and VRAM limits are set prior to XLA initialization
setup_gpu_environment()

# Prevent joblib/sklearn multiprocessing from calling os.fork() after JAX initialization
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import f1_score, r2_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data.dataset import (
    compute_local_hazard_labels,
    compute_occlusion_labels,
    extract_temporal_clips,
    load_episode_splits,
)
from data.schema import normalize_shells
from models.baselines import Generative3DMAE, SingleLODJEPA, Uniform32JEPA
from models.temporal_lstm import DualLSTMTemporalIntegrator
from models.transformer import MultiLODBackbone, SpatialJEPA
from train.checkpoint import load_checkpoint
from train.diagnostics import compute_rankme


# ---------------------------------------------------------------------------
# Frozen Feature Extraction
# ---------------------------------------------------------------------------

@nnx.jit
def _forward_tokens(encoder: Any, batch_x: jax.Array) -> jax.Array:
    """Run frozen forward pass through encoder to extract spatial tokens."""
    return jax.lax.stop_gradient(encoder(batch_x, visible_idx=None))


def _rollout_temporal(
    temporal_model: DualLSTMTemporalIntegrator,
    tokens_seq: jax.Array,
    state_seq: jax.Array,
    action_seq: jax.Array,
    k_steps: int,
) -> jax.Array:
    """Run frozen temporal rollout without gradients."""
    pred_future, _ = temporal_model.rollout(
        tokens_seq, state_seq, action_seq, k_steps=k_steps
    )
    return jax.lax.stop_gradient(pred_future)


@nnx.jit
def _step_single(
    temporal_model: DualLSTMTemporalIntegrator,
    tok_t: jax.Array,
    st_t: jax.Array,
    act_t: jax.Array,
    hidden: Any,
) -> Tuple[jax.Array, Any]:
    """Single JIT-compiled step on GPU (compiles in 0.1s without loop unrolling)."""
    _, c_t, next_hidden = temporal_model.step(tok_t, st_t, act_t, hidden)
    return jax.lax.stop_gradient(c_t), next_hidden


def _step_context_states(
    temporal_model: DualLSTMTemporalIntegrator,
    tokens_seq: jax.Array,
    state_seq: jax.Array,
    action_seq: jax.Array,
) -> jax.Array:
    """Extract fused context representations c_t across trajectory steps without loop unrolling."""
    b, T = tokens_seq.shape[:2]
    hidden = temporal_model.init_hidden(b)
    c_list = []
    for t in range(T):
        c_t, hidden = _step_single(
            temporal_model, tokens_seq[:, t], state_seq[:, t], action_seq[:, t], hidden
        )
        c_list.append(c_t)
    return jnp.stack(c_list, axis=1)


def extract_features(
    model: Any,
    model_type: str,
    shells: np.ndarray | jax.Array,
    state: Optional[np.ndarray | jax.Array] = None,
    action: Optional[np.ndarray | jax.Array] = None,
    temporal_model: Optional[DualLSTMTemporalIntegrator] = None,
    batch_size: int = 32,
    episode_lengths: Optional[List[int]] = None,
) -> np.ndarray:
    """Extract frozen inference embeddings for a given model architecture.

    Supported model types:
      - "m2_static": mean-pool context encoder tokens (B, 192, 256) -> z (B, 256)
      - "b1_lod0"  : mean-pool LOD-0 single-shell tokens (B, 64, 256) -> z (B, 256)
      - "b2_lod2"  : mean-pool LOD-2 single-shell tokens (B, 64, 256) -> z (B, 256)
      - "b3_mae"   : mean-pool Generative 3D-MAE encoder tokens -> z (B, 256)
      - "m1_full"  : concatenate mean-pooled z_t (B, 256) and Dual-LSTM context c_t (B, 192) -> feat (B, 448)

    Args:
        model: Trained or initialized Flax NNX module instance.
        model_type: Identifier in {"m1_full", "m2_static", "b1_lod0", "b2_lod2", "b3_mae"}.
        shells: Input voxel shells array of shape (N, 3, 16, 16, 16, 16).
        state: Optional proprioceptive states array of shape (N, 12).
        action: Optional discrete actions array of shape (N,).
        temporal_model: Optional DualLSTMTemporalIntegrator (required for m1_full).
        batch_size: Batch size for chunked evaluation.

    Returns:
        float32 NumPy feature array:
          - (N, 256) for static models (m2_static, b1_lod0, b2_lod2, b3_mae)
          - (N, 448) for m1_full concatenating z_t and c_t
    """
    valid_types = {"m1_full", "m2_static", "b1_lod0", "b2_lod2", "b3_mae", "b4_uniform32"}
    if model_type not in valid_types:
        raise ValueError(f"Unknown model_type '{model_type}'. Expected one of {valid_types}")

    x = np.asarray(shells)
    if x.dtype == np.uint8:
        x = normalize_shells(x)
    x = jnp.asarray(x, dtype=jnp.float32)

    n_samples = x.shape[0]

    # Resolve encoder
    if model_type in ("m2_static", "m1_full"):
        encoder = model.context_encoder if hasattr(model, "context_encoder") else model
    elif model_type in ("b1_lod0", "b2_lod2", "b4_uniform32"):
        if hasattr(model, "context_encoder"):
            encoder = model.context_encoder
        elif hasattr(model, "jepa") and hasattr(model.jepa, "context_encoder"):
            encoder = model.jepa.context_encoder
        else:
            encoder = model
    elif model_type == "b3_mae":
        encoder = model.encoder if hasattr(model, "encoder") else model
    else:
        encoder = model

    # Process in batches
    z_list = []
    tokens_list = []
    for i in range(0, n_samples, batch_size):
        bx = x[i : i + batch_size]
        if model_type == "b1_lod0":
            bx = bx[:, 0:1]
        elif model_type == "b2_lod2":
            bx = bx[:, 2:3]
        elif model_type == "b4_uniform32" and bx.ndim == 5:
            bx = jnp.expand_dims(bx, axis=1)

        tokens = _forward_tokens(encoder, bx)
        z_batch = jnp.mean(tokens, axis=1)  # (B, 256)
        z_list.append(np.asarray(z_batch))
        if model_type == "m1_full":
            tokens_list.append(tokens)

    z_all = np.concatenate(z_list, axis=0)  # (N, 256)

    if model_type != "m1_full":
        return z_all.astype(np.float32)

    # For m1_full, extract and concatenate fused temporal context c_t (N, 192)
    if temporal_model is None:
        raise ValueError("temporal_model must be provided when model_type='m1_full'")

    all_tokens = jnp.concatenate(tokens_list, axis=0)  # (N, 192, 256)
    st = jnp.zeros((n_samples, 12), dtype=jnp.float32) if state is None else jnp.asarray(state, dtype=jnp.float32)
    act = jnp.zeros((n_samples,), dtype=jnp.int32) if action is None else jnp.asarray(action, dtype=jnp.int32)

    if episode_lengths is not None and sum(episode_lengths) == n_samples:
        c_chunks = []
        offset = 0
        for ep_len in episode_lengths:
            tok_ep = all_tokens[offset : offset + ep_len][None, ...]
            st_ep = st[offset : offset + ep_len][None, ...]
            act_ep = act[offset : offset + ep_len][None, ...]
            c_ep = _step_context_states(temporal_model, tok_ep, st_ep, act_ep)
            c_chunks.append(np.asarray(c_ep[0], dtype=np.float32))
            offset += ep_len
        c_all = np.concatenate(c_chunks, axis=0)
    else:
        # Treat as sequence: (1, N, ...)
        tok_seq = all_tokens[None, ...]
        st_seq = st[None, ...]
        act_seq = act[None, ...]

        c_seq = _step_context_states(temporal_model, tok_seq, st_seq, act_seq)  # (1, N, 192)
        c_all = np.asarray(c_seq[0], dtype=np.float32)  # (N, 192)

    feat = np.concatenate([z_all, c_all], axis=-1)  # (N, 448)
    return feat.astype(np.float32)


# ---------------------------------------------------------------------------
# Scikit-Learn Probes
# ---------------------------------------------------------------------------

def probe_distant_goal(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seeds: Tuple[int, ...] = (0, 1, 2),
) -> Dict[str, float]:
    """Probe distant goal direction and distance using Ridge regression.

    Target y represents relative 3D goal vector (goal_dx, goal_dy, goal_dz).
    Evaluated using Ridge(alpha=1.0) across specified random seeds.

    Args:
        X_train: (N_train, D) training representations.
        y_train: (N_train, 3) or (N_train, K) ground-truth goal vectors.
        X_test: (N_test, D) test representations.
        y_test: (N_test, 3) or (N_test, K) ground-truth goal vectors.
        seeds: Random seeds to evaluate over.

    Returns:
        Dictionary containing r2, r2_std, mean_cosine_sim, and mean_cosine_sim_std.
    """
    r2_list = []
    cos_list = []

    for seed in seeds:
        reg = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 7)))
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        r2_list.append(float(r2))

        # Cosine similarity per sample
        dot = np.sum(y_pred * y_test, axis=-1)
        norm_pred = np.linalg.norm(y_pred, axis=-1)
        norm_test = np.linalg.norm(y_test, axis=-1)
        cos_sim = dot / (norm_pred * norm_test + 1e-7)
        cos_list.append(float(np.mean(cos_sim)))

    return {
        "r2": float(np.mean(r2_list)),
        "r2_std": float(np.std(r2_list)),
        "mean_cosine_sim": float(np.mean(cos_list)),
        "mean_cosine_sim_std": float(np.std(cos_list)),
    }


def probe_local_hazard(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seeds: Tuple[int, ...] = (0, 1, 2),
) -> Dict[str, float]:
    """Probe 1-block local hazard detection using LogisticRegression.

    Target y represents binary classification (1=hazard ahead, 0=clear).
    Gracefully handles single-class edge cases in tiny smoke tests.

    Args:
        X_train: (N_train, D) training representations.
        y_train: (N_train,) binary labels.
        X_test: (N_test, D) test representations.
        y_test: (N_test,) binary labels.
        seeds: Random seeds to evaluate over.

    Returns:
        Dictionary containing f1, f1_std, auroc, and auroc_std.
    """
    y_train = np.asarray(y_train, dtype=int).ravel()
    y_test = np.asarray(y_test, dtype=int).ravel()

    f1_list = []
    auroc_list = []

    train_classes = np.unique(y_train)
    test_classes = np.unique(y_test)

    # Edge case: single class in training set
    if len(train_classes) < 2:
        const_class = int(train_classes[0])
        pred = np.full_like(y_test, const_class)
        f1 = float(f1_score(y_test, pred, zero_division=0.0))
        auroc = 1.0 if np.all(y_test == const_class) else 0.5
        return {
            "f1": f1,
            "f1_std": 0.0,
            "auroc": auroc,
            "auroc_std": 0.0,
        }

    for seed in seeds:
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=seed
            ),
        )
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        f1 = float(f1_score(y_test, pred, zero_division=0.0))
        f1_list.append(f1)

        if len(test_classes) < 2:
            auroc = 1.0 if np.all(pred == y_test) else 0.5
        else:
            prob = clf.predict_proba(X_test)[:, 1]
            auroc = float(roc_auc_score(y_test, prob))
        auroc_list.append(auroc)

    return {
        "f1": float(np.mean(f1_list)),
        "f1_std": float(np.std(f1_list)),
        "auroc": float(np.mean(auroc_list)),
        "auroc_std": float(np.std(auroc_list)),
    }


def probe_kinematic_and_occlusion(
    X_train: np.ndarray,
    v_train: np.ndarray,
    occ_train: np.ndarray,
    X_test: np.ndarray,
    v_test: np.ndarray,
    occ_test: np.ndarray,
    seeds: Tuple[int, ...] = (0, 1, 2),
) -> Dict[str, float]:
    """Probe kinematic velocity (vx, vy, vz) and occlusion memory.

    Args:
        X_train: (N_train, D) training representations.
        v_train: (N_train, 3) ground-truth velocities.
        occ_train: (N_train,) binary occlusion memory labels.
        X_test: (N_test, D) test representations.
        v_test: (N_test, 3) ground-truth velocities.
        occ_test: (N_test,) binary occlusion memory labels.
        seeds: Random seeds to evaluate over.

    Returns:
        Dictionary containing velocity_r2, velocity_r2_std, occlusion_f1, and occlusion_f1_std.
    """
    v_r2_list = []
    occ_f1_list = []

    occ_train = np.asarray(occ_train, dtype=int).ravel()
    occ_test = np.asarray(occ_test, dtype=int).ravel()
    occ_train_classes = np.unique(occ_train)

    for seed in seeds:
        # 1. Velocity regression
        reg = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 7)))
        reg.fit(X_train, v_train)
        v_pred = reg.predict(X_test)
        v_r2 = float(r2_score(v_test, v_pred))
        v_r2_list.append(v_r2)

        # 2. Occlusion classification
        if len(occ_train_classes) < 2:
            pred_occ = np.full_like(occ_test, occ_train_classes[0])
            occ_f1 = float(f1_score(occ_test, pred_occ, zero_division=0.0))
        else:
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=1000, class_weight="balanced", random_state=seed
                ),
            )
            clf.fit(X_train, occ_train)
            pred_occ = clf.predict(X_test)
            occ_f1 = float(f1_score(occ_test, pred_occ, zero_division=0.0))
        occ_f1_list.append(occ_f1)

    return {
        "velocity_r2": float(np.mean(v_r2_list)),
        "velocity_r2_std": float(np.std(v_r2_list)),
        "occlusion_f1": float(np.mean(occ_f1_list)),
        "occlusion_f1_std": float(np.std(occ_f1_list)),
    }


def eval_multistep_rollout(
    temporal_model: Optional[DualLSTMTemporalIntegrator],
    tokens_seq: np.ndarray | jax.Array,
    state_seq: np.ndarray | jax.Array,
    action_seq: np.ndarray | jax.Array,
    ema_targets_seq: np.ndarray | jax.Array,
    horizons: Tuple[int, ...] = (1, 2, 4, 8),
) -> Dict[str, float]:
    """Evaluate multi-step autoregressive rollout and uncentered RankMe.

    Args:
        temporal_model: DualLSTMTemporalIntegrator instance (or None for persistence baseline).
        tokens_seq: (B, T, 192, 256) or (B, T, 256) sequence of voxel tokens.
        state_seq: (B, T, 12) sequence of proprioceptive states.
        action_seq: (B, T) sequence of actions.
        ema_targets_seq: (B, T, 256) ground-truth target representations.
        horizons: Rollout step horizons to evaluate (e.g. 1, 2, 4, 8).

    Returns:
        Dictionary containing cosine_sim_h{k}, l1_error_h{k} for each horizon, plus rankme.
    """
    tok = jnp.asarray(tokens_seq, dtype=jnp.float32)
    st = jnp.asarray(state_seq, dtype=jnp.float32)
    act = jnp.asarray(action_seq, dtype=jnp.int32)
    targets = np.asarray(ema_targets_seq, dtype=np.float32)

    b, T = tok.shape[:2]
    k_max = max(horizons)
    valid_len = T - k_max

    if temporal_model is not None:
        pred_future = _rollout_temporal(temporal_model, tok, st, act, k_steps=k_max)
        pred_future_np = np.asarray(pred_future)[:, :valid_len]  # (B, valid_len, k_max, 256)
    else:
        # Static persistence baseline: predict constant current pooled representation
        z_curr = np.mean(tok, axis=2) if tok.ndim == 4 else np.asarray(tok)
        pred_future_np = np.repeat(z_curr[:, :valid_len, None, :], k_max, axis=2)

    results: Dict[str, float] = {}

    for h in horizons:
        pred_h = pred_future_np[:, :, h - 1, :]  # (B, valid_len, 256)
        tgt_h = targets[:, h : h + valid_len, :] # (B, valid_len, 256)

        # Cosine similarity
        dot = np.sum(pred_h * tgt_h, axis=-1)
        norm_p = np.linalg.norm(pred_h, axis=-1)
        norm_t = np.linalg.norm(tgt_h, axis=-1)
        cos = dot / (norm_p * norm_t + 1e-7)
        results[f"cosine_sim_h{h}"] = float(np.mean(cos))

        # L1 error
        l1 = np.abs(pred_h - tgt_h)
        results[f"l1_error_h{h}"] = float(np.mean(l1))

    # Uncentered RankMe on representations
    results["rankme"] = float(compute_rankme(pred_future_np))

    return results


# ---------------------------------------------------------------------------
# Synthetic Smoke Test Generator & CLI
# ---------------------------------------------------------------------------

def generate_synthetic_probe_data(
    num_episodes: int = 4,
    clip_len: int = 24,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Generate synthetic trajectory episodes for smoke testing the probe suite.

    Returns:
        Dictionary with shells, state, action, meta, hazard, occlusion, and ema_targets.
    """
    rng = np.random.default_rng(seed)
    total_steps = num_episodes * clip_len

    # Voxel shells (N, 3, 16, 16, 16, 16) float32
    shells = rng.uniform(0.0, 1.0, size=(total_steps, 3, 16, 16, 16, 16)).astype(np.float32)

    # State (N, 12): [vx, vy, vz, dx, dy, dz, sin_yaw, cos_yaw, pitch, health, food, grounded]
    state = np.zeros((total_steps, 12), dtype=np.float32)
    state[:, 0:3] = rng.normal(0.0, 0.5, size=(total_steps, 3))  # velocities
    state[:, 3:6] = state[:, 0:3] * 0.2                          # displacements
    yaws = rng.uniform(-np.pi, np.pi, size=total_steps)
    state[:, 6] = np.sin(yaws)
    state[:, 7] = np.cos(yaws)
    state[:, 8] = rng.uniform(-30.0, 30.0, size=total_steps)     # pitch
    state[:, 9] = 20.0                                            # health
    state[:, 10] = 20.0                                           # food
    state[:, 11] = 1.0                                            # grounded

    # Action (N,)
    action = rng.integers(0, 12, size=total_steps, dtype=np.int32)

    # Meta (N, 6): [world_x, world_y, world_z, goal_dx, goal_dy, goal_dz]
    meta = np.zeros((total_steps, 6), dtype=np.float32)
    meta[:, 0:3] = rng.uniform(-50.0, 50.0, size=(total_steps, 3))
    meta[:, 3:6] = rng.uniform(30.0, 48.0, size=(total_steps, 3))  # Distant goal (30-48 blocks)

    # Hazard binary labels: ensure both classes are represented
    hazard = rng.choice([0, 1], size=total_steps, p=[0.7, 0.3]).astype(int)

    # Occlusion binary labels: ensure both classes are represented
    occlusion = rng.choice([0, 1], size=total_steps, p=[0.6, 0.4]).astype(int)

    # Normalized EMA targets (B, T, 256)
    ema_targets = rng.normal(0.0, 1.0, size=(num_episodes, clip_len, 256)).astype(np.float32)
    ema_targets /= (np.linalg.norm(ema_targets, axis=-1, keepdims=True) + 1e-7)

    return {
        "shells": shells,
        "state": state,
        "action": action,
        "meta": meta,
        "hazard": hazard,
        "occlusion": occlusion,
        "ema_targets": ema_targets,
    }


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for probe suite."""
    parser = argparse.ArgumentParser(
        description="Frozen Representation & Dynamics Probe Suite for Squared Sense"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="m2_static",
        choices=["m1_full", "m2_static", "b1_lod0", "b2_lod2", "b3_mae", "b4_uniform32"],
        help="Model architecture variant to probe.",
    )
    parser.add_argument(
        "--train_fractions",
        type=float,
        nargs="+",
        default=[1.0],
        help="Episode-level training fractions for few-shot probing (e.g., 0.1 0.25 1.0).",
    )
    parser.add_argument(
        "--allow_upsample_fallback",
        action="store_true",
        help="Allow 2x upsampling from LOD-1 for b4_uniform32 when --region_dir is omitted (unit tests only).",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to trained model checkpoint (.pkl).",
    )
    parser.add_argument(
        "--temporal_ckpt",
        type=str,
        default=None,
        help="Path to trained Dual-LSTM checkpoint (.pkl) for m1_full.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to evaluation dataset directory.",
    )
    parser.add_argument(
        "--region_dir",
        type=str,
        default=None,
        help="Path to directory containing baked Minecraft regions (.mmap). If provided and --data_dir is omitted, episodes are loaded or generated directly from baked terrain.",
    )
    parser.add_argument(
        "--out_json",
        type=str,
        default=None,
        help="Output path for evaluation JSON summary table.",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Run fast smoke test with synthetic episodes (T=24).",
    )
    return parser.parse_args(args)


def run_probe_suite(
    model_type: str = "m2_static",
    ckpt: Optional[str] = None,
    temporal_ckpt: Optional[str] = None,
    data_dir: Optional[str] = None,
    region_dir: Optional[str] = None,
    out_json: Optional[str] = None,
    smoke_test: bool = True,
    train_fractions: Optional[List[float]] = None,
    allow_upsample_fallback: bool = False,
) -> Dict[str, Any]:
    """Execute the full frozen representation and dynamics probe suite."""
    rng = nnx.Rngs(42)
    effective_data_dir = data_dir or region_dir
    fractions = train_fractions if train_fractions else [1.0]

    # 1. Instantiate model
    temporal_model: Optional[DualLSTMTemporalIntegrator] = None
    if model_type == "m2_static":
        model: Any = SpatialJEPA(num_shells=3, rngs=rng)
    elif model_type == "m1_full":
        model = SpatialJEPA(num_shells=3, rngs=rng)
        temporal_model = DualLSTMTemporalIntegrator(rngs=rng)
    elif model_type == "b1_lod0":
        model = SingleLODJEPA(lod_index=0, rngs=rng)
    elif model_type == "b2_lod2":
        model = SingleLODJEPA(lod_index=2, rngs=rng)
    elif model_type == "b3_mae":
        model = Generative3DMAE(rngs=rng)
    elif model_type == "b4_uniform32":
        model = Uniform32JEPA(rngs=rng)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Load checkpoints if provided
    if ckpt and os.path.exists(ckpt):
        load_checkpoint(ckpt, model)
    if temporal_ckpt and temporal_model and os.path.exists(temporal_ckpt):
        load_checkpoint(temporal_ckpt, temporal_model)

    # 2. Prepare evaluation data
    train_ep_lens: Optional[List[int]] = None
    if smoke_test or effective_data_dir is None:
        num_episodes = 4
        clip_len = 24
        data = generate_synthetic_probe_data(num_episodes=num_episodes, clip_len=clip_len)

        shells = data["shells"]
        if model_type == "b4_uniform32":
            shells_in = np.random.default_rng(42).uniform(
                0.0, 1.0, size=(len(shells), 32, 32, 32, 16)
            ).astype(np.float32)
        else:
            shells_in = shells
        state = data["state"]
        action = data["action"]
        meta = data["meta"]
        hazard = data["hazard"]
        occlusion = data["occlusion"]
        ema_targets = data["ema_targets"]

        # 3. Extract frozen representations
        features = extract_features(
            model=model,
            model_type=model_type,
            shells=shells_in,
            state=state,
            action=action,
            temporal_model=temporal_model,
            batch_size=32,
            episode_lengths=[clip_len] * num_episodes,
        )

        # Train/test split by episode ID (NEVER frame shuffle)
        # Episode 0..2 (72 frames) train, Episode 3 (24 frames) test
        train_n = 3 * clip_len
        train_ep_lens = [clip_len, clip_len, clip_len]
        X_train, X_test = features[:train_n], features[train_n:]
        y_train_goal, y_test_goal = meta[:train_n, 3:6], meta[train_n:, 3:6]
        y_train_haz, y_test_haz = hazard[:train_n], hazard[train_n:]
        v_train, v_test = state[:train_n, 0:3], state[train_n:, 0:3]
        occ_train, occ_test = occlusion[:train_n], occlusion[train_n:]

        # Multi-step rollout
        if model_type == "b4_uniform32":
            tok_seq = shells_in.reshape((num_episodes, clip_len, 1, 32, 32, 32, 16))
        else:
            tok_seq = shells.reshape((num_episodes, clip_len, 3, 16, 16, 16, 16))
            if model_type in ("b1_lod0",):
                tok_seq = tok_seq[:, :, 0:1]
            elif model_type in ("b2_lod2",):
                tok_seq = tok_seq[:, :, 2:3]
        encoder = model.context_encoder if hasattr(model, "context_encoder") else (
            model.encoder if hasattr(model, "encoder") else model
        )

        flat_tok = _forward_tokens(encoder, tok_seq.reshape(-1, *tok_seq.shape[2:]))
        tokens_seq_4d = flat_tok.reshape((num_episodes, clip_len, flat_tok.shape[1], 256))
        state_seq = state.reshape((num_episodes, clip_len, 12))
        action_seq = action.reshape((num_episodes, clip_len))

        rollout_res = eval_multistep_rollout(
            temporal_model=temporal_model,
            tokens_seq=tokens_seq_4d,
            state_seq=state_seq,
            action_seq=action_seq,
            ema_targets_seq=ema_targets,
            horizons=(1, 2, 4, 8),
        )
    else:
        splits = load_episode_splits(effective_data_dir, seed=42)
        train_eps = splits["train"]
        test_eps = splits["test"] if len(splits["test"]) > 0 else splits["val"]
        train_ep_lens = [len(ep["shells"]) for ep in train_eps]

        def _process_episodes(ep_list: List[Dict[str, np.ndarray]]):
            sh_list, st_list, ac_list, me_list, hz_list, oc_list = [], [], [], [], [], []
            for ep in ep_list:
                sh = ep["shells"]
                st = ep["state"]
                ac = ep["action"]
                me = ep["meta"]
                hz = compute_local_hazard_labels(sh)
                oc = compute_occlusion_labels(hz)

                sh_list.append(sh)
                st_list.append(st)
                ac_list.append(ac)
                me_list.append(me)
                hz_list.append(hz)
                oc_list.append(oc)

            cat_sh = np.concatenate(sh_list, axis=0)
            cat_st = np.concatenate(st_list, axis=0)
            cat_ac = np.concatenate(ac_list, axis=0)
            cat_me = np.concatenate(me_list, axis=0)
            cat_hz = np.concatenate(hz_list, axis=0)
            cat_oc = np.concatenate(oc_list, axis=0)
            return cat_sh, cat_st, cat_ac, cat_me, cat_hz, cat_oc

        train_sh, train_st, train_ac, train_me, train_hz, train_oc = _process_episodes(train_eps)
        test_sh, test_st, test_ac, test_me, test_hz, test_oc = _process_episodes(test_eps)

        if model_type == "b4_uniform32":
            if region_dir and os.path.isdir(region_dir):
                import math as _math
                from env.lod_extractor import WorldVolume, extract_uniform32_16ch
                from env.palette import get_palette
                palette = get_palette()
                world = WorldVolume(region_dir, palette_obj=palette)

                def _yaw_to_cardinal(sin_y: float, cos_y: float) -> str:
                    yaw = (_math.degrees(_math.atan2(-sin_y, cos_y)) % 360.0 + 360.0) % 360.0
                    if 45.0 <= yaw < 135.0:
                        return "WEST"
                    if 135.0 <= yaw < 225.0:
                        return "NORTH"
                    if 225.0 <= yaw < 315.0:
                        return "EAST"
                    return "SOUTH"

                def _extract_32_batch(me_arr, st_arr):
                    return np.stack([
                        extract_uniform32_16ch(
                            world, palette,
                            (int(round(me_arr[i, 0])), int(round(me_arr[i, 1])), int(round(me_arr[i, 2]))),
                            _yaw_to_cardinal(float(st_arr[i, 6]), float(st_arr[i, 7])),
                            float(st_arr[i, 8]),
                            as_uint8=False,
                        )
                        for i in range(len(me_arr))
                    ], axis=0)

                train_sh_in = _extract_32_batch(train_me, train_st)
                test_sh_in = _extract_32_batch(test_me, test_st)
            elif allow_upsample_fallback:
                train_sh_in = np.repeat(np.repeat(np.repeat(train_sh[:, 1], 2, axis=1), 2, axis=2), 2, axis=3)
                test_sh_in = np.repeat(np.repeat(np.repeat(test_sh[:, 1], 2, axis=1), 2, axis=2), 2, axis=3)
            else:
                raise ValueError(
                    "b4_uniform32 in probe_suite requires --region_dir pointing to baked Minecraft regions (.mmap) "
                    "for true 1x 32x32x32x16 extraction. Pass --allow_upsample_fallback only for unit tests."
                )
        else:
            train_sh_in = train_sh
            test_sh_in = test_sh

        X_train = extract_features(
            model=model,
            model_type=model_type,
            shells=train_sh_in,
            state=train_st,
            action=train_ac,
            temporal_model=temporal_model,
            batch_size=32,
            episode_lengths=train_ep_lens,
        )
        X_test = extract_features(
            model=model,
            model_type=model_type,
            shells=test_sh_in,
            state=test_st,
            action=test_ac,
            temporal_model=temporal_model,
            batch_size=32,
            episode_lengths=[len(ep["shells"]) for ep in test_eps],
        )

        y_train_goal, y_test_goal = train_me[:, 3:6], test_me[:, 3:6]
        y_train_haz, y_test_haz = train_hz, test_hz
        v_train, v_test = train_st[:, 0:3], test_st[:, 0:3]
        occ_train, occ_test = train_oc, test_oc

        # Multi-step rollout on test episodes
        clips = extract_temporal_clips(test_eps, clip_len=24, stride=12)
        if len(clips["shells"]) == 0:
            clips = extract_temporal_clips(train_eps, clip_len=24, stride=12)

        clip_shells = clips["shells"]
        b, T = clip_shells.shape[:2]
        flat_shells = clip_shells.reshape((b * T, *clip_shells.shape[2:]))

        encoder = model.context_encoder if hasattr(model, "context_encoder") else (
            model.encoder if hasattr(model, "encoder") else model
        )
        target_encoder = model.target_encoder if hasattr(model, "target_encoder") else encoder

        if model_type == "b4_uniform32":
            eval_flat_shells = np.repeat(np.repeat(np.repeat(flat_shells[:, 1:2], 2, axis=2), 2, axis=3), 2, axis=4)
        elif model_type in ("b1_lod0",):
            eval_flat_shells = flat_shells[:, 0:1]
        elif model_type in ("b2_lod2",):
            eval_flat_shells = flat_shells[:, 2:3]
        else:
            eval_flat_shells = flat_shells

        flat_tok = _forward_tokens(encoder, eval_flat_shells)
        tokens_seq_4d = flat_tok.reshape((b, T, flat_tok.shape[1], 256))

        # Compute EMA targets directly on the fly
        target_tokens = _forward_tokens(target_encoder, eval_flat_shells)
        mean_pooled = jnp.mean(target_tokens, axis=1)  # (b * T, 256)
        mean = jnp.mean(mean_pooled, axis=-1, keepdims=True)
        var = jnp.var(mean_pooled, axis=-1, keepdims=True)
        norm_pooled = (mean_pooled - mean) / jnp.sqrt(var + 1e-6)
        ema_targets = np.asarray(norm_pooled).reshape((b, T, 256))

        rollout_res = eval_multistep_rollout(
            temporal_model=temporal_model,
            tokens_seq=tokens_seq_4d,
            state_seq=clips["states"],
            action_seq=clips["actions"],
            ema_targets_seq=ema_targets,
            horizons=(1, 2, 4, 8),
        )
        features = X_test

    # 4. Run frozen probes (and optional episode-level few-shot sweep)
    goal_res = probe_distant_goal(X_train, y_train_goal, X_test, y_test_goal)
    hazard_res = probe_local_hazard(X_train, y_train_haz, X_test, y_test_haz)
    kin_occ_res = probe_kinematic_and_occlusion(X_train, v_train, occ_train, X_test, v_test, occ_test)

    few_shot: Dict[str, Any] = {}
    if train_ep_lens is not None and any(f < 1.0 for f in fractions):
        ep_slices: List[Tuple[int, int]] = []
        off = 0
        for el in train_ep_lens:
            ep_slices.append((off, off + el))
            off += el
        n_eps = len(ep_slices)
        for frac in fractions:
            if frac >= 1.0:
                few_shot["1.0"] = {
                    "distant_goal": goal_res,
                    "local_hazard": hazard_res,
                    "kinematic_and_occlusion": kin_occ_res,
                }
                continue
            k_eps = max(1, int(round(frac * n_eps)))
            seed_goals, seed_hazs, seed_kins = [], [], []
            for s in (0, 1, 2):
                s_rng = np.random.default_rng(s)
                chosen = s_rng.choice(n_eps, size=k_eps, replace=False)
                idx_list = np.concatenate([np.arange(ep_slices[c][0], ep_slices[c][1]) for c in chosen])
                seed_goals.append(probe_distant_goal(X_train[idx_list], y_train_goal[idx_list], X_test, y_test_goal, seeds=(s,)))
                seed_hazs.append(probe_local_hazard(X_train[idx_list], y_train_haz[idx_list], X_test, y_test_haz, seeds=(s,)))
                seed_kins.append(probe_kinematic_and_occlusion(X_train[idx_list], v_train[idx_list], occ_train[idx_list], X_test, v_test, occ_test, seeds=(s,)))
            few_shot[str(frac)] = {
                "num_train_episodes": k_eps,
                "distant_goal": {
                    "r2": float(np.mean([g["r2"] for g in seed_goals])),
                    "r2_std": float(np.std([g["r2"] for g in seed_goals])),
                    "mean_cosine_sim": float(np.mean([g["mean_cosine_sim"] for g in seed_goals])),
                },
                "local_hazard": {
                    "f1": float(np.mean([h["f1"] for h in seed_hazs])),
                    "f1_std": float(np.std([h["f1"] for h in seed_hazs])),
                    "auroc": float(np.mean([h["auroc"] for h in seed_hazs])),
                },
                "kinematic_and_occlusion": {
                    "velocity_r2": float(np.mean([k["velocity_r2"] for k in seed_kins])),
                    "occlusion_f1": float(np.mean([k["occlusion_f1"] for k in seed_kins])),
                },
            }

    # 5. Compile summary table
    summary: Dict[str, Any] = {
        "model_type": model_type,
        "feature_dim": int(features.shape[1]),
        "distant_goal": goal_res,
        "local_hazard": hazard_res,
        "kinematic_and_occlusion": kin_occ_res,
        "multistep_rollout": rollout_res,
        # Flat keys for direct metric lookups
        "r2": goal_res["r2"],
        "mean_cosine_sim": goal_res["mean_cosine_sim"],
        "f1": hazard_res["f1"],
        "auroc": hazard_res["auroc"],
        "velocity_r2": kin_occ_res["velocity_r2"],
        "occlusion_f1": kin_occ_res["occlusion_f1"],
        "rankme": rollout_res["rankme"],
    }
    if few_shot:
        summary["few_shot"] = few_shot
    for h in (1, 2, 4, 8):
        summary[f"cosine_sim_h{h}"] = rollout_res[f"cosine_sim_h{h}"]
        summary[f"l1_error_h{h}"] = rollout_res[f"l1_error_h{h}"]

    if out_json:
        os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    summary = run_probe_suite(
        model_type=args.model_type,
        ckpt=args.ckpt,
        temporal_ckpt=args.temporal_ckpt,
        data_dir=args.data_dir,
        region_dir=args.region_dir,
        out_json=args.out_json,
        smoke_test=args.smoke_test,
        train_fractions=args.train_fractions,
        allow_upsample_fallback=args.allow_upsample_fallback,
    )
    print(f"\n=== Evaluation Probe Summary ({summary['model_type']}) ===")
    print(f"Feature Dimension       : {summary['feature_dim']}")
    print(f"Distant Goal R²         : {summary['distant_goal']['r2']:.4f} ± {summary['distant_goal']['r2_std']:.4f}")
    print(f"Distant Goal Cosine Sim : {summary['distant_goal']['mean_cosine_sim']:.4f} ± {summary['distant_goal']['mean_cosine_sim_std']:.4f}")
    print(f"Local Hazard F1         : {summary['local_hazard']['f1']:.4f} ± {summary['local_hazard']['f1_std']:.4f}")
    print(f"Local Hazard AUROC      : {summary['local_hazard']['auroc']:.4f} ± {summary['local_hazard']['auroc_std']:.4f}")
    print(f"Velocity R²             : {summary['kinematic_and_occlusion']['velocity_r2']:.4f} ± {summary['kinematic_and_occlusion']['velocity_r2_std']:.4f}")
    print(f"Occlusion Memory F1     : {summary['kinematic_and_occlusion']['occlusion_f1']:.4f} ± {summary['kinematic_and_occlusion']['occlusion_f1_std']:.4f}")
    print(f"Uncentered RankMe       : {summary['multistep_rollout']['rankme']:.2f}")
    for h in (1, 2, 4, 8):
        print(f"Rollout H={h} Cosine Sim  : {summary['multistep_rollout'][f'cosine_sim_h{h}']:.4f}")
        print(f"Rollout H={h} L1 Error    : {summary['multistep_rollout'][f'l1_error_h{h}']:.4f}")
    print("===================================================\n")


if __name__ == "__main__":
    main()

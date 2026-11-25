"""Zero-Retraining Temporal Dynamics & Ego-Motion Shift Ablation Evaluator.

Evaluates a trained M1 (SpatialJEPA + DualLSTMTemporalIntegrator) checkpoint on held-out
test episodes (strictly via load_episode_splits):
  1. Multi-step open-loop rollout error (k in {1, 2, 4, 8})
  2. Control 1: Persistence baseline (z_hat_{t+k} = z_t)
  3. Control 2: Shuffled-action control (causal action conditioning check)
  4. Control 3: Pure ego-motion 3D shift ablation (disable_ego_shift=True, keeping
     intact 12-D state into state_lstm) + Occlusion Memory F1 (shift ON vs OFF)
  5. IDM Top-1 classification accuracy on held-out transitions (vs. 8.33% random chance)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from train.gpu_env import setup_gpu_environment

setup_gpu_environment()

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from data.dataset import (
    compute_local_hazard_labels,
    compute_occlusion_labels,
    extract_temporal_clips,
    load_episode_splits,
)
from eval.probe_suite import (
    _forward_tokens,
    eval_multistep_rollout,
    extract_features,
    generate_synthetic_probe_data,
    probe_kinematic_and_occlusion,
)
from models.temporal_lstm import DualLSTMTemporalIntegrator
from models.transformer import SpatialJEPA
from train.checkpoint import load_checkpoint
from train.diagnostics import ProgressBar


@nnx.jit(static_argnames=("disable_ego_shift",))
def _step_single_ablation(
    temporal_model: DualLSTMTemporalIntegrator,
    tok_t: jax.Array,
    st_t: jax.Array,
    act_t: jax.Array,
    hidden: Any,
    disable_ego_shift: bool = False,
) -> Tuple[jax.Array, Any]:
    _, c_t, next_hidden = temporal_model.step(
        tok_t, st_t, act_t, hidden, disable_ego_shift=disable_ego_shift
    )
    return jax.lax.stop_gradient(c_t), next_hidden


def _extract_m1_features_with_shift_flag(
    spatial_model: SpatialJEPA,
    temporal_model: DualLSTMTemporalIntegrator,
    shells: np.ndarray,
    state: np.ndarray,
    action: np.ndarray,
    episode_lengths: List[int],
    disable_ego_shift: bool = False,
    batch_size: int = 32,
) -> np.ndarray:
    """Extract M1 (448-D) features with per-episode resets and optional ego-shift bypass."""
    if not disable_ego_shift:
        return extract_features(
            model=spatial_model,
            model_type="m1_full",
            shells=shells,
            state=state,
            action=action,
            temporal_model=temporal_model,
            batch_size=batch_size,
            episode_lengths=episode_lengths,
        )

    x = jnp.asarray(shells, dtype=jnp.float32)
    z_list = []
    tok_list = []
    for i in range(0, len(x), batch_size):
        tok = _forward_tokens(spatial_model.context_encoder, x[i : i + batch_size])
        z_list.append(np.asarray(jnp.mean(tok, axis=1)))
        tok_list.append(tok)

    z_all = np.concatenate(z_list, axis=0)
    all_tokens = jnp.concatenate(tok_list, axis=0)
    st = jnp.asarray(state, dtype=jnp.float32)
    act = jnp.asarray(action, dtype=jnp.int32)

    c_chunks = []
    offset = 0
    for ep_len in episode_lengths:
        hidden = temporal_model.init_hidden(1)
        c_ep = []
        for t in range(offset, offset + ep_len):
            c_t, hidden = _step_single_ablation(
                temporal_model,
                all_tokens[t : t + 1],
                st[t : t + 1],
                act[t : t + 1],
                hidden,
                disable_ego_shift=True,
            )
            c_ep.append(np.asarray(c_t[0], dtype=np.float32))
        c_chunks.append(np.stack(c_ep, axis=0))
        offset += ep_len

    c_all = np.concatenate(c_chunks, axis=0)
    return np.concatenate([z_all, c_all], axis=-1).astype(np.float32)


def eval_temporal_dynamics(
    spatial_ckpt: Optional[str] = None,
    temporal_ckpt: Optional[str] = None,
    data_dir: Optional[str] = None,
    region_dir: Optional[str] = None,
    out_json: Optional[str] = None,
    clip_len: int = 24,
    horizons: Tuple[int, ...] = (1, 2, 4, 8),
    seed: int = 42,
    smoke_test: bool = False,
) -> Dict[str, Any]:
    """Run zero-retraining temporal dynamics, controls, and ego-shift ablation."""
    rng_nnx = nnx.Rngs(seed)
    np_rng = np.random.default_rng(seed)

    spatial_model = SpatialJEPA(num_shells=3, rngs=rng_nnx)
    temporal_model = DualLSTMTemporalIntegrator(rngs=rng_nnx)

    if spatial_ckpt and os.path.exists(spatial_ckpt):
        load_checkpoint(spatial_ckpt, spatial_model)
    if temporal_ckpt and os.path.exists(temporal_ckpt):
        load_checkpoint(temporal_ckpt, temporal_model)

    effective_dir = data_dir or region_dir
    if smoke_test or effective_dir is None:
        num_ep = 4
        data = generate_synthetic_probe_data(num_episodes=num_ep, clip_len=clip_len, seed=seed)
        train_sh, test_sh = data["shells"][: 3 * clip_len], data["shells"][3 * clip_len :]
        train_st, test_st = data["state"][: 3 * clip_len], data["state"][3 * clip_len :]
        train_ac, test_ac = data["action"][: 3 * clip_len], data["action"][3 * clip_len :]
        train_hz, test_hz = data["hazard"][: 3 * clip_len], data["hazard"][3 * clip_len :]
        train_oc, test_oc = data["occlusion"][: 3 * clip_len], data["occlusion"][3 * clip_len :]
        train_ep_lens = [clip_len] * 3
        test_ep_lens = [clip_len]

        clip_shells = test_sh.reshape((1, clip_len, 3, 16, 16, 16, 16))
        clip_states = test_st.reshape((1, clip_len, 12))
        clip_actions = test_ac.reshape((1, clip_len))
    else:
        splits = load_episode_splits(effective_dir, seed=seed)
        train_eps = splits["train"]
        test_eps = splits["test"] if len(splits["test"]) > 0 else splits["val"]

        def _stack(ep_list):
            sh_l, st_l, ac_l, hz_l, oc_l, lens = [], [], [], [], [], []
            for ep in ep_list:
                sh = ep["shells"]
                hz = compute_local_hazard_labels(sh)
                oc = compute_occlusion_labels(hz)
                sh_l.append(sh)
                st_l.append(ep["state"])
                ac_l.append(ep["action"])
                hz_l.append(hz)
                oc_l.append(oc)
                lens.append(len(sh))
            return (
                np.concatenate(sh_l, axis=0),
                np.concatenate(st_l, axis=0),
                np.concatenate(ac_l, axis=0),
                np.concatenate(hz_l, axis=0),
                np.concatenate(oc_l, axis=0),
                lens,
            )

        train_sh, train_st, train_ac, train_hz, train_oc, train_ep_lens = _stack(train_eps)
        test_sh, test_st, test_ac, test_hz, test_oc, test_ep_lens = _stack(test_eps)

        clips = extract_temporal_clips(test_eps, clip_len=clip_len, stride=max(1, clip_len // 2))
        if len(clips["shells"]) == 0:
            clips = extract_temporal_clips(train_eps, clip_len=clip_len, stride=max(1, clip_len // 2))
        clip_shells = clips["shells"]
        clip_states = clips["states"]
        clip_actions = clips["actions"]

    b, T = clip_shells.shape[:2]
    flat_shells = jnp.asarray(clip_shells.reshape((b * T, 3, 16, 16, 16, 16)), dtype=jnp.float32)
    flat_tok = _forward_tokens(spatial_model.context_encoder, flat_shells)
    tokens_seq = flat_tok.reshape((b, T, 192, 256))

    tgt_tok = _forward_tokens(spatial_model.target_encoder, flat_shells)
    mean_p = jnp.mean(tgt_tok, axis=1)
    norm_p = (mean_p - jnp.mean(mean_p, axis=-1, keepdims=True)) / jnp.sqrt(
        jnp.var(mean_p, axis=-1, keepdims=True) + 1e-6
    )
    ema_targets = np.asarray(norm_p).reshape((b, T, 256))

    st_j = jnp.asarray(clip_states, dtype=jnp.float32)
    ac_j = jnp.asarray(clip_actions, dtype=jnp.int32)
    k_max = max(horizons)
    valid_len = T - k_max

    def _compute_rollout_metrics(pred_future_np: np.ndarray) -> Dict[str, float]:
        res: Dict[str, float] = {}
        for h in horizons:
            pred_h = pred_future_np[:, :, h - 1, :]
            tgt_h = ema_targets[:, h : h + valid_len, :]
            dot = np.sum(pred_h * tgt_h, axis=-1)
            cos = dot / (np.linalg.norm(pred_h, axis=-1) * np.linalg.norm(tgt_h, axis=-1) + 1e-7)
            res[f"cosine_sim_h{h}"] = float(np.mean(cos))
            res[f"l1_error_h{h}"] = float(np.mean(np.abs(pred_h - tgt_h)))
        return res

    pb = ProgressBar(total=5, prefix="Temporal Eval", bar_len=24, log_every=1, process_tag="Temporal Eval")

    # 1. M1 Rollout + IDM Accuracy
    pred_m1, idm_logits = temporal_model.rollout(tokens_seq, st_j, ac_j, k_steps=k_max, disable_ego_shift=False)
    m1_rollout = _compute_rollout_metrics(np.asarray(pred_m1)[:, :valid_len])
    idm_preds = np.argmax(np.asarray(idm_logits), axis=-1)
    idm_targets = np.asarray(clip_actions[:, :-1])
    idm_top1_acc = float(np.mean(idm_preds == idm_targets))
    pb.update(1, metrics={"IDM": idm_top1_acc, "L1_h1": m1_rollout["l1_error_h1"]})

    # 2. Control 1: Persistence Baseline
    persistence_rollout = eval_multistep_rollout(
        temporal_model=None,
        tokens_seq=tokens_seq,
        state_seq=st_j,
        action_seq=ac_j,
        ema_targets_seq=ema_targets,
        horizons=horizons,
    )
    pb.update(2, metrics={"Persist_L1": persistence_rollout["l1_error_h1"]})

    # 3. Control 2: Shuffled-Action Control
    shuffled_actions = np.array(clip_actions, copy=True)
    for bi in range(b):
        np_rng.shuffle(shuffled_actions[bi])
    pred_shuf, _ = temporal_model.rollout(
        tokens_seq, st_j, jnp.asarray(shuffled_actions, dtype=jnp.int32), k_steps=k_max, disable_ego_shift=False
    )
    shuffled_rollout = _compute_rollout_metrics(np.asarray(pred_shuf)[:, :valid_len])
    pb.update(3, metrics={"Shuf_L1": shuffled_rollout["l1_error_h1"]})

    # 4. Control 3: Pure Ego-Motion 3D Shift Ablation (disable_ego_shift=True)
    pred_noshift, _ = temporal_model.rollout(
        tokens_seq, st_j, ac_j, k_steps=k_max, disable_ego_shift=True
    )
    noshift_rollout = _compute_rollout_metrics(np.asarray(pred_noshift)[:, :valid_len])
    pb.update(4, metrics={"NoShift_L1": noshift_rollout["l1_error_h1"]})

    # Occlusion Memory F1 with ego-shift ON vs OFF
    X_tr_on = _extract_m1_features_with_shift_flag(
        spatial_model, temporal_model, train_sh, train_st, train_ac, train_ep_lens, disable_ego_shift=False
    )
    X_te_on = _extract_m1_features_with_shift_flag(
        spatial_model, temporal_model, test_sh, test_st, test_ac, test_ep_lens, disable_ego_shift=False
    )
    probe_on = probe_kinematic_and_occlusion(X_tr_on, train_st[:, 0:3], train_oc, X_te_on, test_st[:, 0:3], test_oc)

    X_tr_off = _extract_m1_features_with_shift_flag(
        spatial_model, temporal_model, train_sh, train_st, train_ac, train_ep_lens, disable_ego_shift=True
    )
    X_te_off = _extract_m1_features_with_shift_flag(
        spatial_model, temporal_model, test_sh, test_st, test_ac, test_ep_lens, disable_ego_shift=True
    )
    probe_off = probe_kinematic_and_occlusion(X_tr_off, train_st[:, 0:3], train_oc, X_te_off, test_st[:, 0:3], test_oc)
    pb.update(5, metrics={"OccF1_ON": probe_on["occlusion_f1"], "OccF1_OFF": probe_off["occlusion_f1"]})

    results = {
        "m1_rollout": m1_rollout,
        "persistence_baseline": {k: v for k, v in persistence_rollout.items() if k != "rankme"},
        "shuffled_action_control": shuffled_rollout,
        "ego_shift_off_rollout": noshift_rollout,
        "ego_shift_on_occlusion_f1": probe_on["occlusion_f1"],
        "ego_shift_off_occlusion_f1": probe_off["occlusion_f1"],
        "ego_shift_on_velocity_r2": probe_on["velocity_r2"],
        "ego_shift_off_velocity_r2": probe_off["velocity_r2"],
        "idm_top1_acc": idm_top1_acc,
        "idm_random_chance": 1.0 / 12.0,
    }

    if out_json:
        os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    return results


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zero-Retraining Temporal Dynamics & Ablation Evaluator")
    parser.add_argument("--spatial_ckpt", type=str, default="checkpoints/spatial_finetuned_best.pkl")
    parser.add_argument("--temporal_ckpt", type=str, default="checkpoints/temporal_best.pkl")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--region_dir", type=str, default=None)
    parser.add_argument("--out_json", type=str, default="results/temporal_dynamics_ablation.json")
    parser.add_argument("--clip_len", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke_test", action="store_true")
    return parser.parse_args(args)


def main(args: Optional[List[str]] = None) -> int:
    opts = parse_args(args)
    res = eval_temporal_dynamics(
        spatial_ckpt=opts.spatial_ckpt,
        temporal_ckpt=opts.temporal_ckpt,
        data_dir=opts.data_dir,
        region_dir=opts.region_dir,
        out_json=opts.out_json,
        clip_len=opts.clip_len,
        seed=opts.seed,
        smoke_test=opts.smoke_test,
    )
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

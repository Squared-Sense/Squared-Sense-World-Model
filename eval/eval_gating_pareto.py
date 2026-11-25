"""Event-Driven Shell Gating Pareto Frontier Sweep (Zero Retraining).

Evaluates an already-trained M1 (m1_full) RL1 policy across
tau_gate in [0.0, 0.01, 0.02, 0.05, 0.10, 0.20] on held-out seeds, reporting:
  - sr_pct, spl, mean_latent_cosine_vs_dense, mean_mflops_per_step, mean_latency_ms
  - Per-shell skip rates (%) for LOD-0, LOD-1, LOD-2, and full Transformer skip %
  - Breakdown overall and by motion regime (Idle, Pitch-Only, Straight Walk, Active Turn)

Uses verified XLA cost_analysis() component MFLOPs:
  - 131.18 MFLOPs per shell CNN
  - 1376.50 MFLOPs for 192-token Transformer
  - 21.25 MFLOPs for DualLSTMTemporalIntegrator step
  -> tau_gate = 0.0 dense step = 3 * 131.18 + 1376.50 + 21.25 = 1791.29 (~1791.30) MFLOPs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
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

from env.minecraft_env import MinecraftNavEnv
from eval.sim_env import VoxelNavEnv, make_debug_bridge_course
from eval.train_ppo_nav import ActorCriticHead, _forward_encoder_tokens, _forward_temporal_step
from models.gating import EventDrivenShellGate
from models.temporal_lstm import DualLSTMTemporalIntegrator
from models.transformer import SpatialJEPA
from train.checkpoint import load_checkpoint
from train.diagnostics import ProgressBar

XLA_MFLOPS_PER_SHELL: float = 131.18
XLA_MFLOPS_TRANSFORMER: float = 1376.50
XLA_MFLOPS_DUAL_LSTM: float = 21.25


def _classify_regime(action: int) -> str:
    if action == 11:
        return "idle"
    if action in (9, 10):
        return "pitch_only"
    if action in (6, 7):
        return "active_turn"
    return "straight_walk"


def eval_gating_pareto(
    policy_ckpt: Optional[str] = None,
    backbone_ckpt: Optional[str] = None,
    temporal_ckpt: Optional[str] = None,
    region_dir: Optional[str] = None,
    taus: Tuple[float, ...] = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20),
    num_episodes: int = 10,
    max_steps: int = 250,
    seed: int = 1000,
    out_json: Optional[str] = None,
    smoke_test: bool = False,
) -> Dict[str, Any]:
    """Run gating Pareto sweep over tau_gate values."""
    rng = nnx.Rngs(seed)
    backbone = SpatialJEPA(num_shells=3, rngs=rng)
    temporal_model = DualLSTMTemporalIntegrator(rngs=rng)
    policy = ActorCriticHead(feat_dim=448, num_actions=12, rngs=rng)

    if backbone_ckpt and os.path.exists(backbone_ckpt):
        load_checkpoint(backbone_ckpt, backbone)
    if temporal_ckpt and os.path.exists(temporal_ckpt):
        load_checkpoint(temporal_ckpt, temporal_model)
    if policy_ckpt and os.path.exists(policy_ckpt):
        load_checkpoint(policy_ckpt, policy)

    if smoke_test:
        taus = (0.0, 0.05)
        num_episodes = 2
        max_steps = 20

    use_real = bool(region_dir and os.path.isdir(region_dir))
    real_env = MinecraftNavEnv(region_dir=region_dir, max_steps=max_steps) if use_real else None

    pareto_results: List[Dict[str, Any]] = []
    pb = ProgressBar(
        total=len(taus) * num_episodes,
        prefix="Gating Pareto",
        bar_len=24,
        log_every=1,
        process_tag="Gating Pareto",
    )
    ep_counter = 0
    for tau in taus:
        gate = EventDrivenShellGate(backbone=backbone.context_encoder, tau_gate=float(tau))
        sr_list, spl_list = [], []
        cos_list, lat_list = [], []

        regime_totals: Dict[str, int] = {"idle": 0, "pitch_only": 0, "straight_walk": 0, "active_turn": 0}
        regime_shell_skips: Dict[str, List[int]] = {k: [0, 0, 0] for k in regime_totals}
        regime_trans_skips: Dict[str, int] = {k: 0 for k in regime_totals}

        total_steps = 0
        total_shell_evals = [0, 0, 0]
        total_trans_evals = 0

        for ep_idx in range(num_episodes):
            if use_real and real_env is not None:
                env: Any = real_env
                obs = env.reset(seed=seed + ep_idx)
                l_ref = float(getattr(env, "shortest_path_len", 10.0))
            else:
                vol, start, wps = make_debug_bridge_course(seed=seed + ep_idx)
                l_ref = max(1.0, float(np.linalg.norm(np.array(wps[-1], dtype=float) - np.array(start, dtype=float))))
                env = VoxelNavEnv(volume=vol, start=start, waypoints=wps, max_steps=max_steps)
                obs = env.reset()

            gate.reset()
            temporal_hidden = temporal_model.init_hidden(1)
            done = False
            step_count = 0
            last_info: Dict[str, Any] = {}

            while not done and step_count < max_steps:
                step_count += 1
                total_steps += 1
                prev_shells = gate.prev_shells
                prev_trans_cnt = gate.transformer_eval_count
                prev_shell_cnts = list(gate.shell_eval_counts)

                t0 = time.perf_counter()
                tokens_gated = gate.step(obs["shells"])
                z_gated = jnp.mean(tokens_gated, axis=1)
                s_arr = jnp.asarray(obs["state"], dtype=jnp.float32).reshape((1, 12))
                a_arr = jnp.zeros((1,), dtype=jnp.int32)
                c_t, temporal_hidden = _forward_temporal_step(temporal_model, tokens_gated, s_arr, a_arr, temporal_hidden)
                feat = jnp.concatenate([z_gated, c_t], axis=-1)
                logits, _ = policy(feat, obs["goal_vec"])
                action = int(jnp.argmax(logits[0]))
                lat_list.append((time.perf_counter() - t0) * 1000.0)

                # Dense reference latent for cosine similarity check
                tokens_dense = _forward_encoder_tokens(
                    backbone.context_encoder, jnp.asarray(obs["shells"], dtype=jnp.float32)[None, ...]
                )
                z_dense = jnp.mean(tokens_dense, axis=1)
                dot = float(jnp.sum(z_gated * z_dense))
                cos_val = dot / (float(jnp.linalg.norm(z_gated) * jnp.linalg.norm(z_dense)) + 1e-7)
                cos_list.append(cos_val)

                # Per-step skip tracking by regime
                regime = _classify_regime(action)
                regime_totals[regime] += 1
                for l in range(3):
                    evaled = gate.shell_eval_counts[l] - prev_shell_cnts[l]
                    total_shell_evals[l] += evaled
                    if evaled == 0:
                        regime_shell_skips[regime][l] += 1
                trans_evaled = gate.transformer_eval_count - prev_trans_cnt
                total_trans_evals += trans_evaled
                if trans_evaled == 0:
                    regime_trans_skips[regime] += 1

                obs, _, done, last_info = env.step(action)

            succ = bool(last_info.get("success", False))
            plen = float(last_info.get("path_length", 0.0))
            sr_list.append(1.0 if succ else 0.0)
            spl_list.append(float(l_ref / max(plen, l_ref)) if succ else 0.0)
            ep_counter += 1
            pb.update(ep_counter, metrics={"tau": float(tau), "SR%": float(np.mean(sr_list) * 100.0)})

        denom = max(1, total_steps)
        mean_mflops = (
            sum(total_shell_evals) * XLA_MFLOPS_PER_SHELL
            + total_trans_evals * XLA_MFLOPS_TRANSFORMER
            + denom * XLA_MFLOPS_DUAL_LSTM
        ) / denom

        regime_breakdown = {}
        for r_name, r_tot in regime_totals.items():
            d = max(1, r_tot)
            regime_breakdown[r_name] = {
                "steps": r_tot,
                "shell_skip_pct": [(regime_shell_skips[r_name][l] / d) * 100.0 for l in range(3)],
                "transformer_skip_pct": (regime_trans_skips[r_name] / d) * 100.0,
            }

        pareto_results.append({
            "tau_gate": float(tau),
            "sr_pct": float(np.mean(sr_list) * 100.0),
            "spl": float(np.mean(spl_list)),
            "mean_latent_cosine_vs_dense": float(np.mean(cos_list)),
            "mean_mflops_per_step": float(mean_mflops),
            "mean_latency_ms": float(np.mean(lat_list)),
            "shell_skip_pct": [(1.0 - total_shell_evals[l] / denom) * 100.0 for l in range(3)],
            "transformer_skip_pct": (1.0 - total_trans_evals / denom) * 100.0,
            "by_regime": regime_breakdown,
        })

    output = {"taus": list(taus), "pareto": pareto_results}
    if out_json:
        os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
    return output


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Event-Driven Shell Gating Pareto Frontier Sweep")
    parser.add_argument("--policy_ckpt", type=str, default="checkpoints/ppo_nav_m1_overnight/best_ppo.pkl")
    parser.add_argument("--backbone_ckpt", type=str, default="checkpoints/spatial_finetuned_best.pkl")
    parser.add_argument("--temporal_ckpt", type=str, default="checkpoints/temporal_best.pkl")
    parser.add_argument("--region_dir", type=str, default=None)
    parser.add_argument("--taus", type=float, nargs="+", default=[0.0, 0.01, 0.02, 0.05, 0.10, 0.20])
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--out_json", type=str, default="results/gating_pareto.json")
    parser.add_argument("--smoke_test", action="store_true")
    return parser.parse_args(args)


def main(args: Optional[List[str]] = None) -> int:
    opts = parse_args(args)
    res = eval_gating_pareto(
        policy_ckpt=opts.policy_ckpt,
        backbone_ckpt=opts.backbone_ckpt,
        temporal_ckpt=opts.temporal_ckpt,
        region_dir=opts.region_dir,
        taus=tuple(opts.taus),
        num_episodes=opts.num_episodes,
        max_steps=opts.max_steps,
        seed=opts.seed,
        out_json=opts.out_json,
        smoke_test=opts.smoke_test,
    )
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

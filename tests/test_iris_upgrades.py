"""Smoke & unit tests for IRIS 27 experimental upgrades:
- Uniform32JEPA (B4) forward/grad, strict --region_dir check, and probe_suite support
- Few-shot episode-level probing (--train_fractions)
- Zero-retraining temporal dynamics & pure ego-shift ablation (eval_temporal_dynamics.py)
- Event-driven shell gating Pareto sweep (eval_gating_pareto.py)
"""

from __future__ import annotations

import json
import math
import os
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from eval.eval_gating_pareto import eval_gating_pareto
from eval.eval_temporal_dynamics import eval_temporal_dynamics
from eval.probe_suite import run_probe_suite
from models.baselines import Uniform32JEPA, sample_3d_jepa_masks_32
from train.losses import spatial_jepa_loss
from train.train_baselines import load_uniform32_dataset, parse_args as parse_base_args, train_baseline


def test_uniform32_jepa_forward_and_strict_region_check(tmp_path) -> None:
    rng = nnx.Rngs(0)
    model = Uniform32JEPA(rngs=rng)
    assert model.count_trainable_params() > 4_000_000

    x = jnp.ones((2, 32, 32, 32, 16), dtype=jnp.float32) * 0.5
    vis, tgt = sample_3d_jepa_masks_32(jax.random.key(0), 2)
    assert vis.shape == (2, 256)
    assert tgt.shape == (2, 256)

    @nnx.jit
    def grad_step(m: Uniform32JEPA):
        def loss_fn(mod: Uniform32JEPA):
            p, t, c = mod(x, vis, tgt)
            loss, _ = spatial_jepa_loss(p, t, c)
            return loss
        return nnx.value_and_grad(loss_fn)(m)

    loss, grads = grad_step(model)
    assert math.isfinite(float(loss))
    grad_norm = sum(float(jnp.sum(jnp.abs(g))) for g in jax.tree.leaves(grads.jepa.context_encoder))
    assert grad_norm > 0.0

    with pytest.raises(ValueError, match="--region_dir"):
        load_uniform32_dataset(
            data_dir="data",
            region_dir=None,
            smoke_test=False,
            seed=42,
            allow_upsample_fallback=False,
        )

    args = parse_base_args(["--model", "b4_uniform32", "--smoke_test", "--save_dir", str(tmp_path / "b4")])
    _, best_val = train_baseline(args)
    assert math.isfinite(best_val)


def test_probe_suite_b4_and_few_shot(tmp_path) -> None:
    out_json = tmp_path / "probe_b4.json"
    summary = run_probe_suite(
        model_type="b4_uniform32",
        out_json=str(out_json),
        smoke_test=True,
        train_fractions=[0.5, 1.0],
    )
    assert summary["feature_dim"] == 256
    assert "few_shot" in summary
    assert "0.5" in summary["few_shot"]
    assert math.isfinite(summary["few_shot"]["0.5"]["distant_goal"]["r2"])


def test_eval_temporal_dynamics_smoke(tmp_path) -> None:
    out_json = tmp_path / "temporal_ablation.json"
    res = eval_temporal_dynamics(out_json=str(out_json), smoke_test=True)
    assert os.path.isfile(out_json)
    for k in (
        "m1_rollout",
        "persistence_baseline",
        "shuffled_action_control",
        "ego_shift_off_rollout",
        "ego_shift_on_occlusion_f1",
        "ego_shift_off_occlusion_f1",
        "idm_top1_acc",
    ):
        assert k in res
    assert 0.0 <= res["idm_top1_acc"] <= 1.0


def test_eval_gating_pareto_smoke(tmp_path) -> None:
    out_json = tmp_path / "pareto.json"
    res = eval_gating_pareto(out_json=str(out_json), smoke_test=True)
    assert os.path.isfile(out_json)
    assert len(res["pareto"]) == 2
    p0 = res["pareto"][0]
    p1 = res["pareto"][1]
    assert p0["tau_gate"] == 0.0
    assert 21.25 <= p0["mean_mflops_per_step"] <= 1791.30
    assert p1["mean_mflops_per_step"] <= p0["mean_mflops_per_step"]
    assert p0["mean_latent_cosine_vs_dense"] == pytest.approx(1.0, abs=1e-4)

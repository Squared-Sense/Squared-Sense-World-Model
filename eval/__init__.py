"""Evaluation and simulation suite for Squared Sense World Model.

Modules:
  - probe_suite: Frozen representation probes and dynamics evaluation.
  - sim_env: Deterministic 3D voxel navigation simulator and test environments.
  - train_ppo_nav: Frozen-backbone PPO and BC policy training.
  - eval_navigation: Zero-shot course navigation evaluation and latency profiling.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "extract_features",
    "probe_distant_goal",
    "probe_local_hazard",
    "probe_kinematic_and_occlusion",
    "eval_multistep_rollout",
    "VoxelNavEnv",
    "make_debug_bridge_course",
    "CardinalSnapper",
    "extract_shells",
    "ActorCriticHead",
    "train_bc_warmstart",
    "train_ppo",
    "evaluate_navigation",
]

_ATTR_TO_MODULE = {
    "evaluate_navigation": "eval.eval_navigation",
    "eval_multistep_rollout": "eval.probe_suite",
    "extract_features": "eval.probe_suite",
    "probe_distant_goal": "eval.probe_suite",
    "probe_kinematic_and_occlusion": "eval.probe_suite",
    "probe_local_hazard": "eval.probe_suite",
    "CardinalSnapper": "eval.sim_env",
    "VoxelNavEnv": "eval.sim_env",
    "extract_shells": "eval.sim_env",
    "make_debug_bridge_course": "eval.sim_env",
    "ActorCriticHead": "eval.train_ppo_nav",
    "train_bc_warmstart": "eval.train_ppo_nav",
    "train_ppo": "eval.train_ppo_nav",
}


def __getattr__(name: str) -> Any:
    """Lazy import module attributes to prevent circular imports during CLI execution."""
    if name in _ATTR_TO_MODULE:
        module = importlib.import_module(_ATTR_TO_MODULE[name])
        val = getattr(module, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


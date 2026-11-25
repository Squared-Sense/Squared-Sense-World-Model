"""Modal Cloud GPU Training & Evaluation Runner for Squared Sense World Model.

Orchestrates serverless cloud GPU execution on Modal (https://modal.com):
  - Container Image: Debian Slim with CUDA 12 JAX, Flax NNX, Optax, Scikit-learn
  - Persistent Volume: 'squared-sense-vol' mounted at /vol
  - Acceleration: NVIDIA L4 (24 GB VRAM, default), configurable to A10G / A100
  - Live Streaming: Real-time stdout/stderr log streaming with wall-clock timestamps
  - Fault Tolerance: vol.commit() after every stage and sub-epoch

Strictly follows docs/TRAIN_CONTRACT.md, docs/EVAL_CONTRACT.md, and docs/MODEL_CONTRACT.md.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import List, Optional

import modal

# ==============================================================================
# Global Configuration & Hardware Targets
# ==============================================================================

# GPU Options: "L4" (24GB VRAM, cost-effective), "A10G" (24GB VRAM), "A100" (40GB/80GB VRAM)
DEFAULT_GPU = "L4"
DEFAULT_TIMEOUT_SEC = 60 * 60 * 4  # 4 Hours max runtime per function
DEFAULT_MEMORY_MB = 16384          # 16 GB Container Host RAM
VOLUME_NAME = "squared-sense-vol"
REMOTE_REPO_PATH = "/root/squared_sense"
REMOTE_VOL_PATH = "/vol"

# ==============================================================================
# Modal Application & Persistent Storage Volume
# ==============================================================================

app = modal.App("squared-sense-world-model")
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# ==============================================================================
# Container Image Definition
# ==============================================================================

cuda_jax_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "jax[cuda12]>=0.12.0.1",
        "flax>=0.12.12",
        "optax>=0.2.9",
        "numpy>=2.0.0",
        "scipy>=1.15.0",
        "scikit-learn>=1.6.0",
        "matplotlib>=3.10.0",
        "einops>=0.8.0",
        "nbtlib>=2.0.4",
        "psutil>=6.0.0",
    )
    .env({
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "PYTHONPATH": REMOTE_REPO_PATH,
        "PYTHONUNBUFFERED": "1",
    })
    .add_local_dir(
        ".",
        remote_path=REMOTE_REPO_PATH,
        ignore=[
            ".git",
            ".venv",
            ".venv*",
            "__pycache__",
            "data",
            "checkpoints",
            "results",
            "paper_figs/final",
            "*.npz",
            "*.pkl",
            "*.log",
            "*.mmap",
        ],
    )
)


def _exec_command(cmd: List[str], cwd: str = REMOTE_REPO_PATH) -> None:
    """Execute shell command with live unbuffered stdout streaming and error propagation."""
    cmd_str = " ".join(cmd)
    print(f"\n[Modal Exec] Running: {cmd_str} (cwd: {cwd})\n", flush=True)
    res = subprocess.run(cmd, cwd=cwd, text=True, bufsize=1)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {res.returncode}: {cmd_str}")


# ==============================================================================
# Remote Functions: GPU Verification & Health Diagnostics
# ==============================================================================

@app.function(
    image=cuda_jax_image,
    gpu=DEFAULT_GPU,
    timeout=600,
    volumes={REMOTE_VOL_PATH: vol},
)
def verify_gpu() -> dict:
    """Verify NVIDIA GPU driver, CUDA JAX backend, VRAM allocation, and Volume paths."""
    print("=== [1] NVIDIA Driver & Hardware Diagnostic (nvidia-smi) ===")
    try:
        subprocess.run(["nvidia-smi"], check=True)
    except Exception as e:
        print(f"Warning: nvidia-smi failed ({e})")

    print("\n=== [2] JAX & Flax CUDA Acceleration Check ===")
    import jax
    import flax
    import optax

    backend = jax.default_backend()
    devices = jax.devices()
    print(f"JAX Version     : {jax.__version__}")
    print(f"Flax Version    : {flax.__version__}")
    print(f"Optax Version   : {optax.__version__}")
    print(f"Default Backend : {backend}")
    print(f"Active Devices  : {devices}")

    if backend != "gpu":
        print(f"WARNING: JAX detected backend '{backend}', expected 'gpu'!")
    else:
        print("SUCCESS: JAX is fully connected to NVIDIA GPU via CUDA 12.")

    print("\n=== [3] Persistent Storage Volume Inspection (/vol) ===")
    vol_subdirs = ["data", "checkpoints", "results", "paper_figs"]
    vol_status = {}
    for sub in vol_subdirs:
        p = os.path.join(REMOTE_VOL_PATH, sub)
        if os.path.exists(p):
            items = os.listdir(p)
            vol_status[sub] = f"EXISTS ({len(items)} items)"
            print(f"  {p}: {len(items)} items")
            for item in items[:5]:
                print(f"    - {item}")
            if len(items) > 5:
                print(f"    ... and {len(items) - 5} more")
        else:
            vol_status[sub] = "NOT_FOUND"
            print(f"  {p}: Directory does not exist yet.")

    vol.commit()
    return {
        "jax_backend": backend,
        "devices": [str(d) for d in devices],
        "volume_status": vol_status,
    }


# ==============================================================================
# Remote Functions: Stage 1A Spatial JEPA Training
# ==============================================================================

@app.function(
    image=cuda_jax_image,
    gpu=DEFAULT_GPU,
    timeout=DEFAULT_TIMEOUT_SEC,
    memory=DEFAULT_MEMORY_MB,
    volumes={REMOTE_VOL_PATH: vol},
)
def train_spatial(
    epochs: int = 80,
    batch_size: int = 32,
    lr: float = 3e-4,
    weight_decay: float = 0.05,
    warmup_epochs: int = 10,
    alpha: float = 0.1,
    beta: float = 0.04,
    no_vicreg: bool = False,
    data_dir: str = f"{REMOTE_VOL_PATH}/data",
    region_dir: str = f"{REMOTE_VOL_PATH}/data/baked_regions",
    save_dir: str = f"{REMOTE_VOL_PATH}/checkpoints/m2_spatial_jepa",
    log_every: int = 25,
    smoke_test: bool = False,
) -> str:
    """Train MultiLODBackbone + JEPAPredictor via 3D spatial masking and EMA target."""
    os.makedirs(save_dir, exist_ok=True)
    cmd = [
        "python",
        "-u",
        "train/train_spatial_jepa.py",
        "--save_dir",
        save_dir,
        "--epochs",
        str(epochs),
        "--batch_size",
        str(batch_size),
        "--lr",
        str(lr),
        "--weight_decay",
        str(weight_decay),
        "--warmup_epochs",
        str(warmup_epochs),
        "--alpha",
        str(alpha),
        "--beta",
        str(beta),
        "--log_every",
        str(log_every),
        "--as_uint8",
        "--drop_last",
    ]
    if no_vicreg:
        cmd.append("--no_vicreg")
    if smoke_test:
        cmd.append("--smoke_test")
    else:
        if os.path.exists(data_dir):
            cmd.extend(["--data_dir", data_dir])
        if os.path.exists(region_dir):
            cmd.extend(["--region_dir", region_dir])

    _exec_command(cmd)
    vol.commit()
    print(f"\n[Modal] Committed checkpoints to volume: {save_dir}")
    return save_dir


# ==============================================================================
# Remote Functions: Unified Baselines Training (B1, B2, B3)
# ==============================================================================

@app.function(
    image=cuda_jax_image,
    gpu=DEFAULT_GPU,
    timeout=DEFAULT_TIMEOUT_SEC,
    memory=DEFAULT_MEMORY_MB,
    volumes={REMOTE_VOL_PATH: vol},
)
def train_baselines(
    model: str = "all",
    epochs: int = 80,
    batch_size: int = 32,
    lr: float = 3e-4,
    data_dir: str = f"{REMOTE_VOL_PATH}/data",
    region_dir: str = f"{REMOTE_VOL_PATH}/data/baked_regions",
    save_dir_prefix: str = f"{REMOTE_VOL_PATH}/checkpoints/baselines",
    log_every: int = 25,
    smoke_test: bool = False,
) -> List[str]:
    """Train parameter-matched baseline models: B1 (LOD-0), B2 (LOD-2), B3 (3D-MAE), B4 (Uniform 32^3)."""
    targets = ["b1_lod0", "b2_lod2", "b3_mae", "b4_uniform32"] if model == "all" else [model]
    trained = []

    for tgt in targets:
        save_dir = f"{save_dir_prefix}/{tgt}"
        os.makedirs(save_dir, exist_ok=True)
        print(f"\n=== Training Baseline [{tgt}] on Cloud GPU ({DEFAULT_GPU}) ===")
        cmd = [
            "python",
            "-u",
            "train/train_baselines.py",
            "--model",
            tgt,
            "--save_dir",
            save_dir,
            "--epochs",
            str(epochs),
            "--batch_size",
            str(batch_size),
            "--lr",
            str(lr),
            "--log_every",
            str(log_every),
            "--as_uint8",
            "--drop_last",
        ]
        if smoke_test:
            cmd.append("--smoke_test")
        else:
            if os.path.exists(data_dir):
                cmd.extend(["--data_dir", data_dir])
            if os.path.exists(region_dir):
                cmd.extend(["--region_dir", region_dir])

        _exec_command(cmd)
        vol.commit()
        print(f"[Modal] Baseline [{tgt}] checkpoint saved & committed to volume: {save_dir}")
        trained.append(save_dir)

    return trained


# ==============================================================================
# Remote Functions: Stage 1B Dual-LSTM Temporal JEPA Training
# ==============================================================================

@app.function(
    image=cuda_jax_image,
    gpu=DEFAULT_GPU,
    timeout=DEFAULT_TIMEOUT_SEC,
    memory=DEFAULT_MEMORY_MB,
    volumes={REMOTE_VOL_PATH: vol},
)
def train_temporal(
    spatial_ckpt: str = f"{REMOTE_VOL_PATH}/checkpoints/m2_spatial_jepa/best.pkl",
    data_dir: str = f"{REMOTE_VOL_PATH}/data",
    save_dir: str = f"{REMOTE_VOL_PATH}/checkpoints/m1_full_temporal",
    phase1_epochs: int = 15,
    phase2_epochs: int = 25,
    batch_size: int = 8,
    clip_len: int = 16,
    k_steps: int = 4,
    lr: float = 3e-4,
    omega: float = 0.2,
    log_every: int = 25,
    smoke_test: bool = False,
) -> str:
    """Train Dual-LSTM Temporal Integrator (Phase 1 frozen spatial, Phase 2 anchored fine-tuning)."""
    os.makedirs(save_dir, exist_ok=True)
    cmd = [
        "python",
        "-u",
        "train/train_temporal_jepa.py",
        "--save_dir",
        save_dir,
        "--phase1_epochs",
        str(phase1_epochs),
        "--phase2_epochs",
        str(phase2_epochs),
        "--batch_size",
        str(batch_size),
        "--clip_len",
        str(clip_len),
        "--k_steps",
        str(k_steps),
        "--lr",
        str(lr),
        "--omega",
        str(omega),
        "--log_every",
        str(log_every),
        "--as_uint8",
        "--drop_last",
    ]
    if smoke_test:
        cmd.append("--smoke_test")
    else:
        if os.path.exists(spatial_ckpt):
            cmd.extend(["--spatial_ckpt", spatial_ckpt])
        if os.path.exists(data_dir):
            cmd.extend(["--data_dir", data_dir])

    _exec_command(cmd)
    vol.commit()
    print(f"\n[Modal] Committed Stage 1B temporal checkpoints to volume: {save_dir}")
    return save_dir


# ==============================================================================
# Remote Functions: Representation & Dynamics Frozen Probes Suite
# ==============================================================================

@app.function(
    image=cuda_jax_image,
    gpu=DEFAULT_GPU,
    timeout=DEFAULT_TIMEOUT_SEC,
    memory=DEFAULT_MEMORY_MB,
    volumes={REMOTE_VOL_PATH: vol},
)
def run_eval(
    data_dir: str = f"{REMOTE_VOL_PATH}/data",
    checkpoints_dir: str = f"{REMOTE_VOL_PATH}/checkpoints",
    results_dir: str = f"{REMOTE_VOL_PATH}/results",
    smoke_test: bool = False,
) -> dict:
    """Evaluate frozen linear and non-linear probes across all model variants."""
    os.makedirs(results_dir, exist_ok=True)

    models_config = [
        ("m2_static", f"{checkpoints_dir}/m2_spatial_jepa/best.pkl", None),
        ("m1_full", f"{checkpoints_dir}/m1_full_temporal/spatial_finetuned_best.pkl", f"{checkpoints_dir}/m1_full_temporal/temporal_best.pkl"),
        ("b1_lod0", f"{checkpoints_dir}/baselines/b1_lod0/best.pkl", None),
        ("b2_lod2", f"{checkpoints_dir}/baselines/b2_lod2/best.pkl", None),
        ("b3_mae", f"{checkpoints_dir}/baselines/b3_mae/best.pkl", None),
    ]

    summaries = {}
    for model_type, ckpt, temp_ckpt in models_config:
        out_json = f"{results_dir}/probe_{model_type}.json"
        print(f"\n=== Running Probe Suite on [{model_type}] ===")
        cmd = [
            "python",
            "-u",
            "eval/probe_suite.py",
            "--model_type",
            model_type,
            "--out_json",
            out_json,
        ]
        if smoke_test:
            cmd.append("--smoke_test")
        else:
            if ckpt and os.path.exists(ckpt):
                cmd.extend(["--ckpt", ckpt])
            if temp_ckpt and os.path.exists(temp_ckpt):
                cmd.extend(["--temporal_ckpt", temp_ckpt])
            if os.path.exists(data_dir):
                cmd.extend(["--data_dir", data_dir])

        _exec_command(cmd)
        vol.commit()
        print(f"[Modal] Probe metrics for [{model_type}] saved to {out_json}")
        summaries[model_type] = out_json

    vol.commit()
    return summaries


# ==============================================================================
# Remote Functions: Frozen-Backbone PPO Policy Navigation Training
# ==============================================================================

@app.function(
    image=cuda_jax_image,
    gpu=DEFAULT_GPU,
    timeout=DEFAULT_TIMEOUT_SEC,
    memory=DEFAULT_MEMORY_MB,
    volumes={REMOTE_VOL_PATH: vol},
)
def train_ppo(
    model_type: str = "m2_static",
    ckpt: str = f"{REMOTE_VOL_PATH}/checkpoints/m2_spatial_jepa/best.pkl",
    temporal_ckpt: Optional[str] = None,
    courses_dir: str = f"{REMOTE_VOL_PATH}/data/courses",
    save_dir: str = f"{REMOTE_VOL_PATH}/checkpoints/ppo_nav_m2",
    total_steps: int = 5000,
    bc_epochs: int = 0,
    smoke_test: bool = False,
) -> str:
    """Train ActorCriticHead navigation policy on top of frozen perception representations."""
    os.makedirs(save_dir, exist_ok=True)
    cmd = [
        "python",
        "-u",
        "eval/train_ppo_nav.py",
        "--model_type",
        model_type,
        "--save_dir",
        save_dir,
        "--total_steps",
        str(total_steps),
        "--bc_epochs",
        str(bc_epochs),
        "--device",
        "gpu",
    ]
    if smoke_test:
        cmd.append("--smoke_test")
    else:
        if ckpt and os.path.exists(ckpt):
            cmd.extend(["--ckpt", ckpt])
        if temporal_ckpt and os.path.exists(temporal_ckpt):
            cmd.extend(["--temporal_ckpt", temporal_ckpt])
        if os.path.exists(courses_dir):
            cmd.extend(["--courses_dir", courses_dir])

    _exec_command(cmd)
    vol.commit()
    print(f"\n[Modal] PPO navigation policy checkpoint committed to: {save_dir}")
    return save_dir


# ==============================================================================
# Remote Functions: Zero-Shot Navigation Evaluation & Compute Profiling
# ==============================================================================

@app.function(
    image=cuda_jax_image,
    gpu=DEFAULT_GPU,
    timeout=DEFAULT_TIMEOUT_SEC,
    memory=DEFAULT_MEMORY_MB,
    volumes={REMOTE_VOL_PATH: vol},
)
def eval_nav(
    model_type: str = "m2_static",
    backbone_ckpt: str = f"{REMOTE_VOL_PATH}/checkpoints/m2_spatial_jepa/best.pkl",
    policy_ckpt: str = f"{REMOTE_VOL_PATH}/checkpoints/ppo_nav_m2/policy_best.pkl",
    temporal_ckpt: Optional[str] = None,
    courses_dir: str = f"{REMOTE_VOL_PATH}/data/courses",
    out_json: str = f"{REMOTE_VOL_PATH}/results/nav_m2.json",
    num_episodes: int = 5,
    max_steps: int = 250,
    use_gating: bool = False,
    smoke_test: bool = False,
) -> str:
    """Evaluate trained policy on navigation courses and profile compute / latency."""
    parent = os.path.dirname(os.path.abspath(out_json))
    os.makedirs(parent, exist_ok=True)

    cmd = [
        "python",
        "-u",
        "eval/eval_navigation.py",
        "--model_type",
        model_type,
        "--out_json",
        out_json,
        "--num_episodes",
        str(num_episodes),
        "--max_steps",
        str(max_steps),
    ]
    if use_gating:
        cmd.append("--use_gating")
    if smoke_test:
        cmd.append("--smoke_test")
    else:
        if backbone_ckpt and os.path.exists(backbone_ckpt):
            cmd.extend(["--backbone_ckpt", backbone_ckpt])
        if policy_ckpt and os.path.exists(policy_ckpt):
            cmd.extend(["--policy_ckpt", policy_ckpt])
        if temporal_ckpt and os.path.exists(temporal_ckpt):
            cmd.extend(["--temporal_ckpt", temporal_ckpt])
        if os.path.exists(courses_dir):
            cmd.extend(["--courses_dir", courses_dir])

    _exec_command(cmd)
    vol.commit()
    print(f"\n[Modal] Navigation evaluation metrics committed to: {out_json}")
    return out_json


# ==============================================================================
# Remote Functions: Springer LNEE Vector Figure Generation
# ==============================================================================

@app.function(
    image=cuda_jax_image,
    gpu=DEFAULT_GPU,
    timeout=1800,
    memory=DEFAULT_MEMORY_MB,
    volumes={REMOTE_VOL_PATH: vol},
)
def generate_figures(
    metrics_dir: str = f"{REMOTE_VOL_PATH}/checkpoints",
    ckpt: str = f"{REMOTE_VOL_PATH}/checkpoints/m2_spatial_jepa/best.pkl",
    out_dir: str = f"{REMOTE_VOL_PATH}/paper_figs",
    demo: bool = False,
) -> str:
    """Generate all 4 Springer Nature LNEE compliant publication vector figures."""
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        "python",
        "-u",
        "paper_figs/make_figures.py",
        "--out_dir",
        out_dir,
    ]
    if demo:
        cmd.append("--demo")
    else:
        if os.path.exists(metrics_dir):
            cmd.extend(["--metrics_dir", metrics_dir])
        if os.path.exists(ckpt):
            cmd.extend(["--ckpt", ckpt])

    _exec_command(cmd)
    vol.commit()
    print(f"\n[Modal] Springer Nature LNEE vector figures committed to: {out_dir}")
    return out_dir


# ==============================================================================
# Remote Functions: End-to-End Orchestrator (run_all)
# ==============================================================================

@app.function(
    image=cuda_jax_image,
    gpu=DEFAULT_GPU,
    timeout=DEFAULT_TIMEOUT_SEC,
    memory=DEFAULT_MEMORY_MB,
    volumes={REMOTE_VOL_PATH: vol},
)
def run_all(smoke_test: bool = False) -> dict:
    """Run full end-to-end cloud pipeline: Stage 1A -> Baselines -> Stage 1B -> Probes -> PPO -> Figures."""
    print("==================================================================")
    print("Starting End-to-End Cloud GPU Pipeline on Modal")
    print(f"Target GPU: {DEFAULT_GPU} | Smoke Test: {smoke_test}")
    print("==================================================================")

    # 1. Hardware verification
    verify_gpu.local()

    # 2. Stage 1A Spatial JEPA
    spatial_dir = train_spatial.local(smoke_test=smoke_test)

    # 3. Baselines
    baseline_dirs = train_baselines.local(model="all", smoke_test=smoke_test)

    # 4. Stage 1B Temporal JEPA
    temporal_dir = train_temporal.local(smoke_test=smoke_test)

    # 5. Probes Suite
    probe_results = run_eval.local(smoke_test=smoke_test)

    # 6. PPO Policy Navigation Training
    ppo_dir = train_ppo.local(smoke_test=smoke_test)

    # 7. Navigation Evaluation
    nav_result = eval_nav.local(smoke_test=smoke_test)

    # 8. Springer Figures
    figs_dir = generate_figures.local(demo=smoke_test)

    vol.commit()
    print("\n==================================================================")
    print("End-to-End Pipeline Completed Successfully!")
    print(f"  Checkpoints: {spatial_dir}, {temporal_dir}")
    print(f"  Baselines  : {baseline_dirs}")
    print(f"  Probes     : {probe_results}")
    print(f"  PPO Policy : {ppo_dir}")
    print(f"  Navigation : {nav_result}")
    print(f"  Figures    : {figs_dir}")
    print("==================================================================")
    return {
        "spatial_dir": spatial_dir,
        "temporal_dir": temporal_dir,
        "baseline_dirs": baseline_dirs,
        "probe_results": probe_results,
        "ppo_dir": ppo_dir,
        "nav_result": nav_result,
        "figs_dir": figs_dir,
    }


# ==============================================================================
# Local Entrypoint for CLI Invocations
# ==============================================================================

@app.local_entrypoint()
def main(
    action: str = "verify",
    epochs: int = 80,
    batch_size: int = 32,
    model: str = "all",
    log_every: int = 25,
    no_vicreg: bool = False,
    smoke_test: bool = False,
):
    """Local CLI driver for Modal executions.

    Usage Examples:
      uv run modal run modal_app.py --action verify
      uv run modal run modal_app.py --action train-spatial --epochs 80 --log-every 25
      uv run modal run modal_app.py --action train-baselines --model all --log-every 25
      uv run modal run modal_app.py --action train-temporal --log-every 25
      uv run modal run modal_app.py --action eval-probes
      uv run modal run modal_app.py --action train-ppo
      uv run modal run modal_app.py --action eval-nav
      uv run modal run modal_app.py --action make-figures
      uv run modal run modal_app.py --action run-all --smoke-test
    """
    act = action.lower().strip()
    if act in ("verify", "check", "gpu"):
        verify_gpu.remote()
    elif act in ("train-spatial", "spatial", "m2"):
        save_dir = f"{REMOTE_VOL_PATH}/checkpoints/m2_no_vicreg" if no_vicreg else f"{REMOTE_VOL_PATH}/checkpoints/m2_spatial_jepa"
        train_spatial.remote(
            epochs=epochs,
            batch_size=batch_size,
            log_every=log_every,
            no_vicreg=no_vicreg,
            save_dir=save_dir,
            smoke_test=smoke_test,
        )
    elif act in ("train-baselines", "baselines"):
        train_baselines.remote(model=model, epochs=epochs, batch_size=batch_size, log_every=log_every, smoke_test=smoke_test)
    elif act in ("train-temporal", "temporal", "m1"):
        train_temporal.remote(batch_size=batch_size, log_every=log_every, smoke_test=smoke_test)
    elif act in ("eval-probes", "probes", "eval"):
        run_eval.remote(smoke_test=smoke_test)
    elif act in ("train-ppo", "ppo"):
        train_ppo.remote(smoke_test=smoke_test)
    elif act in ("eval-nav", "nav"):
        eval_nav.remote(smoke_test=smoke_test)
    elif act in ("make-figures", "figures", "figs"):
        generate_figures.remote(demo=smoke_test)
    elif act in ("run-all", "all", "smoke-test"):
        run_all.remote(smoke_test=smoke_test or (act == "smoke-test"))
    else:
        print(f"Unknown action: '{action}'. Valid actions:")
        print("  verify, train-spatial, train-baselines, train-temporal, eval-probes,")
        print("  train-ppo, eval-nav, make-figures, run-all, smoke-test")

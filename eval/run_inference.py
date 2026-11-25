"""
Real-Time Single-Inference Playback Runner for Squared Sense World Model.

Loads a trained JAX/Flax NNX policy and perception backbone, executes pathfinding
rollouts on real Minecraft terrain (or debug bridge courses), and streams live
observations with rich TELM v1 inference telemetry directly into the shared memory
IPC buffer (squared_sense_mc.bin) for interactive 3D WebGL inspection.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from train.gpu_env import setup_gpu_environment

# Ensure non-greedy allocation and VRAM limits are set prior to XLA initialization
setup_gpu_environment()

from typing import Any, Dict, List, Optional, Tuple

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from eval.sim_env import VoxelNavEnv, make_debug_bridge_course
from eval.train_ppo_nav import ActorCriticHead, extract_step_feature
from env.minecraft_env import MinecraftNavEnv
from env.ipc_emitter import MinecraftIpcEmitter
from env.ipc_telemetry import InferenceTelemetry
from models.baselines import Generative3DMAE, SingleLODJEPA
from models.temporal_lstm import DualLSTMTemporalIntegrator
from models.transformer import SpatialJEPA
from train.checkpoint import load_checkpoint

ACTION_NAMES = [
    "FORWARD",
    "FORWARD_LEFT",
    "FORWARD_RIGHT",
    "STRAFE_LEFT",
    "STRAFE_RIGHT",
    "BACKWARD",
    "TURN_LEFT",
    "TURN_RIGHT",
    "STEP_UP",
    "PITCH_UP",
    "PITCH_DOWN",
    "IDLE",
]


def check_keyboard_input() -> Optional[str]:
    """Non-blocking keyboard check on Windows."""
    try:
        import msvcrt
        if msvcrt.kbhit():
            ch = msvcrt.getch()
            if ch in (b' ', b'p', b'P'):
                return "SPACE"
            elif ch in (b'n', b'N'):
                return "NEXT"
            elif ch in (b'r', b'R'):
                return "RESET"
            elif ch in (b'q', b'Q', b'\x1b'):
                return "QUIT"
    except ImportError:
        pass
    return None


def run_inference_episode(
    env: Any,
    backbone: Any,
    policy: ActorCriticHead,
    emitter: MinecraftIpcEmitter,
    model_type: str = "m2_static",
    temporal_model: Optional[DualLSTMTemporalIntegrator] = None,
    checkpoint_meta: Optional[Dict[str, Any]] = None,
    fps: float = 10.0,
    greedy: bool = True,
    step_pause: bool = False,
    seed: int = 42,
) -> Dict[str, Any]:
    """Executes a single inference rollout while streaming telemetry to MMF."""
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)

    obs = env.reset()
    temporal_hidden = None

    player_pos = getattr(env, "player_pos", getattr(env, "pos", np.array([0, 64, 0])))
    target_pos = getattr(env, "target_pos", getattr(env, "active_waypoint", (0, 64, 10)))
    all_targets_init = list(getattr(env, "targets", [target_pos]))
    optimal_path = getattr(env, "optimal_path", [])
    l_ref = getattr(env, "shortest_path_len", 10.0)

    print("\n=======================================================")
    print(f"Episode Started! Spawn: {tuple(player_pos)} -> Target: {tuple(target_pos)} ({len(all_targets_init)} waypoint(s))")
    if len(all_targets_init) > 1:
        print(f"Waypoints: {[tuple(int(v) for v in t) for t in all_targets_init]}")
    print(f"Reference Path Length: {l_ref:.1f} blocks | FPS: {fps}")
    print("Streaming live frames to 3D WebGL visualizer...")
    print("Controls: [SPACE]=Pause/Resume | [N]=Step | [R]=Reset | [Q]=Quit")
    print("=======================================================\n")

    breadcrumbs: List[Tuple[int, int, int]] = [tuple(player_pos)]
    bfs_tuples: List[Tuple[int, int, int]] = [tuple(p) for p in optimal_path]

    step_delay = 1.0 / fps if fps > 0 else 0.0
    ep_return = 0.0
    step_count = 0
    paused = step_pause
    done = False
    last_info: Dict[str, Any] = {}

    def _emit_current_frame(
        curr_p: Any,
        reward_val: float,
        ret_val: float,
        step_idx: int,
        v_val: float,
        probs: List[float],
        chosen_act: int,
    ) -> None:
        if not hasattr(env, "extractor"):
            return
        tgt = tuple(getattr(env, "target_pos", getattr(env, "active_waypoint", target_pos)))
        curr_bfs = [tuple(p) for p in getattr(env, "optimal_path", bfs_tuples)]
        all_targets = list(getattr(env, "targets", [tgt]))
        t_idx = int(getattr(env, "target_idx", 0))
        t_reached = int(getattr(env, "targets_reached", 0))
        occluded = bool(getattr(env, "compass_occluded", False))

        dist_val = float(np.linalg.norm(np.array(curr_p, dtype=float) - np.array(tgt, dtype=float)))
        w_dx = float(tgt[0] - curr_p[0])
        w_dy = float(tgt[1] - curr_p[1])
        w_dz = float(tgt[2] - curr_p[2])
        bc_rel = [(p[0] - curr_p[0], p[1] - curr_p[1], p[2] - curr_p[2]) for p in breadcrumbs]
        bfs_rel = [(p[0] - curr_p[0], p[1] - curr_p[1], p[2] - curr_p[2]) for p in curr_bfs[:64]]
        wps_rel = [
            (int(round(t[0] - curr_p[0])), int(round(t[1] - curr_p[1])), int(round(t[2] - curr_p[2])))
            for t in all_targets[:16]
        ]
        telem = InferenceTelemetry.pack(
            target_pos=(tgt[0], tgt[1], tgt[2]),
            world_delta=(w_dx, w_dy, w_dz),
            target_dist=dist_val,
            step_reward=float(reward_val),
            episode_return=float(ret_val),
            step_count=step_idx,
            critic_value=v_val,
            action_probabilities=probs,
            chosen_action=chosen_act,
            l_bfs=int(round(l_ref)),
            breadcrumbs=bc_rel,
            optimal_path=bfs_rel,
            checkpoint_loss=float(checkpoint_meta.get("loss", 0.0) if checkpoint_meta else 0.0),
            waypoints=wps_rel,
            active_target_idx=t_idx,
            total_targets=len(all_targets),
            targets_reached=t_reached,
            compass_occluded=occluded,
        )
        c_str = getattr(env, "cardinal", "SOUTH")
        f_idx = getattr(env, "foot_index", 6)
        raw_v = env.extractor.extract_observation(tuple(curr_p), c_str, foot_index=f_idx)
        emitter.emit_observation(raw_v, game_tick=step_idx, telemetry_bytes=telem)

    # Emit initial spawn + target frame immediately (so --pause or slow JIT still shows target)
    _emit_current_frame(player_pos, 0.0, 0.0, 0, 0.0, [0.0] * 12, 11)

    while not done:
        # Check non-blocking keyboard controls
        key_cmd = check_keyboard_input()
        if key_cmd == "SPACE":
            paused = not paused
            print(f"  >>> [{ 'PAUSED' if paused else 'RESUMED' }]")
        elif key_cmd == "QUIT":
            print("  >>> User requested quit.")
            break
        elif key_cmd == "RESET":
            print("  >>> Manual episode reset.")
            break

        if paused:
            if key_cmd == "NEXT":
                pass  # Step single frame
            else:
                time.sleep(0.05)
                continue

        step_t0 = time.perf_counter()

        # 1. Feature Extraction from frozen backbone
        feat, temporal_hidden = extract_step_feature(
            backbone=backbone,
            model_type=model_type,
            shells=obs["shells"],
            state=obs["state"],
            temporal_model=temporal_model,
            temporal_hidden=temporal_hidden,
        )

        # 2. Policy forward pass
        goal_v = obs["goal_vec"]
        logits, values = policy(feat, goal_v)
        action_probs = np.array(jax.nn.softmax(logits[0]))
        v_est = float(values[0])

        # Action masking if available
        if "action_mask" in obs:
            mask = obs["action_mask"]
            masked_probs = action_probs * mask
            s = masked_probs.sum()
            if s > 0:
                action_probs = masked_probs / s

        if greedy:
            action = int(np.argmax(action_probs))
        else:
            key, subkey = jax.random.split(key)
            action = int(jax.random.categorical(subkey, jnp.asarray(np.log(action_probs + 1e-8))))

        # 3. Environment Step
        next_obs, reward, done, last_info = env.step(action)
        ep_return += reward
        step_count += 1

        curr_player = getattr(env, "player_pos", getattr(env, "pos", player_pos))
        breadcrumbs.append(tuple(curr_player))
        if len(breadcrumbs) > 64:
            breadcrumbs.pop(0)

        # 4. Pack and Emit Inference Telemetry
        target_tuple = tuple(getattr(env, "target_pos", getattr(env, "active_waypoint", target_pos)))
        d_curr = float(np.linalg.norm(np.array(curr_player) - np.array(target_tuple)))
        _emit_current_frame(curr_player, float(reward), float(ep_return), step_count, v_est, list(action_probs), action)

        act_name = ACTION_NAMES[action] if action < len(ACTION_NAMES) else f"ACT_{action}"
        print(
            f"  [Step {step_count:02d}] Pos: {tuple(curr_player)} | "
            f"Dist: {d_curr:.1f}m | Action: {act_name:<12} (p={action_probs[action]:.2f}) | "
            f"Critic V(s): {v_est:+.2f} | Rew: {reward:+.2f}"
        )

        obs = next_obs

        # Rate limiter
        elapsed = time.perf_counter() - step_t0
        if step_delay > elapsed:
            time.sleep(step_delay - elapsed)

    success = bool(last_info.get("success", False))
    print(f"\nEpisode Finished! Success: {success} | Total Return: {ep_return:.2f} | Steps: {step_count}")
    return {"success": success, "return": ep_return, "steps": step_count}


def main():
    parser = argparse.ArgumentParser(description="Squared Sense World Model — Real-Time Inference Runner")
    parser.add_argument("--region_dir", type=str, default="data/baked_regions", help="Path to baked Minecraft region memmaps")
    parser.add_argument("--model_type", type=str, default="m2_static", choices=["m1_full", "m2_static", "b1_lod0", "b2_lod2", "b3_mae"])
    parser.add_argument("--ckpt", type=str, default=None, help="Path to trained backbone checkpoint (.pkl)")
    parser.add_argument("--policy_ckpt", type=str, default="checkpoints/ppo_nav/policy_best.pkl", help="Path to trained ActorCriticHead (.pkl)")
    parser.add_argument("--temporal_ckpt", type=str, default=None, help="Path to trained DualLSTM checkpoint (.pkl)")
    parser.add_argument("--episodes", type=int, default=5, help="Number of inference episodes")
    parser.add_argument("--fps", type=float, default=10.0, help="Stepping speed in FPS (0 for unthrottled)")
    parser.add_argument("--pause", action="store_true", help="Start episode in paused state")
    parser.add_argument("--greedy", action="store_true", default=True, help="Take deterministic argmax action")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "gpu", "cpu"], help="Target acceleration device (auto, gpu, cpu)")
    parser.add_argument("--task", type=str, default="rl1", choices=["rl1", "rl2"], help="Navigation task mode: rl1 (single target) or rl2 (5-target chained)")
    parser.add_argument("--num_targets", type=int, default=None, help="Number of chained waypoints per episode (defaults: 1 for rl1, 5 for rl2)")
    parser.add_argument("--max_steps", type=int, default=None, help="Max steps per episode (defaults: 150 for rl1, 600 for rl2)")
    parser.add_argument("--compass_occlusion", type=float, default=None, help="Compass burst occlusion probability (defaults: 0.0 for rl1, 0.5 for rl2)")
    parser.add_argument("--occlusion_burst_len", type=int, default=6, help="Compass occlusion burst length in ticks")

    args = parser.parse_args()

    # Configure GPU memory and platform
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if args.device != "auto":
        jax.config.update("jax_platform_name", args.device)

    backend = jax.default_backend()
    devices = jax.devices()
    print(f"\n=======================================================")
    print(f"Squared Sense World Model — Real-Time Inference Runner")
    print(f"JAX Backend: {backend.upper()} | Devices: {devices}")
    print(f"=======================================================\n")

    rng = nnx.Rngs(args.seed)

    # 1. Instantiate Backbone
    temporal_model: Optional[DualLSTMTemporalIntegrator] = None
    if args.model_type == "m2_static":
        backbone = SpatialJEPA(num_shells=3, rngs=rng)
        feat_dim = 256
    elif args.model_type == "m1_full":
        backbone = SpatialJEPA(num_shells=3, rngs=rng)
        temporal_model = DualLSTMTemporalIntegrator(rngs=rng)
        feat_dim = 448
    elif args.model_type == "b1_lod0":
        backbone = SingleLODJEPA(lod_index=0, rngs=rng)
        feat_dim = 256
    elif args.model_type == "b2_lod2":
        backbone = SingleLODJEPA(lod_index=2, rngs=rng)
        feat_dim = 256
    elif args.model_type == "b3_mae":
        backbone = Generative3DMAE(rngs=rng)
        feat_dim = 256
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    if args.ckpt and os.path.exists(args.ckpt):
        print(f"Loading backbone from {args.ckpt}...")
        load_checkpoint(args.ckpt, backbone)

    if temporal_model and args.temporal_ckpt and os.path.exists(args.temporal_ckpt):
        print(f"Loading temporal model from {args.temporal_ckpt}...")
        load_checkpoint(args.temporal_ckpt, temporal_model)

    # 2. Instantiate Policy
    policy = ActorCriticHead(feat_dim=feat_dim, num_actions=12, rngs=rng)
    checkpoint_meta = {}
    if args.policy_ckpt and os.path.exists(args.policy_ckpt):
        print(f"Loading policy from {args.policy_ckpt}...")
        checkpoint_meta = load_checkpoint(args.policy_ckpt, policy)

    # 3. Instantiate Environment
    num_targets = args.num_targets if args.num_targets is not None else (5 if args.task == "rl2" else 1)
    max_steps = args.max_steps if args.max_steps is not None else (600 if args.task == "rl2" else 150)
    compass_occlusion = args.compass_occlusion if args.compass_occlusion is not None else (0.5 if args.task == "rl2" else 0.0)

    if os.path.exists(args.region_dir):
        print(f"Loading MinecraftNavEnv from {args.region_dir} (task={args.task}, targets={num_targets}, max_steps={max_steps}, occlusion={compass_occlusion})...")
        env = MinecraftNavEnv(
            region_dir=args.region_dir,
            max_steps=max_steps,
            num_targets=num_targets,
            compass_occlusion_prob=compass_occlusion,
            occlusion_burst_len=args.occlusion_burst_len,
        )
    else:
        print(f"Region dir '{args.region_dir}' not found. Using debug bridge course...")
        vol, start, wps = make_debug_bridge_course(seed=args.seed)
        env = VoxelNavEnv(volume=vol, start=start, waypoints=wps, max_steps=max_steps)

    # 4. IPC Emitter
    emitter = MinecraftIpcEmitter()

    # 5. Run Episodes
    success_count = 0
    for ep in range(args.episodes):
        res = run_inference_episode(
            env=env,
            backbone=backbone,
            policy=policy,
            emitter=emitter,
            model_type=args.model_type,
            temporal_model=temporal_model,
            checkpoint_meta=checkpoint_meta,
            fps=args.fps,
            greedy=args.greedy,
            step_pause=args.pause,
            seed=args.seed + ep,
        )
        if res["success"]:
            success_count += 1

    print(f"\nInference complete! Success Rate: {success_count / args.episodes * 100:.1f}%")
    emitter.close()


if __name__ == "__main__":
    main()

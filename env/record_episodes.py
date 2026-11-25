"""
Trajectory Recording Pipeline for Squared Sense JEPA Pretraining.
Steps through MinecraftNavEnv (or VoxelNavEnv fallback) and saves episode archives
in the exact Option B .npz format expected by data/dataset.py:
  - shells: (T, 3, 16, 16, 16, 16) uint8 [0..255]
  - state : (T, 12) float32
  - action: (T,) int32
  - meta  : (T, 6) float32 [player_x, player_y, player_z, goal_dx, goal_dy, goal_dz]
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

import numpy as np

# Ensure root directory is on sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from env.curriculum import DistanceCurriculumScheduler
from env.minecraft_env import MinecraftNavEnv


def select_heuristic_action(goal_vec: np.ndarray, action_mask: Optional[np.ndarray] = None) -> int:
    """
    Selects a greedy movement action that points closest to goal_vec among valid actions.
    Ego coordinates: +Z is forward, +X is right, +Y is up.
    Prioritizes turning (Action 6 Turn-Left, Action 7 Turn-Right) when target is outside forward cone.
    """
    gx, gy, gz = float(goal_vec[0]), float(goal_vec[1]), float(goal_vec[2])
    h_dist = float(np.hypot(gx, gz))
    if h_dist < 0.5 and abs(gy) < 0.5:
        return 11  # Stay / Idle

    # Check forward cone: target is in front if gz > 0 and abs(gx) <= gz
    not_in_forward_cone = (gz <= 0.0) or (abs(gx) > gz)

    if not_in_forward_cone:
        # Prioritize turning to face the goal
        if gx < -0.2:
            turn_act = 6  # Turn-Left (-45°)
        elif gx > 0.2:
            turn_act = 7  # Turn-Right (+45°)
        else:
            # Directly behind (gz <= 0 and abs(gx) <= 0.2)
            turn_act = 7  # Turn-Right (+45°) to turn around

        if action_mask is None or action_mask[turn_act]:
            return turn_act

    # Target is in forward cone (or turn action was masked)
    # Check pitch if vertical offset is large and target is relatively close
    if h_dist < 4.0:
        if gy >= 2.0 and (action_mask is None or action_mask[9]):
            return 9  # Pitch-Up
        elif gy <= -2.0 and (action_mask is None or action_mask[10]):
            return 10  # Pitch-Down

    # If step-up needed
    if gy >= 0.8:
        if action_mask is None or action_mask[8]:
            return 8  # Step-Up-Forward
        elif action_mask is None or action_mask[1]:
            return 1  # Forward+Jump

    actions_to_dir = {
        0: np.array([0.0, 0.0, 1.0]),     # Forward (+Z)
        1: np.array([-0.707, 0.0, 0.707]),# Forward-Left
        2: np.array([0.707, 0.0, 0.707]), # Forward-Right
        3: np.array([-1.0, 0.0, 0.0]),    # Strafe-Left (-X)
        4: np.array([1.0, 0.0, 0.0]),     # Strafe-Right (+X)
        5: np.array([0.0, 0.0, -1.0]),    # Backward (-Z)
        8: np.array([0.0, 1.0, 1.0]),     # Step-Up-Forward (+Z, +Y)
    }

    norm = np.linalg.norm([gx, gy, gz])
    target_dir = np.array([gx, gy, gz]) / (norm + 1e-6)
    best_act = 0
    best_dot = -float("inf")

    for act, dir_vec in actions_to_dir.items():
        if action_mask is not None and not action_mask[act]:
            continue
        dot = float(np.dot(dir_vec, target_dir))
        if dot > best_dot:
            best_dot = dot
            best_act = act

    return best_act


def record_episodes(
    region_dir: Optional[str] = None,
    out_dir: str = "data/episodes",
    num_episodes: int = 10,
    ep_len: int = 50,
    seed: int = 42,
    policy: str = "random",
    curriculum: Optional[DistanceCurriculumScheduler] = None,
) -> List[str]:
    """
    Rolls out trajectories in the environment and serializes them as Option B .npz files.

    Args:
        region_dir: Directory containing baked .mmap and .spawns.npz files.
                    If None or nonexistent, falls back to eval.sim_env.make_env().
        out_dir: Destination folder for output episode archives.
        num_episodes: Number of episodes to record.
        ep_len: Maximum step length per episode.
        seed: Base random seed for reproducibility.
        policy: Exploration policy ('random' or 'heuristic').
        curriculum: Optional curriculum scheduler for target sampling distance.
        fmt: File format ('npz' or 'npy', default 'npz').

    Returns:
        List of generated file paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    created_files: List[str] = []

    # Initialize environment
    if region_dir is not None and os.path.exists(region_dir):
        env = MinecraftNavEnv(region_dir=region_dir, curriculum=curriculum)
        use_real_mc = True
    else:
        from eval.sim_env import make_env
        env = make_env(region_dir=None, seed=seed)
        use_real_mc = False

    rng = np.random.default_rng(seed)

    for ep in range(num_episodes):
        ep_seed = seed + ep
        if use_real_mc:
            obs = env.reset(seed=ep_seed)
        else:
            obs = env.reset()

        shells_buf: List[np.ndarray] = []
        state_buf: List[np.ndarray] = []
        action_buf: List[int] = []
        meta_buf: List[np.ndarray] = []

        for t in range(ep_len):
            # 1. Action selection
            action_mask = obs.get("action_mask", None)
            if policy == "heuristic" and "goal_vec" in obs:
                action = select_heuristic_action(obs["goal_vec"], action_mask)
            else:
                # Random valid action
                if action_mask is not None and np.any(action_mask):
                    valid_actions = np.where(action_mask)[0]
                    action = int(rng.choice(valid_actions))
                else:
                    action = int(rng.integers(0, 12))

            # 2. Step environment
            next_obs, reward, done, info = env.step(action)

            # 3. Buffer collection (convert float32 [0.0..1.0] shells to uint8 [0..255])
            raw_shells = obs["shells"]
            if raw_shells.dtype == np.uint8:
                shells_uint8 = raw_shells
            else:
                shells_uint8 = np.clip(raw_shells * 255.0 + 0.5, 0, 255).astype(np.uint8)

            shells_buf.append(shells_uint8)
            state_buf.append(np.asarray(obs["state"], dtype=np.float32))
            action_buf.append(int(action))

            # Meta vector: [player_x, player_y, player_z, goal_dx, goal_dy, goal_dz]
            if use_real_mc:
                px, py, pz = env.player_pos[0], env.player_pos[1], env.player_pos[2]
            else:
                px, py, pz = env.pos[0], env.pos[1], env.pos[2]

            if "goal_vec" in obs:
                gdx, gdy, gdz = obs["goal_vec"][0], obs["goal_vec"][1], obs["goal_vec"][2]
            else:
                gdx, gdy, gdz = 0.0, 0.0, 0.0

            meta_buf.append(np.array([px, py, pz, gdx, gdy, gdz], dtype=np.float32))

            obs = next_obs
            if done:
                break

        # Save compressed episode file
        out_path = os.path.join(out_dir, f"episode_{ep:04d}.npz")
        np.savez_compressed(
            out_path,
            shells=np.stack(shells_buf, axis=0),
            state=np.stack(state_buf, axis=0),
            action=np.array(action_buf, dtype=np.int32),
            meta=np.stack(meta_buf, axis=0),
        )
        created_files.append(out_path)

    return created_files


def main():
    parser = argparse.ArgumentParser(description="Record Minecraft survival episodes for JEPA pretraining.")
    parser.add_argument("--region_dir", type=str, default=None, help="Directory containing baked region memmaps.")
    parser.add_argument("--out_dir", type=str, default="data/episodes", help="Output directory for episode archives.")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to record.")
    parser.add_argument("--ep_len", type=int, default=50, help="Maximum steps per episode.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--policy", type=str, choices=["random", "heuristic"], default="random", help="Exploration policy.")

    args = parser.parse_args()
    files = record_episodes(
        region_dir=args.region_dir,
        out_dir=args.out_dir,
        num_episodes=args.num_episodes,
        ep_len=args.ep_len,
        seed=args.seed,
        policy=args.policy,
    )
    print(f"Recorded {len(files)} episodes to {args.out_dir}")


if __name__ == "__main__":
    main()


import os
import sys
import numpy as np
from collections import Counter

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.minecraft_env import MinecraftNavEnv
from env.record_episodes import select_heuristic_action
from eval.train_ppo_nav import select_bc_oracle_action

def main():
    region_dir = "data/baked_regions"
    if not os.path.exists(region_dir):
        print(f"Error: {region_dir} does not exist!")
        return

    print("=================================================================")
    print("STEP 2 VERIFICATION: POST-FIX DIAGNOSTIC SCRIPT")
    print("=================================================================")
    
    env = MinecraftNavEnv(region_dir=region_dir, max_steps=100)
    obs = env.reset(seed=42)
    
    print("\n--- 1. Coordinate & Reward Sign Check ---")
    player_pos = env.player_pos.copy()
    target_pos = env.target_pos.copy()
    yaw = env.yaw
    cardinal = env.cardinal
    goal_vec = obs["goal_vec"]
    
    print(f"Initial State:")
    print(f"  Player Pos:   {player_pos}")
    print(f"  Target Pos:   {target_pos}")
    print(f"  World Delta:  {(target_pos - player_pos).tolist()} (target - player)")
    print(f"  Yaw:          {yaw:.2f} deg, Cardinal: {cardinal}")
    print(f"  obs['goal_vec'] (ego_gx, ego_gy, ego_gz): {goal_vec.tolist()}")
    
    d_initial = float(np.linalg.norm(target_pos - player_pos))
    print(f"  Initial Euclidean Dist: {d_initial:.3f}")
    
    # Step Forward (Action 0)
    obs_fwd, rew_fwd, done_fwd, info_fwd = env.step(0)
    pos_fwd = env.player_pos.copy()
    d_fwd = float(np.linalg.norm(target_pos - pos_fwd))
    print(f"\nStep Forward (Action 0):")
    print(f"  New Pos:      {pos_fwd.tolist()}")
    print(f"  Pos Delta:    {(pos_fwd - player_pos).tolist()}")
    print(f"  Dist: {d_initial:.3f} -> {d_fwd:.3f} (delta = {d_fwd - d_initial:+.3f})")
    print(f"  Reward:       {rew_fwd:+.4f}")
    print(f"  Blocked:      {info_fwd.get('blocked', False)}")
    print(f"  New goal_vec: {obs_fwd['goal_vec'].tolist()}")

    # Reset again to seed=42 and step Backward (Action 5)
    env.reset(seed=42)
    obs_bwd, rew_bwd, done_bwd, info_bwd = env.step(5)
    pos_bwd = env.player_pos.copy()
    d_bwd = float(np.linalg.norm(target_pos - pos_bwd))
    print(f"\nStep Backward (Action 5):")
    print(f"  New Pos:      {pos_bwd.tolist()}")
    print(f"  Pos Delta:    {(pos_bwd - player_pos).tolist()}")
    print(f"  Dist: {d_initial:.3f} -> {d_bwd:.3f} (delta = {d_bwd - d_initial:+.3f})")
    print(f"  Reward:       {rew_bwd:+.4f}")
    print(f"  Blocked:      {info_bwd.get('blocked', False)}")
    print(f"  New goal_vec: {obs_bwd['goal_vec'].tolist()}")

    # Check Turn-Left (Action 6) and Turn-Right (Action 7) heading shaping reward
    env.reset(seed=42)
    obs_turn, rew_turn, done_turn, info_turn = env.step(7)  # Turn right (+45 deg)
    print(f"\nStep Turn-Right (Action 7):")
    print(f"  New Yaw:      {env.yaw:.2f} deg, Cardinal: {env.cardinal}")
    print(f"  Reward:       {rew_turn:+.4f} (includes heading shaping bonus!)")
    
    closer_fwd = d_fwd < d_initial
    closer_bwd = d_bwd < d_initial
    print(f"\nSign Verification:")
    print(f"  Forward got closer: {closer_fwd} | Reward: {rew_fwd:+.4f} (Reward > 0 if closer: {rew_fwd > 0 if closer_fwd else 'N/A'})")
    print(f"  Backward got closer: {closer_bwd} | Reward: {rew_bwd:+.4f}")
    print(f"  goal_vec[2] (ego_gz): {goal_vec[2]:+.3f}")

    # Oracle 20-episode check using select_bc_oracle_action
    print("\n--- 2. Oracle Sanity Check (20 Episodes via select_bc_oracle_action) ---")
    action_counts = Counter()
    successes = 0
    total_steps = 0
    total_blocked = 0
    episodes = 20
    
    ACTION_NAMES = {
        0: "Forward",
        1: "Forward+Jump",
        2: "Sprint+Forward",
        3: "Strafe-Left",
        4: "Strafe-Right",
        5: "Backward",
        6: "Turn-Left-45",
        7: "Turn-Right-45",
        8: "Step-Up-Fwd",
        9: "Pitch-Up",
        10: "Pitch-Down",
        11: "Idle",
    }

    for ep in range(episodes):
        obs_ep = env.reset(seed=1000 + ep)
        done = False
        ep_steps = 0
        ep_success = False
        while not done and ep_steps < 100:
            act = select_bc_oracle_action(env, obs_ep)
            action_counts[act] += 1
            obs_ep, r, done, info_step = env.step(act)
            if info_step.get("blocked", False):
                total_blocked += 1
            ep_steps += 1
            if info_step.get("success", False):
                ep_success = True
        if ep_success:
            successes += 1
        total_steps += ep_steps

    sr = (successes / episodes) * 100.0
    blocked_pct = (total_blocked / total_steps * 100.0) if total_steps > 0 else 0.0
    print(f"Oracle Success Rate: {successes}/{episodes} ({sr:.1f}%) across {total_steps} total steps")
    print(f"Blocked Moves: {total_blocked}/{total_steps} ({blocked_pct:.1f}%)")
    print("Action distribution:")
    for a in range(12):
        cnt = action_counts[a]
        pct = (cnt / total_steps * 100.0) if total_steps > 0 else 0.0
        print(f"  Action {a:2d} ({ACTION_NAMES[a]:14s}): {cnt:5d} ({pct:5.1f}%)")
        
    turn_actions = action_counts[6] + action_counts[7] + action_counts[9] + action_counts[10]
    print(f"Total Turning/Pitch Actions (6, 7, 9, 10): {turn_actions} ({(turn_actions/total_steps*100.0):.1f}%)")
    print("=================================================================\n")

if __name__ == "__main__":
    main()

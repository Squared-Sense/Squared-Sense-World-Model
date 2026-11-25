"""
High-Performance Minecraft 3D Voxel Navigation Environment.
Strictly implements the 16-channel contract (docs/CONTRACT.md and docs/EVAL_CONTRACT.md).
"""

from __future__ import annotations

import math
import os
import glob
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from data.schema import (
    Channel,
    COURSE_SHAPE,
    LOD_SPECS,
    MAX_POOL_CHANNELS,
    AVG_POOL_CHANNELS,
    SHELL_VOXEL_SHAPE,
    STATE_DIM
)
from env.palette import BlockPalette, get_palette
from env.lod_extractor import WorldVolume, LodExtractor, calculate_foot_index, GRID_SIZE
from env.physics_kernel import MovementPhysicsKernel, MoveOutcome
from env.spawn_indexer import SpawnSampler
from env.bfs_target import BoundedBfsTargetGenerator
from env.curriculum import DistanceCurriculumScheduler
from env.ipc_emitter import MinecraftIpcEmitter


CARDINAL_ORDER = ["SOUTH", "WEST", "NORTH", "EAST"]

CARDINAL_VECTORS = {
    "SOUTH": {"forward": (0, 1),  "right": (-1, 0)},
    "WEST":  {"forward": (-1, 0), "right": (0, -1)},
    "NORTH": {"forward": (0, -1), "right": (1, 0)},
    "EAST":  {"forward": (1, 0),  "right": (0, 1)}
}

CARDINAL_YAW_DEG = {
    "SOUTH": 0.0,
    "WEST": 90.0,
    "NORTH": 180.0,
    "EAST": 270.0
}


class MinecraftNavEnv:
    """
    3D Voxel Navigation Environment operating on baked Minecraft region memmaps.
    Produces identical observation shapes and dictionaries as VoxelNavEnv in eval/sim_env.py:
      - shells: (3, 16, 16, 16, 16) float32 in [0.0, 1.0] (LOD, X, Y, Z, C)
      - state: (12,) float32 proprioception
      - goal_vec: (3,) float32 relative 3D goal direction
      - action_mask: (12,) bool mask of safe/valid actions
    """

    NUM_ACTIONS = 12

    def __init__(
        self,
        region_dir: str,
        spawns_path: Optional[str] = None,
        palette_obj: Optional[BlockPalette] = None,
        curriculum: Optional[DistanceCurriculumScheduler] = None,
        max_steps: Optional[int] = None,
        goal_reach_radius: float = 1.5,
        mask_fatal: bool = True,
        render_mode: Optional[str] = None,
        num_targets: int = 1,
        compass_occlusion_prob: float = 0.0,
        occlusion_burst_len: int = 6,
        flash_compass_ticks: int = 0,
        skip_shells: bool = False,
    ) -> None:
        self.region_dir = region_dir
        self.palette = palette_obj or get_palette()
        self.world = WorldVolume(region_dir, palette_obj=self.palette)

        if spawns_path is None:
            candidates = sorted(glob.glob(os.path.join(region_dir, "*.spawns.npz")))
        else:
            candidates = [spawns_path] if os.path.exists(spawns_path) else []

        if not candidates:
            raise FileNotFoundError(
                f"No '*.spawns.npz' files found in region_dir '{region_dir}'. "
                f"Precompute traversable spawns first using: "
                f"uv run python -m env.spawn_indexer --region_dir \"{region_dir}\""
            )

        self.spawn_sampler = SpawnSampler.from_npz(candidates)
        if self.world.regions:
            rxs = [rx for rx, _ in self.world.regions.keys()]
            rzs = [rz for _, rz in self.world.regions.keys()]
            self.spawn_sampler.filter_interior_margin(
                min_x=min(rxs) * 512,
                max_x=(max(rxs) + 1) * 512 - 1,
                min_z=min(rzs) * 512,
                max_z=(max(rzs) + 1) * 512 - 1,
                margin=64,
            )

        self.extractor = LodExtractor(self.world, self.palette)
        self.kernel = MovementPhysicsKernel(self.world, self.palette)
        self.target_gen = BoundedBfsTargetGenerator(self.kernel)
        self.curriculum = curriculum or DistanceCurriculumScheduler()

        self.num_targets = max(1, int(num_targets))
        self.compass_occlusion_prob = float(compass_occlusion_prob)
        self.occlusion_burst_len = max(1, int(occlusion_burst_len))
        self.flash_compass_ticks = max(0, int(flash_compass_ticks))
        self.skip_shells = bool(skip_shells)
        if max_steps is None:
            self.max_steps = 600 if self.num_targets > 1 else 250
        else:
            self.max_steps = int(max_steps)

        self.goal_reach_radius = goal_reach_radius
        self.mask_fatal = mask_fatal
        self.render_mode = render_mode

        self.emitter: Optional[MinecraftIpcEmitter] = None
        if self.render_mode in ("human", "ipc"):
            self.emitter = MinecraftIpcEmitter()

        # State tracking
        self.player_pos = np.array([0, 64, 0], dtype=int)
        self.prev_pos = np.copy(self.player_pos)
        self.target_pos = np.array([0, 64, 10], dtype=int)
        self.targets: List[np.ndarray] = [self.target_pos]
        self.target_paths: List[List[Tuple[int, int, int]]] = []
        self.target_idx = 0
        self.targets_reached = 0
        self._target_reach_steps: List[int] = []
        self.optimal_path: List[Tuple[int, int, int]] = []
        self.shortest_path_len = 10.0

        self.compass_occluded = False
        self._occlusion_phase = 0
        self._occlusion_rng = np.random.default_rng()

        self.cardinal = "SOUTH"
        self.yaw = 0.0
        self.camera_pitch = 0.0
        self.step_count = 0
        self._leg_step_count = 0
        self.prev_action = 11  # Stay / Idle
        self.path_length = 0.0

    def _distance_to_target(self) -> float:
        return float(np.linalg.norm(self.player_pos.astype(float) - self.target_pos.astype(float)))

    def _snap_yaw_to_cardinal(self, yaw_deg: float) -> str:
        norm = (yaw_deg % 360.0 + 360.0) % 360.0
        if 45.0 <= norm < 135.0:
            return "WEST"
        elif 135.0 <= norm < 225.0:
            return "NORTH"
        elif 225.0 <= norm < 315.0:
            return "EAST"
        else:
            return "SOUTH"

    def _build_action_mask(self) -> np.ndarray:
        """Computes valid/safe action mask across the 12 discrete actions."""
        mask = np.ones(self.NUM_ACTIONS, dtype=bool)
        if not self.mask_fatal:
            return mask

        # Check translations
        actions_to_dx_dz = {
            0: (0, 1),   # Forward (+Z in ego)
            1: (-1, 1),  # Forward-Left
            2: (1, 1),   # Forward-Right
            3: (-1, 0),  # Strafe-Left
            4: (1, 0),   # Strafe-Right
            5: (0, -1),  # Backward
            8: (0, 1)    # Step-Up-Forward
        }

        fwd_v = CARDINAL_VECTORS[self.cardinal]["forward"]
        right_v = CARDINAL_VECTORS[self.cardinal]["right"]

        for act, (lx, lz) in actions_to_dx_dz.items():
            # Rotate ego delta (lx, lz) into world (wx, wz)
            wx = lx * right_v[0] + lz * fwd_v[0]
            wz = lx * right_v[1] + lz * fwd_v[1]
            _, _, _, outcome, _ = self.kernel.step(
                self.player_pos[0],
                self.player_pos[1],
                self.player_pos[2],
                wx,
                wz
            )
            if outcome in (MoveOutcome.FATAL_FALL, MoveOutcome.FATAL_HAZARD, MoveOutcome.FATAL_VOID, MoveOutcome.BLOCKED):
                mask[act] = False

        if not np.any(mask):
            mask[11] = True  # Stay is always safe fallback
        return mask

    def _extract_16ch_shells(self) -> np.ndarray:
        """
        Extracts multi-LOD concentric voxel shells into (3, 16, 16, 16, 16) float32
        channel-last tensor matching docs/CONTRACT.md.
        """
        foot_idx = calculate_foot_index(self.camera_pitch)
        raw_obs = self.extractor.extract_observation(
            tuple(self.player_pos),
            self.cardinal,
            foot_index=foot_idx
        )

        shells = np.zeros((3, 16, 16, 16, 16), dtype=np.float32)

        blocks_dict = raw_obs.get("blocks", raw_obs)
        for lod in range(3):
            lod_data = blocks_dict[f"lod{lod}"]
            block_ids = lod_data["block_id"]

            # Channel 0: TERRAIN (Solid)
            shells[lod, ..., Channel.TERRAIN] = lod_data["occupancy"]
            # Channel 1: AIR
            shells[lod, ..., Channel.AIR] = lod_data["passability"]
            # Channel 2: HAZARD
            shells[lod, ..., Channel.HAZARD] = lod_data["hazard"]
            # Channel 3: LIQUID
            shells[lod, ..., Channel.LIQUID] = self.palette.lut_channel_liquid[block_ids].astype(np.float32)
            # Channel 4: CLIMBABLE
            shells[lod, ..., Channel.CLIMBABLE] = self.palette.lut_channel_climbable[block_ids].astype(np.float32)
            # Channel 5: FOLIAGE
            shells[lod, ..., Channel.FOLIAGE] = self.palette.lut_channel_foliage[block_ids].astype(np.float32)

            # Channel 6: WORLD_Y normalized in [-64..320] -> [0..1]
            stride = 1 if lod == 0 else (2 if lod == 1 else 4)
            gy_indices = np.arange(16)
            world_y_vals = self.player_pos[1] + (gy_indices - foot_idx) * stride
            norm_y = np.clip((world_y_vals - (-64.0)) / 384.0, 0.0, 1.0).astype(np.float32)
            shells[lod, :, :, :, Channel.WORLD_Y] = norm_y[None, :, None]

            # Channel 7: COLLISION_HEIGHT
            shells[lod, ..., Channel.COLLISION_HEIGHT] = self.palette.lut_collision_height[block_ids]
            # Channel 9: RESOURCE
            shells[lod, ..., Channel.RESOURCE] = self.palette.lut_channel_resource[block_ids].astype(np.float32)

            # Channel 10: WALKABLE_SURFACE
            # Solid block with air above it
            occ = lod_data["occupancy"]
            passab = lod_data["passability"]
            walkable = np.zeros_like(occ)
            walkable[:, :-1, :] = (occ[:, :-1, :] > 0.5) & (passab[:, 1:, :] > 0.5)
            shells[lod, ..., Channel.WALKABLE_SURFACE] = walkable

            # Channel 11: LIGHT_LEVEL
            shells[lod, ..., Channel.LIGHT_LEVEL] = lod_data["light_level"]

            # Channel 12: LETHAL_DROP
            # Passable air with >=3 air blocks below
            drop = np.zeros_like(passab)
            if lod == 0:
                drop[:, 3:, :] = (passab[:, 3:, :] > 0.5) & (passab[:, 2:-1, :] > 0.5) & (passab[:, 1:-2, :] > 0.5) & (passab[:, :-3, :] > 0.5)
            shells[lod, ..., Channel.LETHAL_DROP] = drop

            # Channel 13: SURFACE_SHELL
            shells[lod, ..., Channel.SURFACE_SHELL] = lod_data.get("visibility", np.ones_like(occ))

            # Channel 14: BLOCK_HARDNESS (normalized 0..1 from 0..50)
            shells[lod, ..., Channel.BLOCK_HARDNESS] = np.clip(lod_data["hardness"] / 50.0, 0.0, 1.0)

            # Channel 15: OUT_OF_BOUNDS (0 for within volume)
            shells[lod, ..., Channel.OUT_OF_BOUNDS] = 0.0

            # Channel 8: WAYPOINT Beacon Injection
            # Project active target (and remaining unreached targets in RL2) into egocentric grid coordinates
            fwd_v = CARDINAL_VECTORS[self.cardinal]["forward"]
            right_v = CARDINAL_VECTORS[self.cardinal]["right"]

            remaining_targets = self.targets[self.target_idx :] if self.targets else [self.target_pos]
            # Write future targets first (0.5) so active target (1.0) always takes precedence
            for offset_idx in range(len(remaining_targets) - 1, -1, -1):
                tgt = remaining_targets[offset_idx]
                beacon_val = 1.0 if offset_idx == 0 else 0.5

                dx_w = tgt[0] - self.player_pos[0]
                dy_w = tgt[1] - self.player_pos[1]
                dz_w = tgt[2] - self.player_pos[2]

                ego_x = dx_w * right_v[0] + dz_w * right_v[1]
                ego_z = dx_w * fwd_v[0] + dz_w * fwd_v[1]
                ego_y = dy_w

                gx = int(round(8 + ego_x / stride))
                gy = int(round(foot_idx + ego_y / stride))
                gz = int(round(2 + ego_z / stride))

                if 0 <= gx < 16 and 0 <= gz < 16:
                    y_min = max(0, gy - 2)
                    y_max = min(16, gy + 4)
                    if y_min < y_max:
                        shells[lod, gx, y_min:y_max, gz, Channel.WAYPOINT] = np.maximum(
                            shells[lod, gx, y_min:y_max, gz, Channel.WAYPOINT],
                            beacon_val,
                        )

        return shells

    def _get_obs(self) -> Dict[str, np.ndarray]:
        shells = (
            np.zeros(SHELL_VOXEL_SHAPE, dtype=np.float32)
            if self.skip_shells
            else self._extract_16ch_shells()
        )

        dx, dy, dz = (self.player_pos - self.prev_pos).astype(np.float32)
        yaw_rad = math.radians(self.yaw)
        grounded = 1.0 if self.world.get_block(self.player_pos[0], self.player_pos[1] - 1, self.player_pos[2]) != 0 else 0.0

        state = np.array([
            dx, dy, dz,
            dx, dy, dz,
            math.sin(yaw_rad),
            math.cos(yaw_rad),
            float(self.camera_pitch),
            20.0,
            20.0,
            grounded
        ], dtype=np.float32)

        # Egocentric 3D relative goal vector (with optional occlusion for RL2/RL3/RL4)
        if self.compass_occluded:
            goal_vec = np.zeros(3, dtype=np.float32)
        else:
            dx_w = self.target_pos[0] - self.player_pos[0]
            dy_w = self.target_pos[1] - self.player_pos[1]
            dz_w = self.target_pos[2] - self.player_pos[2]

            fwd_v = CARDINAL_VECTORS[self.cardinal]["forward"]
            right_v = CARDINAL_VECTORS[self.cardinal]["right"]

            ego_gx = float(dx_w * right_v[0] + dz_w * right_v[1])
            ego_gz = float(dx_w * fwd_v[0] + dz_w * fwd_v[1])
            ego_gy = float(dy_w)

            goal_vec = np.array([ego_gx, ego_gy, ego_gz], dtype=np.float32)
        action_mask = self._build_action_mask()

        return {
            "shells": shells,
            "state": state,
            "goal_vec": goal_vec,
            "action_mask": action_mask
        }

    def reset(self, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)

        # 1. Sample spawn position
        spawn_coord = self.spawn_sampler.sample(rng=rng)
        self.player_pos = np.array(spawn_coord, dtype=int)
        self.prev_pos = np.copy(self.player_pos)

        # 2. Pick random cardinal heading
        self.cardinal = str(rng.choice(CARDINAL_ORDER))
        self.yaw = CARDINAL_YAW_DEG[self.cardinal]
        self.camera_pitch = 0.0
        self.step_count = 0
        self._leg_step_count = 0
        self.path_length = 0.0
        self.prev_action = 11

        # 3. Generate reachable BFS target(s)
        foot_idx = calculate_foot_index(self.camera_pitch)
        if self.num_targets == 1:
            dist_range = self.curriculum.get_distance_range()
            target_coord, l_bfs, path = self.target_gen.generate_target(
                spawn=tuple(self.player_pos),
                cardinal=self.cardinal,
                foot_index=foot_idx,
                target_dist_range=dist_range,
                rng=rng
            )
            self.targets = [np.array(target_coord, dtype=int)]
            self.target_paths = [path]
            self.shortest_path_len = max(1.0, float(l_bfs))
        else:
            self.targets = []
            self.target_paths = []
            curr_anchor = tuple(self.player_pos)
            visited_targets = {curr_anchor}
            total_bfs_len = 0.0
            for t_i in range(self.num_targets):
                best_coord, best_l, best_path = curr_anchor, 0, [curr_anchor]
                # Try primary cardinal first, then fallback orientations to avoid dead-ends
                try_cardinals = [self.cardinal] + [c for c in CARDINAL_ORDER if c != self.cardinal]
                for try_card in try_cardinals:
                    t_coord, l_bfs, path = self.target_gen.generate_target(
                        spawn=curr_anchor,
                        cardinal=try_card,
                        foot_index=foot_idx,
                        target_dist_range=(10, 24),
                        rng=rng,
                    )
                    if l_bfs > 0 and tuple(t_coord) not in visited_targets:
                        best_coord, best_l, best_path = t_coord, l_bfs, path
                        break
                    elif l_bfs > best_l:
                        best_coord, best_l, best_path = t_coord, l_bfs, path
                self.targets.append(np.array(best_coord, dtype=int))
                self.target_paths.append(best_path)
                visited_targets.add(tuple(best_coord))
                total_bfs_len += max(1.0, float(best_l))
                curr_anchor = tuple(best_coord)
            self.shortest_path_len = max(1.0, total_bfs_len)

        self.target_idx = 0
        self.targets_reached = 0
        self._target_reach_steps = []
        self.target_pos = self.targets[0]
        self.optimal_path = self.target_paths[0]

        # 4. Initialize compass occlusion state (RL3/RL4 flash & beacon precedence)
        self._occlusion_rng = rng
        if self.flash_compass_ticks > 0:
            self._occlusion_phase = 0
            self.compass_occluded = False
        elif self.compass_occlusion_prob >= 1.0:
            self._occlusion_phase = 0
            self.compass_occluded = True
        elif self.compass_occlusion_prob > 0.0:
            self._occlusion_phase = int(rng.integers(0, self.occlusion_burst_len))
            self.compass_occluded = bool(rng.random() < self.compass_occlusion_prob)
        else:
            self._occlusion_phase = 0
            self.compass_occluded = False

        return self._get_obs()

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        was_occluded = bool(self.compass_occluded)
        self.step_count += 1
        self._leg_step_count += 1
        self.prev_pos = np.copy(self.player_pos)
        prev_yaw = self.yaw
        d_prev = self._distance_to_target()

        reward = -0.02  # Time penalty
        collision = False
        blocked = False
        done = False
        success = False

        # Compute pre-action horizontal heading alignment with target
        yaw_rad_prev = math.radians(prev_yaw)
        fwd_prev_x = -math.sin(yaw_rad_prev)
        fwd_prev_z = math.cos(yaw_rad_prev)
        dx_t_prev = float(self.target_pos[0] - self.prev_pos[0])
        dz_t_prev = float(self.target_pos[2] - self.prev_pos[2])
        norm_t_prev = math.hypot(dx_t_prev, dz_t_prev)
        cos_prev = (fwd_prev_x * dx_t_prev + fwd_prev_z * dz_t_prev) / (norm_t_prev + 1e-6)

        fwd_v = CARDINAL_VECTORS[self.cardinal]["forward"]
        right_v = CARDINAL_VECTORS[self.cardinal]["right"]

        # Parse 12 actions
        # 0: Forward, 1: Forward-Left, 2: Forward-Right, 3: Strafe-Left, 4: Strafe-Right,
        # 5: Backward, 6: Turn-Left (-45°), 7: Turn-Right (+45°), 8: Step-Up-Forward,
        # 9: Pitch-Up (+15°), 10: Pitch-Down (-15°), 11: Stay / Idle
        ego_dx, ego_dz = 0, 0

        if action == 0:     # Forward
            ego_dx, ego_dz = 0, 1
        elif action == 1:   # Forward-Left
            ego_dx, ego_dz = -1, 1
        elif action == 2:   # Forward-Right
            ego_dx, ego_dz = 1, 1
        elif action == 3:   # Strafe-Left
            ego_dx, ego_dz = -1, 0
        elif action == 4:   # Strafe-Right
            ego_dx, ego_dz = 1, 0
        elif action == 5:   # Backward
            ego_dx, ego_dz = 0, -1
        elif action == 6:   # Turn-Left (-45° -> updates yaw and snaps cardinal)
            self.yaw = (self.yaw - 45.0) % 360.0
            self.cardinal = self._snap_yaw_to_cardinal(self.yaw)
        elif action == 7:   # Turn-Right (+45° -> updates yaw and snaps cardinal)
            self.yaw = (self.yaw + 45.0) % 360.0
            self.cardinal = self._snap_yaw_to_cardinal(self.yaw)
        elif action == 8:   # Step-Up-Forward
            ego_dx, ego_dz = 0, 1
        elif action == 9:   # Pitch-Up (+15°)
            self.camera_pitch = min(90.0, self.camera_pitch + 15.0)
        elif action == 10:  # Pitch-Down (-15°)
            self.camera_pitch = max(-90.0, self.camera_pitch - 15.0)
        elif action == 11:  # Stay / Idle
            pass

        self.prev_action = action

        # Apply horizontal movement physics if translational action
        if ego_dx != 0 or ego_dz != 0:
            world_dx = ego_dx * right_v[0] + ego_dz * fwd_v[0]
            world_dz = ego_dx * right_v[1] + ego_dz * fwd_v[1]

            nx, ny, nz, outcome, fall_dist = self.kernel.step(
                self.player_pos[0],
                self.player_pos[1],
                self.player_pos[2],
                world_dx,
                world_dz
            )

            if outcome in (MoveOutcome.SUCCESS, MoveOutcome.STEP_UP, MoveOutcome.FALL):
                self.player_pos = np.array([nx, ny, nz], dtype=int)
                step_dist = float(np.linalg.norm(self.player_pos - self.prev_pos))
                self.path_length += step_dist
                if np.array_equal(self.player_pos, self.prev_pos):
                    reward -= 0.15
                    collision = True
                    blocked = True
            elif outcome == MoveOutcome.BLOCKED:
                reward -= 0.15
                collision = True
                blocked = True
            elif outcome == MoveOutcome.FATAL_HAZARD:
                reward -= 5.0
                collision = True
                done = True
            elif outcome == MoveOutcome.FATAL_FALL:
                reward -= 3.0
                collision = True
                done = True
            elif outcome == MoveOutcome.FATAL_VOID:
                reward -= 5.0
                collision = True
                done = True

        d_curr = self._distance_to_target()

        # Potential-based distance shaping: +0.25 * (d_{t-1} - d_t)
        reward += 0.25 * (d_prev - d_curr)

        # Heading-alignment shaping bonus: +0.05 * (cos_after - cos_before)
        yaw_rad_curr = math.radians(self.yaw)
        fwd_curr_x = -math.sin(yaw_rad_curr)
        fwd_curr_z = math.cos(yaw_rad_curr)
        dx_t_curr = float(self.target_pos[0] - self.player_pos[0])
        dz_t_curr = float(self.target_pos[2] - self.player_pos[2])
        norm_t_curr = math.hypot(dx_t_curr, dz_t_curr)
        cos_curr = (fwd_curr_x * dx_t_curr + fwd_curr_z * dz_t_curr) / (norm_t_curr + 1e-6)
        reward += 0.05 * (cos_curr - cos_prev)

        # Check goal reach (dist <= 1.5 blocks)
        if d_curr <= self.goal_reach_radius:
            reward += 10.0
            self.targets_reached += 1
            self._target_reach_steps.append(int(self.step_count))
            self.target_idx += 1
            if self.target_idx >= len(self.targets):
                done = True
                success = True
                self.curriculum.record_result(True, self.path_length, self.shortest_path_len)
            else:
                # Advance to next target in chain without terminating episode or resetting memory.
                # Reset leg tick counter so RL3/RL4 flash mode grants a fresh compass flash per leg.
                self._leg_step_count = 0
                self.target_pos = self.targets[self.target_idx]
                self.optimal_path = self.target_paths[self.target_idx]

        # Timeout check
        if self.step_count >= self.max_steps:
            done = True
            if not success:
                self.curriculum.record_result(False, self.path_length, self.shortest_path_len)

        # Update compass occlusion state for the next observation
        if self.flash_compass_ticks > 0:
            self.compass_occluded = self._leg_step_count >= self.flash_compass_ticks
        elif self.compass_occlusion_prob >= 1.0:
            self.compass_occluded = True
        elif self.compass_occlusion_prob > 0.0:
            if (self.step_count + self._occlusion_phase) % self.occlusion_burst_len == 0:
                if abs(self.compass_occlusion_prob - 0.5) < 1e-6:
                    self.compass_occluded = not self.compass_occluded
                else:
                    self.compass_occluded = bool(self._occlusion_rng.random() < self.compass_occlusion_prob)

        obs = self._get_obs()
        info = {
            "success": success,
            "path_length": self.path_length,
            "targets_reached": self.targets_reached,
            "num_targets": self.num_targets,
            "waypoints_reached": self.targets_reached,
            "total_waypoints": self.num_targets,
            "target_reach_steps": list(self._target_reach_steps),
            "compass_occluded": was_occluded,
            "collision": collision,
            "blocked": blocked,
            "shortest_path_len": self.shortest_path_len,
            "target_distance": d_curr
        }

        return obs, float(reward), done, info

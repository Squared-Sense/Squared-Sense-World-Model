"""Deterministic 3D Voxel Navigation Simulator for Squared Sense.

Simulates agent movement in a 16-channel voxel arena volume:
  - Arena volume: (16, 128, 64, 128) uint8 (C, X, Y, Z)
  - 12 discrete actions: translation, 45° turning, 1-block step-up, pitch, idle
  - Physics: 1-block step-up, drop physics (dy <= -3 triggers fall penalty),
    stepping on hazard (ch 2 >= 128) triggers termination
  - Potential-based reward function matching docs/EVAL_CONTRACT.md
  - Concentric multi-LOD shell observation extraction with cardinal snapping

Strictly follows docs/EVAL_CONTRACT.md and docs/CONTRACT.md.
"""

from __future__ import annotations

import math
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from data.schema import (
    AVG_POOL_CHANNELS,
    COURSE_SHAPE,
    LOD_SPECS,
    MAX_POOL_CHANNELS,
    STATE_DIM,
    Channel,
    normalize_shells,
)
from env.palette import get_palette
from env.physics_kernel import MovementPhysicsKernel, MoveOutcome, VoxelCourseAdapter


# Module-level channel indexing lists for ultra-fast pooling
_MAX_CH: List[int] = list(MAX_POOL_CHANNELS)
_AVG_CH: List[int] = list(AVG_POOL_CHANNELS)


# ---------------------------------------------------------------------------
# Shell Extraction & Cardinal Snapping (with Fallback)
# ---------------------------------------------------------------------------

try:
    from data.shell_extractor import CardinalSnapper, extract_shells
except ImportError:
    class CardinalSnapper:
        """Cardinals yaw snapping with ±10° rotational hysteresis deadband.

        Snaps yaw to nearest 90° cardinal axis (0°, 90°, 180°, 270°) and retains
        the snapped direction until yaw departs by more than 45° + deadband (55° for 10°).
        """

        def __init__(self, initial_yaw: float = 0.0, deadband: float = 10.0) -> None:
            self.deadband = deadband
            self.snapped_yaw = self._nearest_cardinal(initial_yaw)

        @staticmethod
        def _nearest_cardinal(yaw: float) -> float:
            norm = yaw % 360.0
            snapped = round(norm / 90.0) * 90.0
            return float(snapped % 360.0)

        @staticmethod
        def _angle_diff(a: float, b: float) -> float:
            diff = (a - b + 180.0) % 360.0 - 180.0
            return diff

        def snap(self, yaw: float) -> float:
            norm = yaw % 360.0
            diff = abs(self._angle_diff(norm, self.snapped_yaw))
            if diff > (45.0 + self.deadband):
                self.snapped_yaw = self._nearest_cardinal(norm)
            return self.snapped_yaw


    def extract_shells(
        volume: np.ndarray,
        pos: Tuple[int, int, int] | np.ndarray,
        yaw: float = 0.0,
        pitch: float = 0.0,
        use_salience_pool: bool = True,
        snapper: Optional[CardinalSnapper] = None,
        is_padded: bool = False,
        forward_anchor: int = 2,
        foot_anchor: int = 6,
    ) -> np.ndarray:
        """Extract multi-LOD concentric voxel shells centered at agent position.

        Args:
            volume: (16, 128, 64, 128) uint8 volume array (C, X, Y, Z) or pre-padded volume.
            pos: (x, y, z) agent integer coordinates in volume.
            yaw: Agent yaw angle in degrees.
            pitch: Agent pitch angle in degrees.
            use_salience_pool: If True, uses max-pooling on MAX_POOL_CHANNELS and
              avg-pooling on AVG_POOL_CHANNELS; if False, uses strided subsampling.
            snapper: Optional CardinalSnapper instance.
            is_padded: Whether volume has 64-voxel margin padding.
            forward_anchor: Player forward anchor index in grid (default 2 matching mod).
            foot_anchor: Player vertical foot index in grid (default 6 matching mod).

        Returns:
            (3, 16, 16, 16, 16) float32 array in [0.0, 1.0] with channel-last layout (LOD, X, Y, Z, C).
        """
        if snapper is None:
            snapper = CardinalSnapper(yaw)
        cardinal_yaw = snapper.snap(yaw)

        margin = 64 if is_padded else 0
        px = int(round(pos[0])) + margin
        py = int(round(pos[1])) + margin
        pz = int(round(pos[2])) + margin
        c_dim, x_dim, y_dim, z_dim = volume.shape

        shells = np.empty((3, 16, 16, 16, 16), dtype=np.float32)
        inv_255 = np.float32(1.0 / 255.0)

        for lod_idx, spec in enumerate(LOD_SPECS):
            scale = spec.scale
            span = 16 * scale
            shift_y = int(round(4.0 * math.sin(math.radians(pitch)))) * scale

            rear_offset = forward_anchor * scale
            front_offset = (16 - forward_anchor) * scale
            below_offset = foot_anchor * scale
            above_offset = (16 - foot_anchor) * scale

            if cardinal_yaw == 0.0:
                x_min, x_max = px - 8 * scale, px + 8 * scale
                z_min, z_max = pz - rear_offset, pz + front_offset
                rot_k = 0
            elif cardinal_yaw == 90.0:
                x_min, x_max = px - rear_offset, px + front_offset
                z_min, z_max = pz - 8 * scale, pz + 8 * scale
                rot_k = 1
            elif cardinal_yaw == 180.0:
                x_min, x_max = px - 8 * scale, px + 8 * scale
                z_min, z_max = pz - front_offset, pz + rear_offset
                rot_k = 2
            else:  # 270.0
                x_min, x_max = px - front_offset, px + rear_offset
                z_min, z_max = pz - 8 * scale, pz + 8 * scale
                rot_k = 3

            y_min = py - below_offset + shift_y
            y_max = py + above_offset + shift_y

            if is_padded or (x_min >= 0 and x_max <= x_dim and y_min >= 0 and y_max <= y_dim and z_min >= 0 and z_max <= z_dim):
                crop = volume[:, x_min:x_max, y_min:y_max, z_min:z_max]
            else:
                crop = np.zeros((16, span, span, span), dtype=np.uint8)
                crop[Channel.OUT_OF_BOUNDS] = 255
                src_x0, src_x1 = max(0, x_min), min(x_dim, x_max)
                src_y0, src_y1 = max(0, y_min), min(y_dim, y_max)
                src_z0, src_z1 = max(0, z_min), min(z_dim, z_max)
                if src_x1 > src_x0 and src_y1 > src_y0 and src_z1 > src_z0:
                    dst_x0, dst_x1 = src_x0 - x_min, src_x1 - x_min
                    dst_y0, dst_y1 = src_y0 - y_min, src_y1 - y_min
                    dst_z0, dst_z1 = src_z0 - z_min, src_z1 - z_min
                    crop[:, dst_x0:dst_x1, dst_y0:dst_y1, dst_z0:dst_z1] = volume[
                        :, src_x0:src_x1, src_y0:src_y1, src_z0:src_z1
                    ]

            if rot_k != 0:
                crop = np.rot90(crop, k=rot_k, axes=(1, 3))

            if scale == 1:
                np.multiply(crop.transpose(1, 2, 3, 0), inv_255, out=shells[lod_idx], casting="unsafe")
            elif not use_salience_pool:
                sub = crop[:, ::scale, ::scale, ::scale]
                np.multiply(sub.transpose(1, 2, 3, 0), inv_255, out=shells[lod_idx], casting="unsafe")
            elif scale == 2:
                m = np.empty((16, 16, 16, 16), dtype=np.uint8)
                c_max = crop[_MAX_CH]
                m_max = np.maximum(c_max[:, 0::2, :, :], c_max[:, 1::2, :, :])
                m_max = np.maximum(m_max[:, :, 0::2, :], m_max[:, :, 1::2, :])
                m_max = np.maximum(m_max[:, :, :, 0::2], m_max[:, :, :, 1::2])
                m[_MAX_CH] = m_max

                c_avg = crop[_AVG_CH]
                a = c_avg[:, 0::2, :, :].astype(np.uint16) + c_avg[:, 1::2, :, :]
                a = a[:, :, 0::2, :] + a[:, :, 1::2, :]
                a = a[:, :, :, 0::2] + a[:, :, :, 1::2]
                m[_AVG_CH] = ((a + 4) >> 3).astype(np.uint8)
                np.multiply(m.transpose(1, 2, 3, 0), inv_255, out=shells[lod_idx], casting="unsafe")
            else:  # scale == 4
                m = np.empty((16, 16, 16, 16), dtype=np.uint8)
                c_max = crop[_MAX_CH]
                s1 = np.maximum(c_max[:, 0::2, :, :], c_max[:, 1::2, :, :])
                s1 = np.maximum(s1[:, :, 0::2, :], s1[:, :, 1::2, :])
                s1 = np.maximum(s1[:, :, :, 0::2], s1[:, :, :, 1::2])

                s2 = np.maximum(s1[:, 0::2, :, :], s1[:, 1::2, :, :])
                s2 = np.maximum(s2[:, :, 0::2, :], s2[:, :, 1::2, :])
                m[_MAX_CH] = np.maximum(s2[:, :, :, 0::2], s2[:, :, :, 1::2])

                c_avg = crop[_AVG_CH]
                a1 = c_avg[:, 0::2, :, :].astype(np.uint16) + c_avg[:, 1::2, :, :]
                a1 = a1[:, :, 0::2, :] + a1[:, :, 1::2, :]
                a1 = a1[:, :, :, 0::2] + a1[:, :, :, 1::2]

                a2 = a1[:, 0::2, :, :] + a1[:, 1::2, :]
                a2 = a2[:, :, 0::2, :] + a2[:, :, 1::2, :]
                a2 = a2[:, :, :, 0::2] + a2[:, :, :, 1::2]
                m[_AVG_CH] = ((a2 + 32) >> 6).astype(np.uint8)
                np.multiply(m.transpose(1, 2, 3, 0), inv_255, out=shells[lod_idx], casting="unsafe")

        return shells


# ---------------------------------------------------------------------------
# Debug Course Generator
# ---------------------------------------------------------------------------

def make_debug_bridge_course(
    seed: int = 0,
) -> Tuple[np.ndarray, Tuple[int, int, int], List[Tuple[int, int, int]]]:
    """Generate a (16, 128, 64, 128) test arena with a 1-block-wide bridge over lava.

    Features:
      - Start position: (64, 20, 20)
      - Distant waypoint: (64, 20, 56) (36 blocks ahead along +Z)
      - 1-block-wide bridge: x=64, y=19, z in [20, 60] (solid walkable terrain)
      - Lava trench: x in [60, 64) and (64, 68], y=19, z in [20, 58] (hazard ch 2)
      - Solid wall behind start: x=64, y in [20, 22], z=19 (wall bump test)
      - Air above walkable bridge and lava

    Args:
        seed: Random seed (unused for deterministic debug bridge).

    Returns:
        (volume, start, waypoints) tuple.
    """
    volume = np.zeros(COURSE_SHAPE, dtype=np.uint8)

    # Fill default passable air everywhere
    volume[Channel.AIR] = 255

    # 1. 1-block-wide bridge along +Z
    bridge_x = 64
    bridge_y = 19
    z_start, z_end = 20, 62

    volume[Channel.TERRAIN, bridge_x, bridge_y, z_start:z_end] = 255
    volume[Channel.COLLISION_HEIGHT, bridge_x, bridge_y, z_start:z_end] = 255
    volume[Channel.WALKABLE_SURFACE, bridge_x, bridge_y, z_start:z_end] = 255
    volume[Channel.AIR, bridge_x, bridge_y, z_start:z_end] = 0

    # Ensure air clearance above the bridge (height 2 blocks for agent)
    volume[Channel.AIR, bridge_x, bridge_y + 1 : bridge_y + 4, z_start:z_end] = 255

    # 2. Lava trench directly flanking the bridge
    for x in range(60, 69):
        if x == bridge_x:
            continue
        volume[Channel.HAZARD, x, bridge_y, z_start:z_end] = 255
        volume[Channel.AIR, x, bridge_y, z_start:z_end] = 0
        volume[Channel.TERRAIN, x, bridge_y, z_start:z_end] = 0
        # Air clearance above lava so agent can fall or step into it
        volume[Channel.AIR, x, bridge_y + 1 : bridge_y + 4, z_start:z_end] = 255

    # 3. Solid wall immediately behind start position (at z=19)
    wall_z = 19
    volume[Channel.TERRAIN, bridge_x, bridge_y + 1 : bridge_y + 3, wall_z] = 255
    volume[Channel.COLLISION_HEIGHT, bridge_x, bridge_y + 1 : bridge_y + 3, wall_z] = 255
    volume[Channel.AIR, bridge_x, bridge_y + 1 : bridge_y + 3, wall_z] = 0

    start = (64, 20, 20)
    waypoints = [(64, 20, 56)]  # Exactly 36 blocks ahead

    return volume, start, waypoints


# ---------------------------------------------------------------------------
# Deterministic 3D Voxel Navigation Simulator
# ---------------------------------------------------------------------------

class VoxelNavEnv:
    """Deterministic 3D Voxel Navigation Simulator.

    Implements:
      - 16-channel uint8 arena volume (16, 128, 64, 128)
      - Active waypoint beacon injection into ch 8 (255)
      - 12 discrete actions with 1-block step-up, drop physics, and hazard detection
      - Potential-based distance shaping reward function
    """

    NUM_ACTIONS = 12

    def __init__(
        self,
        volume: np.ndarray,
        start: Tuple[int, int, int],
        waypoints: List[Tuple[int, int, int]],
        max_steps: int = 250,
        use_salience_pool: bool = True,
        courses: Optional[List[Tuple[np.ndarray, Tuple[int, int, int], List[Tuple[int, int, int]], float]]] = None,
    ) -> None:
        self.courses = courses
        self.course_idx = 0
        self.volume = np.copy(volume)
        self.margin = 64
        self.padded_volume = np.zeros((16, 128 + 128, 64 + 128, 128 + 128), dtype=np.uint8)
        self.padded_volume[Channel.OUT_OF_BOUNDS] = 255
        self.padded_volume[:, self.margin : self.margin + 128, self.margin : self.margin + 64, self.margin : self.margin + 128] = self.volume

        self.start = (int(start[0]), int(start[1]), int(start[2]))
        self.waypoints = [(int(w[0]), int(w[1]), int(w[2])) for w in waypoints]
        self.max_steps = max_steps
        self.use_salience_pool = use_salience_pool

        # Consolidated MovementPhysicsKernel & VoxelCourseAdapter
        self.palette = get_palette()
        self.adapter = VoxelCourseAdapter(self.volume, self.palette)
        self.kernel = MovementPhysicsKernel(self.adapter, self.palette, max_safe_fall=2)

        # State tracking
        self.pos = np.array(self.start, dtype=int)
        self.prev_pos = np.copy(self.pos)
        self.yaw = 0.0
        self.pitch = 0.0
        self.step_count = 0
        self.active_waypoint_idx = 0
        self.waypoints_reached = 0
        self.path_length = 0.0
        self.snapper = CardinalSnapper(self.yaw)
        self._obs_cache_key: Optional[Tuple[int, int, int, float, float, int]] = None
        self._cached_shells: Optional[np.ndarray] = None

        # Invalidate / initialize beacon in volume ch 8
        self._inject_waypoint_beacon(self.active_waypoint_idx)

    def _inject_waypoint_beacon(self, wp_idx: int) -> None:
        """Inject active waypoint beacon into ch 8 of volume."""
        self._obs_cache_key = None
        self._cached_shells = None
        # Clear existing waypoints in ch 8
        self.volume[Channel.WAYPOINT] = 0
        self.padded_volume[Channel.WAYPOINT] = 0
        if wp_idx < len(self.waypoints):
            wx, wy, wz = self.waypoints[wp_idx]
            # Set beacon pillar at waypoint so it is visible in LOD-2
            self.volume[Channel.WAYPOINT, wx, max(0, wy - 2) : min(64, wy + 4), wz] = 255
            self.padded_volume[
                Channel.WAYPOINT,
                wx + self.margin,
                max(0, wy - 2) + self.margin : min(64, wy + 4) + self.margin,
                wz + self.margin,
            ] = 255

    @property
    def active_waypoint(self) -> Tuple[int, int, int]:
        """Return coordinates of the active waypoint."""
        if self.active_waypoint_idx < len(self.waypoints):
            return self.waypoints[self.active_waypoint_idx]
        return self.waypoints[-1]

    def _distance_to_active(self) -> float:
        """Euclidean distance to the current active waypoint."""
        target = np.array(self.active_waypoint, dtype=float)
        return float(np.linalg.norm(self.pos.astype(float) - target))

    def _is_solid(self, x: int, y: int, z: int) -> bool:
        """Check if voxel is solid walkable terrain."""
        return self.adapter.is_solid(x, y, z)

    def _is_hazard(self, x: int, y: int, z: int) -> bool:
        """Check if voxel is a hazard (lava, fire, magma)."""
        return self.adapter.is_hazard(x, y, z)

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Construct observation dictionary."""
        cache_key = (
            int(self.pos[0]),
            int(self.pos[1]),
            int(self.pos[2]),
            float(self.yaw),
            float(self.pitch),
            self.active_waypoint_idx,
        )
        if self._obs_cache_key == cache_key and self._cached_shells is not None:
            shells = self._cached_shells
        else:
            shells = extract_shells(
                volume=self.padded_volume,
                pos=self.pos,
                yaw=self.yaw,
                pitch=self.pitch,
                use_salience_pool=self.use_salience_pool,
                snapper=self.snapper,
                is_padded=True,
            )
            self._obs_cache_key = cache_key
            self._cached_shells = shells

        # Proprioceptive state (12,)
        # [vx, vy, vz, dx, dy, dz, sin_yaw, cos_yaw, pitch, health, food, grounded]
        dx, dy, dz = (self.pos - self.prev_pos).astype(np.float32)
        yaw_rad = math.radians(self.yaw)
        grounded = 1.0 if self._is_solid(self.pos[0], self.pos[1] - 1, self.pos[2]) else 0.0

        state = np.array(
            [
                dx, dy, dz,  # vx, vy, vz
                dx, dy, dz,  # dx, dy, dz
                math.sin(yaw_rad),
                math.cos(yaw_rad),
                self.pitch / 90.0,
                20.0 / 20.0,        # health normalized
                20.0 / 20.0,        # food normalized
                grounded,           # grounded
            ],
            dtype=np.float32,
        )

        # 3D goal vector relative to agent position in cardinal egocentric frame
        goal_target = np.array(self.active_waypoint, dtype=np.float32)
        dx_w = goal_target[0] - float(self.pos[0])
        dy_w = goal_target[1] - float(self.pos[1])
        dz_w = goal_target[2] - float(self.pos[2])

        cardinal_yaw = self.snapper.snap(self.yaw)
        rad = math.radians(cardinal_yaw)
        fwd_x = math.sin(rad)
        fwd_z = math.cos(rad)
        right_x = math.cos(rad)
        right_z = -math.sin(rad)

        ego_gx = float(dx_w * right_x + dz_w * right_z)
        ego_gz = float(dx_w * fwd_x + dz_w * fwd_z)
        ego_gy = float(dy_w)
        goal_vec = np.array([ego_gx, ego_gy, ego_gz], dtype=np.float32)

        return {
            "shells": shells,
            "state": state,
            "goal_vec": goal_vec,
        }

    def reset(self, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Reset environment to initial state."""
        if self.courses and len(self.courses) > 1:
            vol, st, wps, _ = self.courses[self.course_idx % len(self.courses)]
            self.course_idx += 1
            self.volume = np.copy(vol)
            self.padded_volume[Channel.OUT_OF_BOUNDS] = 255
            self.padded_volume[:, self.margin : self.margin + 128, self.margin : self.margin + 64, self.margin : self.margin + 128] = self.volume
            self.start = (int(st[0]), int(st[1]), int(st[2]))
            self.waypoints = [(int(w[0]), int(w[1]), int(w[2])) for w in wps]
            self.adapter = VoxelCourseAdapter(self.volume, self.palette)
            self.kernel = MovementPhysicsKernel(self.adapter, self.palette, max_safe_fall=2)

        self.pos = np.array(self.start, dtype=int)
        self.prev_pos = np.copy(self.pos)
        self.yaw = 0.0
        self.pitch = 0.0
        self.step_count = 0
        self.active_waypoint_idx = 0
        self.waypoints_reached = 0
        self.path_length = 0.0
        self.snapper = CardinalSnapper(self.yaw)

        self._inject_waypoint_beacon(0)
        return self._get_obs()

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """Execute one simulation step.

        Actions:
          0: Forward, 1: Forward-Left, 2: Forward-Right, 3: Strafe-Left, 4: Strafe-Right,
          5: Backward, 6: Turn-Left (-45°), 7: Turn-Right (+45°), 8: Step-Up-Forward,
          9: Pitch-Up (+15°), 10: Pitch-Down (-15°), 11: Stay / Idle.

        Returns:
            (obs, reward, done, info) tuple.
        """
        self.step_count += 1
        self.prev_pos = np.copy(self.pos)
        d_prev = self._distance_to_active()

        reward = -0.02  # Per-step time penalty
        collision = False
        wall_bump = False
        fall_penalty = False
        hazard_stepped = False
        done = False
        success = False

        # Direction vectors based on current yaw (in degrees)
        # 0°: +Z forward, +X right
        rad = math.radians(self.yaw)
        f_x, f_z = math.sin(rad), math.cos(rad)
        r_x, r_z = math.cos(rad), -math.sin(rad)
        l_x, l_z = -r_x, -r_z

        move_dx, move_dz = 0, 0
        step_up_req = False

        # Parse action
        if action == 0:  # Forward
            move_dx, move_dz = int(round(f_x)), int(round(f_z))
        elif action == 1:  # Forward-Left
            move_dx, move_dz = int(round(f_x + l_x)), int(round(f_z + l_z))
        elif action == 2:  # Forward-Right
            move_dx, move_dz = int(round(f_x + r_x)), int(round(f_z + r_z))
        elif action == 3:  # Strafe-Left
            move_dx, move_dz = int(round(l_x)), int(round(l_z))
        elif action == 4:  # Strafe-Right
            move_dx, move_dz = int(round(r_x)), int(round(r_z))
        elif action == 5:  # Backward
            move_dx, move_dz = int(round(-f_x)), int(round(-f_z))
        elif action == 6:  # Turn-Left (-45°)
            self.yaw = (self.yaw - 45.0) % 360.0
        elif action == 7:  # Turn-Right (+45°)
            self.yaw = (self.yaw + 45.0) % 360.0
        elif action == 8:  # Step-Up-Forward
            move_dx, move_dz = int(round(f_x)), int(round(f_z))
            step_up_req = True
        elif action == 9:  # Pitch-Up (+15°)
            self.pitch = min(90.0, self.pitch + 15.0)
        elif action == 10:  # Pitch-Down (-15°)
            self.pitch = max(-90.0, self.pitch - 15.0)
        elif action == 11:  # Stay / Idle
            pass
        else:
            raise ValueError(f"Invalid action {action}. Must be in [0, 11]")

        # Process translation and collision via MovementPhysicsKernel
        if move_dx != 0 or move_dz != 0 or step_up_req:
            curr_x, curr_y, curr_z = self.pos
            if step_up_req and not self._is_solid(curr_x + move_dx, curr_y, curr_z + move_dz):
                wall_bump = True
                reward += -0.1
            else:
                next_x, next_y, next_z, outcome, fall_dist = self.kernel.step(
                    curr_x, curr_y, curr_z, move_dx, move_dz
                )
                if outcome == MoveOutcome.BLOCKED:
                    wall_bump = True
                    reward += -0.1
                elif outcome == MoveOutcome.FATAL_HAZARD:
                    self.pos = np.array([next_x, next_y, next_z], dtype=int)
                    hazard_stepped = True
                elif outcome in (MoveOutcome.FATAL_FALL, MoveOutcome.FATAL_VOID):
                    self.pos = np.array([next_x, next_y, next_z], dtype=int)
                    fall_penalty = True
                elif outcome in (MoveOutcome.SUCCESS, MoveOutcome.STEP_UP, MoveOutcome.FALL):
                    self.pos = np.array([next_x, next_y, next_z], dtype=int)

        # Hazard check (stepping into ch 2 >= 128)
        px, py, pz = self.pos
        if self._is_hazard(px, py, pz) or self._is_hazard(px, py - 1, pz):
            hazard_stepped = True

        # Penalties and termination
        if hazard_stepped:
            reward += -5.0
            done = True
            collision = True
        elif fall_penalty:
            reward += -3.0
            done = True
            collision = True

        # Cumulative path length
        step_dist = float(np.linalg.norm((self.pos - self.prev_pos).astype(float)))
        self.path_length += step_dist

        # Potential-based distance shaping
        d_curr = self._distance_to_active()
        reward += 0.25 * (d_prev - d_curr)

        # Waypoint reach check
        if not done and d_curr <= 1.5:
            reward += 10.0
            self.waypoints_reached += 1
            if self.active_waypoint_idx >= len(self.waypoints) - 1:
                done = True
                success = True
            else:
                self.active_waypoint_idx += 1
                self._inject_waypoint_beacon(self.active_waypoint_idx)

        # Time limit
        if self.step_count >= self.max_steps:
            done = True

        info = {
            "success": bool(success),
            "waypoints_reached": int(self.waypoints_reached),
            "total_waypoints": int(len(self.waypoints)),
            "collision": bool(collision or wall_bump),
            "path_length": float(self.path_length),
            "step": int(self.step_count),
        }

        obs = self._get_obs()
        return obs, float(reward), bool(done), info


def make_env(
    region_dir: Optional[str] = None,
    seed: int = 42,
    **kwargs: Any
) -> Any:
    """Factory creating MinecraftNavEnv if region_dir exists, else debug bridge VoxelNavEnv."""
    import os
    if region_dir is not None and os.path.exists(region_dir):
        from env.minecraft_env import MinecraftNavEnv
        return MinecraftNavEnv(region_dir=region_dir, **kwargs)
    volume, start, waypoints = make_debug_bridge_course(seed=seed)
    return VoxelNavEnv(volume=volume, start=start, waypoints=waypoints, **kwargs)


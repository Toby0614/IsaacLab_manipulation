# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions for a minimal 'pure grasp' task.

These are intentionally small and easy to tune for bring-up:
- reach shaping (dense): negative distance EE -> object
- grasp bonus (sparse): +1 when grasped
- lift shaping (dense-ish): positive once object is grasped and lifted above a threshold
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .observations import object_grasped


def constant(env: "ManagerBasedRLEnv", value: float = 1.0) -> torch.Tensor:
    """Constant reward-rate term."""
    return torch.full((env.num_envs,), float(value), device=env.device, dtype=torch.float32)


# =============================================================================
# OFFICIAL ISAACLAB LIFT-STYLE REWARDS
# =============================================================================
# These are based on isaaclab_tasks/manager_based/manipulation/lift/mdp/rewards.py
# Key insight: NO explicit grasp detection - just reward object state directly!

def object_is_lifted(
    env: "ManagerBasedRLEnv",
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
) -> torch.Tensor:
    """Reward 1.0 if object is above minimal_height, 0.0 otherwise.
    
    NO grasp check! This is the official IsaacLab approach.
    The robot will learn to grasp because that's the only way to lift consistently.
    
    NOTE: minimal_height is in ENV-LOCAL frame (relative to table/ground).
    """
    obj: RigidObject = env.scene[object_cfg.name]
    # Convert to env-local frame
    obj_z = obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return torch.where(obj_z > float(minimal_height), 1.0, 0.0)


def object_ee_distance_tanh(
    env: "ManagerBasedRLEnv",
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward for EE being close to object: 1 - tanh(distance / std).
    
    Returns value in [0, 1]. Uses tanh kernel (official IsaacLab style).
    """
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    
    cube_pos_w = obj.data.root_pos_w
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    
    distance = torch.norm(cube_pos_w - ee_w, dim=1)
    return 1.0 - torch.tanh(distance / float(std))


def object_goal_distance_tanh(
    env: "ManagerBasedRLEnv",
    std: float,
    minimal_height: float,
    goal_pos: tuple[float, float, float],
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
) -> torch.Tensor:
    """Reward for object being close to goal, ONLY when lifted.
    
    reward = (height > minimal_height) * (1 - tanh(distance / std))
    
    This is the official IsaacLab approach - gate by height, not grasp.
    
    NOTE: All positions are in ENV-LOCAL frame (not world frame!)
    """
    obj: RigidObject = env.scene[object_cfg.name]
    # CRITICAL: Convert to env-local frame!
    obj_pos = obj.data.root_pos_w - env.scene.env_origins
    
    # Check if lifted (in env-local Z)
    lifted = obj_pos[:, 2] > float(minimal_height)
    
    # Distance to goal (3D, in env-local frame)
    goal = obj_pos.new_tensor(goal_pos).unsqueeze(0)
    distance = torch.norm(obj_pos - goal, dim=1)
    
    goal_reward = 1.0 - torch.tanh(distance / float(std))
    return lifted.to(dtype=goal_reward.dtype) * goal_reward


def pickplace_success_bonus(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    goal_pos: tuple[float, float, float] = (0.7, 0.2, 0.0203),
    goal_half_extents_xy: tuple[float, float] = (0.05, 0.05),
    table_z: float = 0.0203,
    place_height_tol: float = 0.03,
    robot_name: str = "robot",
    open_thresh: float = 0.5,
) -> torch.Tensor:
    """Success bonus: 1.0 if object is in goal region, on table, and gripper open."""
    obj: RigidObject = env.scene[object_cfg.name]
    pos = obj.data.root_pos_w - env.scene.env_origins
    
    # XY in goal
    gx, gy = float(goal_pos[0]), float(goal_pos[1])
    hx, hy = float(goal_half_extents_xy[0]), float(goal_half_extents_xy[1])
    in_xy = torch.logical_and(
        torch.abs(pos[:, 0] - gx) <= hx,
        torch.abs(pos[:, 1] - gy) <= hy
    )
    
    # Height near table
    h = pos[:, 2] - float(table_z)
    height_ok = torch.abs(h) < float(place_height_tol)
    
    # Gripper open
    open_frac = _gripper_open_fraction(env, robot_name=robot_name)
    opened = open_frac > float(open_thresh)
    
    success = torch.logical_and(torch.logical_and(in_xy, height_ok), opened)
    return success.to(dtype=torch.float32)


def _gripper_open_fraction(env: "ManagerBasedRLEnv", robot_name: str = "robot") -> torch.Tensor:
    """Approximate gripper opening fraction in [0,1] for parallel grippers (Franka hand)."""
    robot = env.scene[robot_name]
    if not hasattr(env.cfg, "gripper_joint_names") or not hasattr(env.cfg, "gripper_open_val"):
        # Conservative fallback: treat as closed (encourages opening only when explicitly rewarded)
        return torch.zeros((env.num_envs,), device=env.device, dtype=torch.float32)
    joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
    if len(joint_ids) < 1:
        return torch.zeros((env.num_envs,), device=env.device, dtype=torch.float32)
    q = robot.data.joint_pos[:, joint_ids]
    open_val = float(env.cfg.gripper_open_val)
    open_val = max(open_val, 1e-6)
    # For Panda, both finger joints are typically positive and equal.
    opening = torch.mean(torch.abs(q), dim=1)
    return torch.clamp(opening / open_val, 0.0, 1.0)


def _cube_height_above_table(env: "ManagerBasedRLEnv", object_name: str = "cube_2", table_z: float = 0.0203) -> torch.Tensor:
    """Cube height above table plane (meters), computed in env frame."""
    obj: RigidObject = env.scene[object_name]
    z = obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return z - float(table_z)


def _in_goal_xy(
    env: "ManagerBasedRLEnv",
    object_name: str,
    goal_pos: tuple[float, float, float],
    goal_half_extents_xy: tuple[float, float] = (0.05, 0.05),
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_name]
    pos = obj.data.root_pos_w - env.scene.env_origins
    gx, gy = float(goal_pos[0]), float(goal_pos[1])
    hx, hy = float(goal_half_extents_xy[0]), float(goal_half_extents_xy[1])
    return torch.logical_and(torch.abs(pos[:, 0] - gx) <= hx, torch.abs(pos[:, 1] - gy) <= hy)


def reach_exp(
    env: "ManagerBasedRLEnv",
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    sigma: float = 0.08,
) -> torch.Tensor:
    """Dense reach reward: exp(-distance/sigma) in [0,1]."""
    dist = ee_object_distance(env, ee_frame_cfg=ee_frame_cfg, object_cfg=object_cfg)
    return torch.exp(-dist / max(float(sigma), 1e-6))


def close_when_near(
    env: "ManagerBasedRLEnv",
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    robot_name: str = "robot",
    sigma: float = 0.05,
) -> torch.Tensor:
    """Reward closing gripper only when near the target cube (bootstraps grasp discovery)."""
    near = torch.exp(-ee_object_distance(env, ee_frame_cfg=ee_frame_cfg, object_cfg=object_cfg) / max(float(sigma), 1e-6))
    open_frac = _gripper_open_fraction(env, robot_name=robot_name)
    return near * (1.0 - open_frac)


def grasp_hold(
    env: "ManagerBasedRLEnv",
    robot_name: str = "robot",
    ee_frame_name: str = "ee_frame",
    object_name: str = "cube_2",
    diff_threshold: float = 0.06,
) -> torch.Tensor:
    """Small per-step reward for holding a valid grasp (should not dominate the task objective)."""
    g = object_grasped(
        env,
        robot_cfg=SceneEntityCfg(robot_name),
        ee_frame_cfg=SceneEntityCfg(ee_frame_name),
        object_cfg=SceneEntityCfg(object_name),
        diff_threshold=diff_threshold,
    ).to(dtype=torch.float32)
    return g


class GraspStartBonusTerm(ManagerTermBase):
    """One-time bonus when a grasp is first achieved (transition 0->1)."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._prev = torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)
        self._given = torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._prev[env_ids] = False
        self._given[env_ids] = False

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        robot_name: str = "robot",
        ee_frame_name: str = "ee_frame",
        object_name: str = "cube_2",
        diff_threshold: float = 0.06,
    ) -> torch.Tensor:
        now = object_grasped(
            env,
            robot_cfg=SceneEntityCfg(robot_name),
            ee_frame_cfg=SceneEntityCfg(ee_frame_name),
            object_cfg=SceneEntityCfg(object_name),
            diff_threshold=diff_threshold,
        )
        # Only pay ONCE per episode to avoid "flicker farming" where grasp toggles on/off many times.
        first_event = torch.logical_and(now, torch.logical_not(self._prev))
        first_event = torch.logical_and(first_event, torch.logical_not(self._given))
        event = first_event.to(dtype=torch.float32)
        self._given |= first_event
        self._prev = now
        # dt-cancel so weight becomes fixed bonus
        dt = float(getattr(env, "step_dt", 0.0) or 0.0)
        return event / max(dt, 1e-6)


def lift_to_carry(
    env: "ManagerBasedRLEnv",
    robot_name: str = "robot",
    ee_frame_name: str = "ee_frame",
    object_name: str = "cube_2",
    diff_threshold: float = 0.06,
    h_carry: float = 0.12,
    table_z: float = 0.0203,
) -> torch.Tensor:
    """Lift shaping: grasp * clip(h / h_carry, 0, 1)."""
    g = grasp_hold(env, robot_name=robot_name, ee_frame_name=ee_frame_name, object_name=object_name, diff_threshold=diff_threshold)
    h = torch.clamp(_cube_height_above_table(env, object_name=object_name, table_z=table_z), min=0.0)
    return g * torch.clamp(h / max(float(h_carry), 1e-6), 0.0, 1.0)


def drag_penalty_when_grasped(
    env: "ManagerBasedRLEnv",
    robot_name: str = "robot",
    ee_frame_name: str = "ee_frame",
    object_name: str = "cube_2",
    diff_threshold: float = 0.06,
    table_z: float = 0.0203,
    sigma: float = 0.02,
) -> torch.Tensor:
    """Penalty magnitude when grasped but still near the table (discourages dragging/pushing while 'grasping')."""
    g = grasp_hold(env, robot_name=robot_name, ee_frame_name=ee_frame_name, object_name=object_name, diff_threshold=diff_threshold)
    h = torch.clamp(_cube_height_above_table(env, object_name=object_name, table_z=table_z), min=0.0)
    return g * torch.exp(-h / max(float(sigma), 1e-6))


def carry_to_goal_xy(
    env: "ManagerBasedRLEnv",
    object_name: str = "cube_2",
    goal_pos: tuple[float, float, float] = (0.70, 0.20, 0.0203),
    table_z: float = 0.0203,
    lift_on: float = 0.03,
    lift_full: float = 0.06,
    sigma_goal: float = 0.15,
) -> torch.Tensor:
    """Carry shaping gated by lift height (prevents pushing on table).

    g_lift = clamp((h - lift_on)/(lift_full-lift_on), 0, 1)
    r = g_lift * exp(-d_goal_xy/sigma_goal)
    """
    obj: RigidObject = env.scene[object_name]
    pos = obj.data.root_pos_w - env.scene.env_origins
    goal = pos.new_tensor(goal_pos).unsqueeze(0)
    d_goal_xy = torch.linalg.vector_norm(pos[:, :2] - goal[:, :2], dim=1)
    h = _cube_height_above_table(env, object_name=object_name, table_z=table_z)
    g_lift = torch.clamp((h - float(lift_on)) / max(float(lift_full - lift_on), 1e-6), 0.0, 1.0)
    return g_lift * torch.exp(-d_goal_xy / max(float(sigma_goal), 1e-6))


def carry_to_goal_xy_when_grasped(
    env: "ManagerBasedRLEnv",
    robot_name: str = "robot",
    ee_frame_name: str = "ee_frame",
    object_name: str = "cube_2",
    diff_threshold: float = 0.06,
    goal_pos: tuple[float, float, float] = (0.70, 0.20, 0.0203),
    table_z: float = 0.0203,
    lift_on: float = 0.03,
    lift_full: float = 0.06,
    sigma_goal: float = 0.15,
) -> torch.Tensor:
    """Carry shaping gated by (grasp AND lift height).

    This is safer than height-only gating: it won't pay for "bump the cube upward then push it".
    """
    g = grasp_hold(env, robot_name=robot_name, ee_frame_name=ee_frame_name, object_name=object_name, diff_threshold=diff_threshold)
    return g * carry_to_goal_xy(
        env,
        object_name=object_name,
        goal_pos=goal_pos,
        table_z=table_z,
        lift_on=lift_on,
        lift_full=lift_full,
        sigma_goal=sigma_goal,
    )


def carry_then_lower_to_place_when_grasped(
    env: "ManagerBasedRLEnv",
    robot_name: str = "robot",
    ee_frame_name: str = "ee_frame",
    object_name: str = "cube_2",
    diff_threshold: float = 0.06,
    goal_pos: tuple[float, float, float] = (0.70, 0.20, 0.0203),
    # height references
    table_z: float = 0.0203,
    transport_height: float = 0.06,
    # gating: require the cube to actually be off the table before paying carry/placing rewards
    lift_on: float = 0.03,
    # switching behavior: far from goal -> prefer transport height; near goal -> prefer near-table height
    switch_dist_xy: float = 0.10,
    switch_temp: float = 0.03,
    # shaping scales
    sigma_xy: float = 0.15,
    sigma_h: float = 0.03,
) -> torch.Tensor:
    """Combined shaping that fixes the common 'hover above goal forever' local optimum.

    Key idea:
    - While far from goal (XY), we prefer a safe *transport* height.
    - Once near the goal in XY, we smoothly shift the preferred height down toward the table (place height).
      This creates a *direct incentive to lower* after arriving over the goal, instead of hovering.

    Reward (when grasped) is roughly:
        gate(grasp) * gate(lift) * exp(-d_xy/sigma_xy) * exp(-|h - h_des|/sigma_h)

    Where h_des transitions from transport_height -> 0 as d_xy goes from >switch_dist to <switch_dist.
    """
    # --- grasp gate (binary-ish) ---
    g = grasp_hold(env, robot_name=robot_name, ee_frame_name=ee_frame_name, object_name=object_name, diff_threshold=diff_threshold)

    # --- positions ---
    obj: RigidObject = env.scene[object_name]
    pos = obj.data.root_pos_w - env.scene.env_origins
    goal = pos.new_tensor(goal_pos).unsqueeze(0)
    d_xy = torch.linalg.vector_norm(pos[:, :2] - goal[:, :2], dim=1)  # (B,)

    # --- height above table ---
    h = _cube_height_above_table(env, object_name=object_name, table_z=table_z)  # (B,)

    # --- lift gate to prevent reward farming by pushing on the table ---
    lifted = (h > float(lift_on)).to(dtype=torch.float32)

    # --- smooth switch: far => 1, near => 0 ---
    # w ~ 1 when d_xy >> switch_dist, w ~ 0 when d_xy << switch_dist
    w = torch.sigmoid((d_xy - float(switch_dist_xy)) / max(float(switch_temp), 1e-6))
    h_des = w * float(transport_height)  # near goal -> 0

    # --- shaping terms ---
    r_xy = torch.exp(-d_xy / max(float(sigma_xy), 1e-6))
    r_h = torch.exp(-torch.abs(h - h_des) / max(float(sigma_h), 1e-6))

    return g * lifted * r_xy * r_h


def hover_penalty_in_goal_xy_when_grasped(
    env: "ManagerBasedRLEnv",
    robot_name: str = "robot",
    ee_frame_name: str = "ee_frame",
    object_name: str = "cube_2",
    diff_threshold: float = 0.06,
    goal_pos: tuple[float, float, float] = (0.70, 0.20, 0.0203),
    goal_half_extents_xy: tuple[float, float] = (0.05, 0.05),
    table_z: float = 0.0203,
    # penalize keeping the cube unnecessarily high while already over the goal region
    hover_height: float = 0.05,
) -> torch.Tensor:
    """Penalty that removes the 'park above goal' attractor.

    Returns a non-negative scalar (meters above hover_height) when:
    - cube is inside goal XY region
    - cube is considered grasp-held

    You should use this with a NEGATIVE weight in the reward config.
    """
    g = grasp_hold(env, robot_name=robot_name, ee_frame_name=ee_frame_name, object_name=object_name, diff_threshold=diff_threshold)

    obj: RigidObject = env.scene[object_name]
    pos = obj.data.root_pos_w - env.scene.env_origins
    gx, gy = float(goal_pos[0]), float(goal_pos[1])
    hx, hy = float(goal_half_extents_xy[0]), float(goal_half_extents_xy[1])
    in_xy = torch.logical_and(torch.abs(pos[:, 0] - gx) <= hx, torch.abs(pos[:, 1] - gy) <= hy).to(dtype=torch.float32)

    h = _cube_height_above_table(env, object_name=object_name, table_z=table_z)
    return g * in_xy * torch.clamp(h - float(hover_height), min=0.0)


class HeightProgressWhenGraspedTerm(ManagerTermBase):
    """Potential-based lift progress: reward positive delta height while grasped (no 'parking' reward)."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._prev_h = torch.zeros((env.num_envs,), device=env.device, dtype=torch.float32)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        object_name = str(self.cfg.params.get("object_name", "cube_2"))
        table_z = float(self.cfg.params.get("table_z", 0.0203))
        h = _cube_height_above_table(self._env, object_name=object_name, table_z=table_z)
        self._prev_h[env_ids] = h[env_ids]

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        robot_name: str = "robot",
        ee_frame_name: str = "ee_frame",
        object_name: str = "cube_2",
        diff_threshold: float = 0.06,
        table_z: float = 0.0203,
        max_up_rate: float = 0.25,
    ) -> torch.Tensor:
        g = grasp_hold(env, robot_name=robot_name, ee_frame_name=ee_frame_name, object_name=object_name, diff_threshold=diff_threshold)
        h = _cube_height_above_table(env, object_name=object_name, table_z=table_z)
        dt = float(getattr(env, "step_dt", 0.0) or 0.0)
        rate = (h - self._prev_h) / max(dt, 1e-6)  # m/s
        self._prev_h = h
        # reward only upward progress; clamp to avoid spikes
        return g * torch.clamp(rate, min=0.0, max=float(max_up_rate))


class GoalProgressAfterLiftWhenGraspedTerm(ManagerTermBase):
    """Potential-based goal progress: reward reduction in goal XY distance once grasped+lifted."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._prev_d = torch.zeros((env.num_envs,), device=env.device, dtype=torch.float32)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        object_name = str(self.cfg.params.get("object_name", "cube_2"))
        goal_pos = tuple(self.cfg.params.get("goal_pos", (0.70, 0.20, 0.0203)))
        obj: RigidObject = self._env.scene[object_name]
        pos = obj.data.root_pos_w - self._env.scene.env_origins
        goal = pos.new_tensor(goal_pos).unsqueeze(0)
        d = torch.linalg.vector_norm(pos[:, :2] - goal[:, :2], dim=1)
        self._prev_d[env_ids] = d[env_ids]

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        robot_name: str = "robot",
        ee_frame_name: str = "ee_frame",
        object_name: str = "cube_2",
        diff_threshold: float = 0.06,
        goal_pos: tuple[float, float, float] = (0.70, 0.20, 0.0203),
        table_z: float = 0.0203,
        lift_on: float = 0.03,
        max_prog_rate: float = 0.25,
    ) -> torch.Tensor:
        # gate by grasp + lift height (prevents paying for pushing)
        g = grasp_hold(env, robot_name=robot_name, ee_frame_name=ee_frame_name, object_name=object_name, diff_threshold=diff_threshold)
        h = _cube_height_above_table(env, object_name=object_name, table_z=table_z)
        gate = g * (h > float(lift_on)).to(dtype=torch.float32)

        obj: RigidObject = env.scene[object_name]
        pos = obj.data.root_pos_w - env.scene.env_origins
        goal = pos.new_tensor(goal_pos).unsqueeze(0)
        d = torch.linalg.vector_norm(pos[:, :2] - goal[:, :2], dim=1)

        dt = float(getattr(env, "step_dt", 0.0) or 0.0)
        prog_rate = (self._prev_d - d) / max(dt, 1e-6)  # m/s improvement
        self._prev_d = d
        # reward only positive progress; clamp to avoid spikes
        return gate * torch.clamp(prog_rate, min=0.0, max=float(max_prog_rate))


class PlaceHeightErrorProgressInGoalTerm(ManagerTermBase):
    """Potential-based place/lower progress: reward reducing |height-above-table| while in goal.

    This directly teaches the missing sub-skill you're seeing in logs:
    - agent lifts + carries, but doesn't reliably lower/set down near the table in the goal region.

    Reward is only paid when:
    - cube is grasped (heuristic)
    - cube has been lifted above `lift_on` at least once this episode (tracked internally)
    - cube is inside goal XY region
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._ever_lifted = torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)
        self._prev_err = torch.zeros((env.num_envs,), device=env.device, dtype=torch.float32)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        object_name = str(self.cfg.params.get("object_name", "cube_2"))
        table_z = float(self.cfg.params.get("table_z", 0.0203))
        h = _cube_height_above_table(self._env, object_name=object_name, table_z=table_z)
        self._ever_lifted[env_ids] = False
        self._prev_err[env_ids] = torch.abs(h[env_ids])

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        robot_name: str = "robot",
        ee_frame_name: str = "ee_frame",
        object_name: str = "cube_2",
        diff_threshold: float = 0.06,
        goal_pos: tuple[float, float, float] = (0.70, 0.20, 0.0203),
        goal_half_extents_xy: tuple[float, float] = (0.05, 0.05),
        table_z: float = 0.0203,
        lift_on: float = 0.03,
        max_prog_rate: float = 0.25,
    ) -> torch.Tensor:
        g = grasp_hold(env, robot_name=robot_name, ee_frame_name=ee_frame_name, object_name=object_name, diff_threshold=diff_threshold)
        h = _cube_height_above_table(env, object_name=object_name, table_z=table_z)

        # Update lift history (only counts if grasped).
        self._ever_lifted |= torch.logical_and(g > 0.5, h > float(lift_on))

        in_goal = _in_goal_xy(env, object_name=object_name, goal_pos=goal_pos, goal_half_extents_xy=goal_half_extents_xy)
        gate = (g > 0.5).to(torch.float32) * self._ever_lifted.to(torch.float32) * in_goal.to(torch.float32)

        err = torch.abs(h)
        dt = float(getattr(env, "step_dt", 0.0) or 0.0)
        prog_rate = (self._prev_err - err) / max(dt, 1e-6)
        self._prev_err = err
        return gate * torch.clamp(prog_rate, min=0.0, max=float(max_prog_rate))


def hover_high_in_goal_penalty(
    env: "ManagerBasedRLEnv",
    robot_name: str = "robot",
    ee_frame_name: str = "ee_frame",
    object_name: str = "cube_2",
    diff_threshold: float = 0.06,
    goal_pos: tuple[float, float, float] = (0.70, 0.20, 0.0203),
    goal_half_extents_xy: tuple[float, float] = (0.05, 0.05),
    table_z: float = 0.0203,
    hover_thresh: float = 0.06,
) -> torch.Tensor:
    """Penalty for hovering too high while in the goal region (encourages lowering before release)."""
    g = grasp_hold(env, robot_name=robot_name, ee_frame_name=ee_frame_name, object_name=object_name, diff_threshold=diff_threshold)
    in_goal = _in_goal_xy(env, object_name=object_name, goal_pos=goal_pos, goal_half_extents_xy=goal_half_extents_xy)
    h = _cube_height_above_table(env, object_name=object_name, table_z=table_z)
    viol = torch.relu(h - float(hover_thresh))
    return in_goal.to(torch.float32) * g * viol


class ReleaseEventBonusWhenReadyTerm(ManagerTermBase):
    """One-time bonus when the gripper opens while 'ready' (in goal, near table, stable).

    This is often the missing trigger for PPO: without an explicit event bonus, it can learn to
    *never* open because opening is a risky discontinuous action.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._given = torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)
        self._prev_open = torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)
        self._ever_lifted = torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._given[env_ids] = False
        self._prev_open[env_ids] = False
        self._ever_lifted[env_ids] = False

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        robot_name: str = "robot",
        ee_frame_name: str = "ee_frame",
        object_name: str = "cube_2",
        diff_threshold: float = 0.06,
        lift_on: float = 0.03,
        goal_pos: tuple[float, float, float] = (0.70, 0.20, 0.0203),
        goal_half_extents_xy: tuple[float, float] = (0.05, 0.05),
        table_z: float = 0.0203,
        place_height_tol: float = 0.03,
        vel_thresh: float = 0.10,
        open_event_thresh: float = 0.8,
    ) -> torch.Tensor:
        # track lift history (so we don't reward opening after pushing)
        g = grasp_hold(env, robot_name=robot_name, ee_frame_name=ee_frame_name, object_name=object_name, diff_threshold=diff_threshold)
        h = _cube_height_above_table(env, object_name=object_name, table_z=table_z)
        self._ever_lifted |= torch.logical_and(g > 0.5, h > float(lift_on))

        in_goal = _in_goal_xy(env, object_name=object_name, goal_pos=goal_pos, goal_half_extents_xy=goal_half_extents_xy)
        height_ok = torch.abs(h) < float(place_height_tol)
        obj: RigidObject = env.scene[object_name]
        v = torch.linalg.vector_norm(obj.data.root_lin_vel_w, dim=1)
        stable = v < float(vel_thresh)
        ready = torch.logical_and(torch.logical_and(in_goal, height_ok), stable)
        ready = torch.logical_and(ready, self._ever_lifted)

        open_frac = _gripper_open_fraction(env, robot_name=robot_name)
        open_now = open_frac > float(open_event_thresh)

        # first time we cross into "open" while ready
        event = torch.logical_and(open_now, torch.logical_not(self._prev_open))
        event = torch.logical_and(event, ready)
        event = torch.logical_and(event, torch.logical_not(self._given))

        self._given |= event
        self._prev_open = open_now

        dt = float(getattr(env, "step_dt", 0.0) or 0.0)
        return event.to(torch.float32) / max(dt, 1e-6)


def stable_in_goal_before_release(
    env: "ManagerBasedRLEnv",
    object_name: str = "cube_2",
    goal_pos: tuple[float, float, float] = (0.70, 0.20, 0.0203),
    goal_half_extents_xy: tuple[float, float] = (0.05, 0.05),
    table_z: float = 0.0203,
    place_height_tol: float = 0.015,
    vel_sigma: float = 0.25,
) -> torch.Tensor:
    """Reward being in goal region and stable near table height (encourages 'set down, settle')."""
    in_goal = _in_goal_xy(env, object_name=object_name, goal_pos=goal_pos, goal_half_extents_xy=goal_half_extents_xy)
    h = _cube_height_above_table(env, object_name=object_name, table_z=table_z)
    h_ok = torch.exp(-torch.abs(h) / max(float(place_height_tol), 1e-6))
    obj: RigidObject = env.scene[object_name]
    v = torch.linalg.vector_norm(obj.data.root_lin_vel_w, dim=1)
    stable = torch.exp(-v / max(float(vel_sigma), 1e-6))
    return in_goal.to(dtype=torch.float32) * h_ok * stable


def release_when_ready(
    env: "ManagerBasedRLEnv",
    object_name: str = "cube_2",
    goal_pos: tuple[float, float, float] = (0.70, 0.20, 0.0203),
    goal_half_extents_xy: tuple[float, float] = (0.05, 0.05),
    table_z: float = 0.0203,
    place_height_tol: float = 0.015,
    vel_thresh: float = 0.05,
    robot_name: str = "robot",
) -> torch.Tensor:
    """Explicit reward for opening the gripper when the cube is correctly placed and stable."""
    in_goal = _in_goal_xy(env, object_name=object_name, goal_pos=goal_pos, goal_half_extents_xy=goal_half_extents_xy)
    h = _cube_height_above_table(env, object_name=object_name, table_z=table_z)
    height_ok = torch.abs(h) < float(place_height_tol)
    obj: RigidObject = env.scene[object_name]
    v = torch.linalg.vector_norm(obj.data.root_lin_vel_w, dim=1)
    stable = v < float(vel_thresh)
    ready = torch.logical_and(torch.logical_and(in_goal, height_ok), stable)
    open_frac = _gripper_open_fraction(env, robot_name=robot_name)
    return ready.to(dtype=torch.float32) * open_frac


def bad_release_penalty(
    env: "ManagerBasedRLEnv",
    object_name: str = "cube_2",
    goal_pos: tuple[float, float, float] = (0.70, 0.20, 0.0203),
    goal_half_extents_xy: tuple[float, float] = (0.05, 0.05),
    robot_name: str = "robot",
    table_z: float = 0.0203,
    open_thresh: float = 0.8,
    height_thresh: float = 0.06,
) -> torch.Tensor:
    """Penalty for opening the gripper while cube is high and not in goal (discourages random releases)."""
    in_goal = _in_goal_xy(env, object_name=object_name, goal_pos=goal_pos, goal_half_extents_xy=goal_half_extents_xy)
    h = _cube_height_above_table(env, object_name=object_name, table_z=table_z)
    open_frac = _gripper_open_fraction(env, robot_name=robot_name)
    bad = torch.logical_and(open_frac > float(open_thresh), torch.logical_and(h > float(height_thresh), torch.logical_not(in_goal)))
    return bad.to(dtype=torch.float32)


class EarlyReleaseInGoalPenaltyTerm(ManagerTermBase):
    """Penalty for opening the gripper in the goal region while the cube is still too high.

    Why this matters (from your logs):
    - You started getting non-zero `release_event_bonus`, but `success_grasp` stayed low / fell.
    - That often means the policy opens "kinda near goal" but still above the table or not settled.

    We only activate this after the cube has been truly lifted once (to avoid penalizing opening at reset).
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._ever_lifted = torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._ever_lifted[env_ids] = False

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        robot_name: str = "robot",
        ee_frame_name: str = "ee_frame",
        object_name: str = "cube_2",
        diff_threshold: float = 0.06,
        lift_on: float = 0.03,
        goal_pos: tuple[float, float, float] = (0.70, 0.20, 0.0203),
        goal_half_extents_xy: tuple[float, float] = (0.05, 0.05),
        table_z: float = 0.0203,
        open_thresh: float = 0.8,
        allowed_height: float = 0.02,
    ) -> torch.Tensor:
        # Lift history (requires grasp heuristic).
        g = grasp_hold(env, robot_name=robot_name, ee_frame_name=ee_frame_name, object_name=object_name, diff_threshold=diff_threshold)
        h = _cube_height_above_table(env, object_name=object_name, table_z=table_z)
        self._ever_lifted |= torch.logical_and(g > 0.5, h > float(lift_on))

        in_goal = _in_goal_xy(env, object_name=object_name, goal_pos=goal_pos, goal_half_extents_xy=goal_half_extents_xy)
        open_frac = _gripper_open_fraction(env, robot_name=robot_name)
        opened = open_frac > float(open_thresh)

        # Penalize opening above allowed height while in goal, only after a real lift occurred.
        too_high = h > float(allowed_height)
        bad = torch.logical_and(self._ever_lifted, torch.logical_and(in_goal, torch.logical_and(opened, too_high)))
        return bad.to(torch.float32)


def slam_penalty_near_goal(
    env: "ManagerBasedRLEnv",
    object_name: str = "cube_2",
    goal_pos: tuple[float, float, float] = (0.70, 0.20, 0.0203),
    sigma_goal: float = 0.15,
    vz_thresh: float = 0.2,
) -> torch.Tensor:
    """Penalty for high downward velocity near the goal (discourage slamming)."""
    obj: RigidObject = env.scene[object_name]
    pos = obj.data.root_pos_w - env.scene.env_origins
    goal = pos.new_tensor(goal_pos).unsqueeze(0)
    d_goal_xy = torch.linalg.vector_norm(pos[:, :2] - goal[:, :2], dim=1)
    g_near = torch.exp(-d_goal_xy / max(float(sigma_goal), 1e-6))
    vz = obj.data.root_lin_vel_w[:, 2]
    slam = torch.relu(-vz - float(vz_thresh))
    return g_near * slam


class ActionSecondDifferenceL2Term(ManagerTermBase):
    """Temporal consistency penalty (second-order action smoothness).

    Penalizes: ||a_t - 2 a_{t-1} + a_{t-2}||^2

    This targets high-frequency oscillations better than a simple first-difference action-rate penalty.
    Use with a NEGATIVE weight in reward config.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        act_dim = int(sum(env.action_manager.action_term_dim))
        self._prev = torch.zeros((env.num_envs, act_dim), device=env.device, dtype=torch.float32)
        self._prev2 = torch.zeros((env.num_envs, act_dim), device=env.device, dtype=torch.float32)
        self._initialized = torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        # On reset, we don't know meaningful previous actions. Initialize from current action if available.
        a = self._env.action_manager.action
        if a is None:
            self._prev[env_ids] = 0.0
            self._prev2[env_ids] = 0.0
        else:
            self._prev[env_ids] = a[env_ids].detach()
            self._prev2[env_ids] = a[env_ids].detach()
        self._initialized[env_ids] = True

    def __call__(self, env: "ManagerBasedRLEnv") -> torch.Tensor:
        a = env.action_manager.action
        # If we haven't initialized yet (first call before reset), just return zeros and initialize.
        if not torch.all(self._initialized):
            missing = torch.logical_not(self._initialized)
            self._prev[missing] = a[missing].detach()
            self._prev2[missing] = a[missing].detach()
            self._initialized[missing] = True
            return torch.zeros((env.num_envs,), device=env.device, dtype=torch.float32)

        d2 = a - 2.0 * self._prev + self._prev2
        # shift history
        self._prev2 = self._prev
        self._prev = a.detach()
        return torch.sum(torch.square(d2), dim=1)


def object_motion_l2_when_grasped(
    env: "ManagerBasedRLEnv",
    robot_name: str = "robot",
    ee_frame_name: str = "ee_frame",
    object_name: str = "cube_2",
    diff_threshold: float = 0.06,
    table_z: float = 0.0203,
    lift_on: float = 0.02,
    ang_vel_scale: float = 0.2,
) -> torch.Tensor:
    """Penalty (positive scalar) for moving the object too fast while carrying.

    This is meant to reduce "wobbling"/vibration in the arm that shows up as object shake.

    - gated by a grasp heuristic AND a minimal lift height (to avoid penalizing pre-grasp motion)
    - penalizes linear speed squared + scaled angular speed squared

    Use with a NEGATIVE weight in reward config.
    """
    g = grasp_hold(
        env,
        robot_name=robot_name,
        ee_frame_name=ee_frame_name,
        object_name=object_name,
        diff_threshold=diff_threshold,
    )
    h = _cube_height_above_table(env, object_name=object_name, table_z=table_z)
    gate = g * (h > float(lift_on)).to(torch.float32)

    obj: RigidObject = env.scene[object_name]
    v = torch.linalg.vector_norm(obj.data.root_lin_vel_w, dim=1)
    w = torch.linalg.vector_norm(obj.data.root_ang_vel_w, dim=1)
    return gate * (v * v + float(ang_vel_scale) * (w * w))


def object_xy_speed_during_lift_penalty(
    env: "ManagerBasedRLEnv",
    robot_name: str = "robot",
    ee_frame_name: str = "ee_frame",
    object_name: str = "cube_2",
    diff_threshold: float = 0.06,
    table_z: float = 0.0203,
    lift_on: float = 0.01,
    lift_target: float = 0.06,
) -> torch.Tensor:
    """Penalty (positive scalar) for horizontal motion while lifting.

    Encourages the behavior you described: "lift straight up" after grasp,
    instead of immediately swinging/dragging sideways.

    Active only when:
    - grasped (heuristic)
    - height is between lift_on and lift_target (i.e., during the lift phase)

    Use with a NEGATIVE weight in reward config.
    """
    g = grasp_hold(
        env,
        robot_name=robot_name,
        ee_frame_name=ee_frame_name,
        object_name=object_name,
        diff_threshold=diff_threshold,
    )
    h = _cube_height_above_table(env, object_name=object_name, table_z=table_z)
    in_lift_band = torch.logical_and(h > float(lift_on), h < float(lift_target))
    gate = g * in_lift_band.to(torch.float32)

    obj: RigidObject = env.scene[object_name]
    vxy2 = torch.sum(torch.square(obj.data.root_lin_vel_w[:, :2]), dim=1)
    return gate * vxy2

class OtherCubesDisplacementPenaltyTerm(ManagerTermBase):
    """Penalty for moving other cubes away from their reset positions (better than proximity penalties)."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._init_xy = None

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        other_names = tuple(self.cfg.params.get("other_names", ("cube_1", "cube_3")))
        if self._init_xy is None:
            # allocate for all envs
            self._init_xy = torch.zeros((self._env.num_envs, len(other_names), 2), device=self._env.device)
        # compute current positions for requested env_ids
        for k, name in enumerate(other_names):
            obj: RigidObject = self._env.scene[name]
            pos = obj.data.root_pos_w - self._env.scene.env_origins
            self._init_xy[env_ids, k, :] = pos[env_ids, :2]

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        other_names: tuple[str, ...] = ("cube_1", "cube_3"),
        slack: float = 0.02,
    ) -> torch.Tensor:
        if self._init_xy is None:
            # first call before reset; initialize for all envs
            self.reset()
        curr = []
        for name in other_names:
            obj: RigidObject = env.scene[name]
            pos = obj.data.root_pos_w - env.scene.env_origins
            curr.append(pos[:, :2])
        curr_xy = torch.stack(curr, dim=1)  # (B, K, 2)
        d = torch.linalg.vector_norm(curr_xy - self._init_xy, dim=-1)  # (B, K)
        viol = torch.relu(d - float(slack))
        return torch.sum(viol, dim=1)


def other_cubes_in_goal_count(
    env: "ManagerBasedRLEnv",
    other_names: tuple[str, ...] = ("cube_1", "cube_3"),
    goal_pos: tuple[float, float, float] = (0.70, 0.20, 0.0203),
    goal_half_extents_xy: tuple[float, float] = (0.05, 0.05),
) -> torch.Tensor:
    """Count how many non-target cubes are inside the goal XY region (penalize disturbances)."""
    counts = torch.zeros((env.num_envs,), device=env.device, dtype=torch.float32)
    for name in other_names:
        in_goal = _in_goal_xy(env, object_name=name, goal_pos=goal_pos, goal_half_extents_xy=goal_half_extents_xy)
        counts += in_goal.to(dtype=torch.float32)
    return counts


class StableInGoalAfterLiftTerm(ManagerTermBase):
    """Stateful variant of `stable_in_goal_before_release` that only pays after a true lift happened.

    This prevents the reward-hack you observed where the policy learns to *push* the cube into the goal
    (no grasp / no lift) and still collects large 'stable in goal' and 'release' rewards.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._ever_lifted = torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._ever_lifted[env_ids] = False

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        robot_name: str = "robot",
        ee_frame_name: str = "ee_frame",
        object_name: str = "cube_2",
        diff_threshold: float = 0.06,
        lift_height_thresh: float = 0.03,
        goal_pos: tuple[float, float, float] = (0.70, 0.20, 0.0203),
        goal_half_extents_xy: tuple[float, float] = (0.05, 0.05),
        table_z: float = 0.0203,
        place_height_tol: float = 0.015,
        vel_sigma: float = 0.25,
    ) -> torch.Tensor:
        # Update lift history (lift must happen while "grasped" by the current detector).
        grasped = object_grasped(
            env,
            robot_cfg=SceneEntityCfg(robot_name),
            ee_frame_cfg=SceneEntityCfg(ee_frame_name),
            object_cfg=SceneEntityCfg(object_name),
            diff_threshold=diff_threshold,
        )
        h = _cube_height_above_table(env, object_name=object_name, table_z=table_z)
        self._ever_lifted |= torch.logical_and(grasped, h > float(lift_height_thresh))

        # Original "stable in goal near table" shaping.
        in_goal = _in_goal_xy(env, object_name=object_name, goal_pos=goal_pos, goal_half_extents_xy=goal_half_extents_xy)
        h_ok = torch.exp(-torch.abs(h) / max(float(place_height_tol), 1e-6))
        obj: RigidObject = env.scene[object_name]
        v = torch.linalg.vector_norm(obj.data.root_lin_vel_w, dim=1)
        stable = torch.exp(-v / max(float(vel_sigma), 1e-6))
        shaped = in_goal.to(dtype=torch.float32) * h_ok * stable

        return shaped * self._ever_lifted.to(dtype=torch.float32)


class ReleaseWhenReadyAfterLiftTerm(ManagerTermBase):
    """Stateful variant of `release_when_ready` that only pays after a true lift happened."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._ever_lifted = torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._ever_lifted[env_ids] = False

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        robot_name: str = "robot",
        ee_frame_name: str = "ee_frame",
        object_name: str = "cube_2",
        diff_threshold: float = 0.06,
        lift_height_thresh: float = 0.03,
        goal_pos: tuple[float, float, float] = (0.70, 0.20, 0.0203),
        goal_half_extents_xy: tuple[float, float] = (0.05, 0.05),
        table_z: float = 0.0203,
        place_height_tol: float = 0.015,
        vel_thresh: float = 0.05,
    ) -> torch.Tensor:
        grasped = object_grasped(
            env,
            robot_cfg=SceneEntityCfg(robot_name),
            ee_frame_cfg=SceneEntityCfg(ee_frame_name),
            object_cfg=SceneEntityCfg(object_name),
            diff_threshold=diff_threshold,
        )
        h = _cube_height_above_table(env, object_name=object_name, table_z=table_z)
        self._ever_lifted |= torch.logical_and(grasped, h > float(lift_height_thresh))

        in_goal = _in_goal_xy(env, object_name=object_name, goal_pos=goal_pos, goal_half_extents_xy=goal_half_extents_xy)
        height_ok = torch.abs(h) < float(place_height_tol)
        obj: RigidObject = env.scene[object_name]
        v = torch.linalg.vector_norm(obj.data.root_lin_vel_w, dim=1)
        stable = v < float(vel_thresh)
        ready = torch.logical_and(torch.logical_and(in_goal, height_ok), stable)
        open_frac = _gripper_open_fraction(env, robot_name=robot_name)

        return ready.to(dtype=torch.float32) * open_frac * self._ever_lifted.to(dtype=torch.float32)


def undesired_contact_force_penalty(
    env: "ManagerBasedRLEnv",
    sensor_names: tuple[str, ...] = ("hand_contact_undesired", "leftfinger_contact_undesired", "rightfinger_contact_undesired"),
    threshold: float = 2.0,
) -> torch.Tensor:
    """Penalty for undesired contacts measured by ContactSensors with filtering enabled.

    This is intended to implement: "touching anything else BUT cube_2 is penalized".
    We achieve that by configuring the contact sensors with `filter_prim_paths_expr` that include
    only the forbidden objects (e.g., table, cube_1, cube_3, ground). Contacts with cube_2 are
    not reported by these sensors and therefore not penalized.

    Returns:
        Non-negative penalty magnitude (will be multiplied by a negative weight in reward cfg).
    """
    penalties = []
    thr = float(threshold)
    for name in sensor_names:
        sensor = env.scene.sensors[name]
        data = sensor.data

        # Prefer filtered force matrix if available, else fall back to net forces.
        forces = None
        if hasattr(data, "force_matrix_w_history"):
            # (B, H, num_bodies, num_filters, 3)
            forces = data.force_matrix_w_history
        elif hasattr(data, "force_matrix_w"):
            # (B, num_bodies, num_filters, 3)
            forces = data.force_matrix_w.unsqueeze(1)
        elif hasattr(data, "net_forces_w_history"):
            forces = data.net_forces_w_history.unsqueeze(3)  # (B, H, num_bodies, 1, 3)
        elif hasattr(data, "net_forces_w"):
            forces = data.net_forces_w.unsqueeze(1).unsqueeze(3)

        if forces is None:
            raise AttributeError(
                f"Contact sensor '{name}' does not expose known force fields. "
                "Expected `force_matrix_w(_history)` or `net_forces_w(_history)`."
            )

        # magnitude of force at each time/body/filter
        mags = torch.linalg.vector_norm(forces, dim=-1)  # (B, H, bodies, filters)
        # take max over history (detect any contact spike since last observation)
        mags = mags.max(dim=1).values  # (B, bodies, filters)
        # aggregate over bodies + filters
        mags = mags.reshape(mags.shape[0], -1).sum(dim=1)  # (B,)

        penalties.append(torch.clamp(mags - thr, min=0.0))

    if not penalties:
        return torch.zeros((env.num_envs,), device=env.device, dtype=torch.float32)
    return torch.stack(penalties, dim=0).sum(dim=0)


def joint_velocity_soft_limit_penalty(
    env: "ManagerBasedRLEnv",
    robot_name: str = "robot",
    soft_ratio: float = 0.85,
    max_per_joint: float = 1.0,
) -> torch.Tensor:
    """Soft penalty when joint velocities exceed a fraction of the (soft) joint velocity limits.

    Notes:
    - Uses privileged robot state (joint velocities + limits); does NOT require extra sensors.
    - `soft_ratio` is a fraction of the per-joint limits. Example: 0.85 means penalize above 85% of the limit.
    - Violation is clipped per-joint to avoid huge gradients.
    """
    robot = env.scene[robot_name]
    vel = robot.data.joint_vel
    limits = robot.data.soft_joint_vel_limits  # set from actuator velocity_limit (USD/PhysX if not overridden)

    # guard against weird zeros
    limits = torch.clamp(limits, min=1e-6)
    violation = torch.abs(vel) - limits * float(soft_ratio)
    violation = torch.clamp(violation, min=0.0, max=float(max_per_joint))
    return torch.sum(violation, dim=1)


def joint_effort_soft_limit_penalty(
    env: "ManagerBasedRLEnv",
    robot_name: str = "robot",
    soft_ratio: float = 0.85,
    max_per_joint: float = 1.0,
) -> torch.Tensor:
    """Soft penalty when (approx) applied joint efforts exceed a fraction of joint effort limits.

    For the Franka Panda in your setup, effort limits are explicitly configured via `effort_limit_sim`
    in the robot asset config (and also exist as PhysX joint max forces).
    """
    robot = env.scene[robot_name]
    tau = robot.data.applied_torque
    limits = robot.data.joint_effort_limits

    limits = torch.clamp(limits, min=1e-6)
    violation = torch.abs(tau) - limits * float(soft_ratio)
    violation = torch.clamp(violation, min=0.0, max=float(max_per_joint))
    return torch.sum(violation, dim=1)


def ee_object_distance(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
) -> torch.Tensor:
    """Distance between end-effector target frame and the object (meters)."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    obj_pos_w = obj.data.root_pos_w
    return torch.linalg.vector_norm(obj_pos_w - ee_pos_w, dim=1)


def reach_shaping(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    distance_scale: float = 1.0,
) -> torch.Tensor:
    """Dense shaping: negative EE->object distance."""
    dist = ee_object_distance(env, ee_frame_cfg=ee_frame_cfg, object_cfg=object_cfg)
    return -distance_scale * dist


def lift_shaping(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    min_height: float = 0.08,
    height_scale: float = 1.0,
) -> torch.Tensor:
    """Positive shaping once object is lifted above a threshold.

    Returns max(0, z - min_height) * height_scale.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    z = obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return torch.clamp(z - min_height, min=0.0) * height_scale


def lift_shaping_when_grasped_tanh(
    env: "ManagerBasedRLEnv",
    robot_name: str = "robot",
    ee_frame_name: str = "ee_frame",
    object_name: str = "cube_2",
    diff_threshold: float = 0.06,
    min_height: float = 0.08,
    height_scale: float = 0.10,
) -> torch.Tensor:
    """Bounded lift shaping (0..1), gated by grasp.

    This avoids a common exploit where the policy maximizes return by just holding the object up
    (or getting it into a high-z state) without ever placing it.
    """
    grasp = object_grasped(
        env,
        robot_cfg=SceneEntityCfg(robot_name),
        ee_frame_cfg=SceneEntityCfg(ee_frame_name),
        object_cfg=SceneEntityCfg(object_name),
        diff_threshold=diff_threshold,
    ).to(dtype=torch.float32)

    obj: RigidObject = env.scene[object_name]
    z = obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    h = torch.clamp(z - float(min_height), min=0.0)
    # Smooth saturation around ~height_scale meters above the threshold.
    shaped = torch.tanh(h / max(float(height_scale), 1e-6))
    return shaped * grasp


def place_shaping(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    goal_pos: tuple[float, float, float] = (0.70, 0.20, 0.0203),
    distance_scale: float = 1.0,
) -> torch.Tensor:
    """Dense shaping toward a placement goal (negative XY distance)."""
    obj: RigidObject = env.scene[object_cfg.name]
    obj_pos = obj.data.root_pos_w - env.scene.env_origins
    goal = obj_pos.new_tensor(goal_pos).unsqueeze(0)
    xy_dist = torch.linalg.vector_norm(obj_pos[:, :2] - goal[:, :2], dim=1)
    return -distance_scale * xy_dist


def place_shaping_when_grasped_exp(
    env: "ManagerBasedRLEnv",
    robot_name: str = "robot",
    ee_frame_name: str = "ee_frame",
    object_name: str = "cube_2",
    diff_threshold: float = 0.06,
    goal_pos: tuple[float, float, float] = (0.70, 0.20, 0.0203),
    distance_scale: float = 10.0,
) -> torch.Tensor:
    """Positive bounded shaping (0..1), higher when closer to goal, gated by grasp."""
    grasp = object_grasped(
        env,
        robot_cfg=SceneEntityCfg(robot_name),
        ee_frame_cfg=SceneEntityCfg(ee_frame_name),
        object_cfg=SceneEntityCfg(object_name),
        diff_threshold=diff_threshold,
    ).to(dtype=torch.float32)

    obj: RigidObject = env.scene[object_name]
    obj_pos = obj.data.root_pos_w - env.scene.env_origins
    goal = obj_pos.new_tensor(goal_pos).unsqueeze(0)
    xy_dist = torch.linalg.vector_norm(obj_pos[:, :2] - goal[:, :2], dim=1)
    shaped = torch.exp(-float(distance_scale) * xy_dist)
    return shaped * grasp


def termination_term_fixed_bonus(
    env: "ManagerBasedRLEnv",
    term_name: str = "success_grasp",
) -> torch.Tensor:
    """Fixed one-time bonus tied to a termination term (dt-cancelled).

    This guarantees reward and termination are aligned even if the success logic is stateful.
    """
    event = env.termination_manager.get_term(term_name).to(dtype=torch.float32)
    dt = float(getattr(env, "step_dt", 0.0) or 0.0)
    return event / max(dt, 1e-6)


class PickPlaceSuccessEventBonusTerm(ManagerTermBase):
    """One-time success bonus computed *directly* from state (not from termination manager).

    Why this exists:
    - Your training logs show `Episode_Termination/success_grasp` > 0 while `Episode_Reward/success_bonus` ~ 0.
    - This is a classic timing issue: reward is often computed before termination flags are finalized,
      so tying bonus to `termination_manager.get_term(...)` can under-pay or miss events.

    This term mirrors the success condition used in `PickPlaceReleaseSuccessWithLiftHistoryTerm`:
    - Must have ever grasped + ever lifted (while grasped)
    - Final state: in goal XY, near-table height, stable, gripper open
    It pays ONCE per episode when the condition first becomes true (dt-cancelled).
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._ever_grasped = torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)
        self._ever_lifted = torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)
        self._given = torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._ever_grasped[env_ids] = False
        self._ever_lifted[env_ids] = False
        self._given[env_ids] = False

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        robot_name: str = "robot",
        ee_frame_name: str = "ee_frame",
        object_name: str = "cube_2",
        diff_threshold: float = 0.06,
        lift_height_thresh: float = 0.03,
        goal_center: tuple[float, float, float] = (0.70, 0.20, 0.0203),
        goal_half_extents_xy: tuple[float, float] = (0.05, 0.05),
        table_z: float = 0.0203,
        place_height_tol: float = 0.02,
        vel_thresh: float = 0.07,
        open_thresh: float = 0.7,
    ) -> torch.Tensor:
        # update history
        grasped = object_grasped(
            env,
            robot_cfg=SceneEntityCfg(robot_name),
            ee_frame_cfg=SceneEntityCfg(ee_frame_name),
            object_cfg=SceneEntityCfg(object_name),
            diff_threshold=diff_threshold,
        )
        self._ever_grasped |= grasped

        obj: RigidObject = env.scene[object_name]
        pos = obj.data.root_pos_w - env.scene.env_origins
        h = pos[:, 2] - float(table_z)
        self._ever_lifted |= torch.logical_and(grasped, h > float(lift_height_thresh))

        # final state
        gx, gy = float(goal_center[0]), float(goal_center[1])
        hx, hy = float(goal_half_extents_xy[0]), float(goal_half_extents_xy[1])
        in_xy = torch.logical_and(torch.abs(pos[:, 0] - gx) <= hx, torch.abs(pos[:, 1] - gy) <= hy)
        height_ok = torch.abs(h) < float(place_height_tol)
        v = torch.linalg.vector_norm(obj.data.root_lin_vel_w, dim=1)
        stable = v < float(vel_thresh)
        open_frac = _gripper_open_fraction(env, robot_name=robot_name)
        opened = open_frac > float(open_thresh)

        ready = torch.logical_and(self._ever_grasped, self._ever_lifted)
        success_now = torch.logical_and(ready, torch.logical_and(torch.logical_and(in_xy, height_ok), torch.logical_and(stable, opened)))

        event = torch.logical_and(success_now, torch.logical_not(self._given))
        self._given |= event

        dt = float(getattr(env, "step_dt", 0.0) or 0.0)
        return event.to(torch.float32) / max(dt, 1e-6)


def proximity_penalty_to_other_cubes(
    env: ManagerBasedRLEnv,
    object_name: str = "cube_2",
    other_names: tuple[str, str] = ("cube_1", "cube_3"),
    threshold: float = 0.08,
    penalty_scale: float = 1.0,
) -> torch.Tensor:
    """Penalty when the target object is too close to other cubes (proxy for 'touching').

    This does NOT require contact sensors, and does not expose other-cube positions to the policy.
    It's computed from privileged state for reward shaping.

    Returns:
    - 0 when all distances >= threshold
    - negative value when any distance < threshold (linear in violation)
    """
    obj: RigidObject = env.scene[object_name]
    obj_pos = obj.data.root_pos_w - env.scene.env_origins

    min_dist = None
    for name in other_names:
        other: RigidObject = env.scene[name]
        other_pos = other.data.root_pos_w - env.scene.env_origins
        d = torch.linalg.vector_norm(obj_pos[:, :2] - other_pos[:, :2], dim=1)  # XY distance
        min_dist = d if min_dist is None else torch.minimum(min_dist, d)

    violation = torch.clamp(threshold - min_dist, min=0.0)
    return -penalty_scale * violation


def gate_by_boolean(reward: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Multiply reward by a boolean (0/1) gate."""
    return reward * gate.to(dtype=reward.dtype)


# =============================================================================
# ULTRA-SIMPLIFIED REWARD FUNCTIONS (FetchPickAndPlace style)
# =============================================================================

def lift_height_reward(
    env: "ManagerBasedRLEnv",
    robot_name: str = "robot",
    ee_frame_name: str = "ee_frame",
    object_name: str = "cube_2",
    diff_threshold: float = 0.06,
    table_z: float = 0.0203,
    max_height: float = 0.15,
) -> torch.Tensor:
    """Simple continuous height reward when grasped: reward = h / max_height.
    
    Returns 0 if not grasped, otherwise returns normalized height (0 to 1).
    This is the SIMPLEST possible lift reward - no thresholds, just "higher = better".
    
    WARNING: This reward NEVER stops growing with height! Use lift_to_transport_height instead.
    """
    grasped = object_grasped(
        env,
        robot_cfg=SceneEntityCfg(robot_name),
        ee_frame_cfg=SceneEntityCfg(ee_frame_name),
        object_cfg=SceneEntityCfg(object_name),
        diff_threshold=diff_threshold,
    )
    
    obj: RigidObject = env.scene[object_name]
    h = obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2] - float(table_z)
    h = torch.clamp(h, min=0.0, max=float(max_height))
    
    # Normalize to [0, 1]
    reward = h / float(max_height)
    
    return reward * grasped.to(dtype=reward.dtype)


def lift_to_transport_height(
    env: "ManagerBasedRLEnv",
    robot_name: str = "robot",
    ee_frame_name: str = "ee_frame",
    object_name: str = "cube_2",
    diff_threshold: float = 0.06,
    table_z: float = 0.0203,
    target_height: float = 0.06,
) -> torch.Tensor:
    """Lift reward that CAPS at a target transport height.
    
    Returns:
    - 0 if not grasped
    - h / target_height if h < target_height (incentivizes lifting)
    - 1.0 if h >= target_height (no further reward for going higher!)
    
    This prevents the robot from learning that "higher = always better".
    Once at transport height, the robot should focus on carrying to goal.
    """
    grasped = object_grasped(
        env,
        robot_cfg=SceneEntityCfg(robot_name),
        ee_frame_cfg=SceneEntityCfg(ee_frame_name),
        object_cfg=SceneEntityCfg(object_name),
        diff_threshold=diff_threshold,
    )
    
    obj: RigidObject = env.scene[object_name]
    h = obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2] - float(table_z)
    h = torch.clamp(h, min=0.0)
    
    # Cap at target height - no reward for going higher!
    reward = torch.clamp(h / float(target_height), max=1.0)
    
    return reward * grasped.to(dtype=reward.dtype)


def object_to_goal_distance_when_lifted(
    env: "ManagerBasedRLEnv",
    robot_name: str = "robot",
    ee_frame_name: str = "ee_frame",
    object_name: str = "cube_2",
    diff_threshold: float = 0.06,
    goal_pos: tuple[float, float, float] = (0.7, 0.2, 0.0203),
    table_z: float = 0.0203,
    min_lift: float = 0.02,
    sigma: float = 0.20,
) -> torch.Tensor:
    """Distance-to-goal reward (XY only), only active when grasped AND lifted.
    
    WARNING: This uses XY distance only and DEACTIVATES when lowering!
    Use object_to_goal_distance_3d_when_grasped instead for pick-and-place.
    
    reward = exp(-dist_xy^2 / sigma^2) when grasped and lifted, 0 otherwise.
    """
    grasped = object_grasped(
        env,
        robot_cfg=SceneEntityCfg(robot_name),
        ee_frame_cfg=SceneEntityCfg(ee_frame_name),
        object_cfg=SceneEntityCfg(object_name),
        diff_threshold=diff_threshold,
    )
    
    obj: RigidObject = env.scene[object_name]
    pos = obj.data.root_pos_w - env.scene.env_origins
    h = pos[:, 2] - float(table_z)
    
    # Gate by lift height
    lifted = h > float(min_lift)
    active = torch.logical_and(grasped, lifted)
    
    # Distance to goal (XY only - we want it over the goal, not necessarily at goal height)
    goal = pos.new_tensor(goal_pos[:2]).unsqueeze(0)
    dist_sq = torch.sum((pos[:, :2] - goal) ** 2, dim=1)
    reward = torch.exp(-dist_sq / (float(sigma) ** 2))
    
    return reward * active.to(dtype=reward.dtype)


def object_to_goal_distance_3d_when_grasped(
    env: "ManagerBasedRLEnv",
    robot_name: str = "robot",
    ee_frame_name: str = "ee_frame",
    object_name: str = "cube_2",
    diff_threshold: float = 0.06,
    goal_pos: tuple[float, float, float] = (0.7, 0.2, 0.0203),
    sigma: float = 0.15,
) -> torch.Tensor:
    """3D distance-to-goal reward when grasped (no lift gate!).
    
    THIS IS THE KEY FIX FOR PICK-AND-PLACE:
    - Uses full 3D distance, including Z coordinate
    - Goal is at TABLE level (z = TABLE_Z ≈ 0.02)
    - As robot LOWERS cube toward table, distance decreases, reward INCREASES!
    - No min_lift gate that would deactivate when placing
    
    This creates proper incentives:
    - Hovering at 15cm above goal: dist = 0.15m, reward = exp(-1) ≈ 0.37
    - Lowering to 3cm above goal: dist = 0.03m, reward = exp(-0.04) ≈ 0.96
    - Placed at goal: dist = 0m, reward = 1.0
    """
    grasped = object_grasped(
        env,
        robot_cfg=SceneEntityCfg(robot_name),
        ee_frame_cfg=SceneEntityCfg(ee_frame_name),
        object_cfg=SceneEntityCfg(object_name),
        diff_threshold=diff_threshold,
    )
    
    obj: RigidObject = env.scene[object_name]
    pos = obj.data.root_pos_w - env.scene.env_origins  # (B, 3)
    
    # Full 3D distance to goal (goal_pos includes the final Z = table level)
    goal = pos.new_tensor(goal_pos).unsqueeze(0)  # (1, 3)
    dist_sq = torch.sum((pos - goal) ** 2, dim=1)  # (B,)
    reward = torch.exp(-dist_sq / (float(sigma) ** 2))
    
    # Only reward when grasped (no lift gate!)
    return reward * grasped.to(dtype=reward.dtype)


def simple_place_success(
    env: "ManagerBasedRLEnv",
    object_name: str = "cube_2",
    goal_pos: tuple[float, float, float] = (0.7, 0.2, 0.0203),
    goal_half_extents_xy: tuple[float, float] = (0.05, 0.05),
    table_z: float = 0.0203,
    place_height_tol: float = 0.04,
    robot_name: str = "robot",
    open_thresh: float = 0.5,
) -> torch.Tensor:
    """Simple success reward: 1.0 if cube is in goal region, near table, and gripper open.
    
    No history tracking - just checks the current state.
    """
    obj: RigidObject = env.scene[object_name]
    pos = obj.data.root_pos_w - env.scene.env_origins
    
    # Check XY in goal region
    gx, gy = float(goal_pos[0]), float(goal_pos[1])
    hx, hy = float(goal_half_extents_xy[0]), float(goal_half_extents_xy[1])
    in_xy = torch.logical_and(
        torch.abs(pos[:, 0] - gx) <= hx,
        torch.abs(pos[:, 1] - gy) <= hy
    )
    
    # Check height near table
    h = pos[:, 2] - float(table_z)
    height_ok = torch.abs(h) < float(place_height_tol)
    
    # Check gripper is open
    open_frac = _gripper_open_fraction(env, robot_name=robot_name)
    opened = open_frac > float(open_thresh)
    
    success = torch.logical_and(torch.logical_and(in_xy, height_ok), opened)
    return success.to(dtype=torch.float32)


# =============================================================================
# IMPROVED PICK-AND-PLACE REWARDS (Fixes for poor placement learning)
# =============================================================================

def lift_to_transport_capped(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    table_z: float = 0.0203,
    target_height: float = 0.08,
) -> torch.Tensor:
    """Lift reward that CAPS at transport height (no grasp gate, simpler).
    
    Returns: min(h / target_height, 1.0) where h is height above table.
    
    This prevents the robot from maximizing "just keep high" behavior.
    Once at transport height, this reward saturates and the goal-tracking
    reward should take over.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    h = obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2] - float(table_z)
    h = torch.clamp(h, min=0.0)
    return torch.clamp(h / float(target_height), max=1.0)


def object_goal_distance_3d_tanh(
    env: "ManagerBasedRLEnv",
    std: float,
    goal_pos: tuple[float, float, float],
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    min_height_gate: float = 0.0,
    table_z: float = 0.0203,
) -> torch.Tensor:
    """FULL 3D distance to goal reward: 1 - tanh(dist_3d / std).
    
    THIS IS THE KEY FIX FOR PLACEMENT:
    - Goal Z is at table level, so as robot LOWERS cube, distance decreases!
    - No deactivation when lowering (unlike height-gated rewards)
    - Optional min_height_gate to avoid rewarding cubes sitting on table randomly
    
    With goal_pos = (0.7, 0.2, 0.0203) and std=0.15:
    - Cube at (0.5, 0.0, 0.10): dist≈0.25, reward≈0.09
    - Cube at (0.7, 0.2, 0.10): dist≈0.08, reward≈0.60
    - Cube at (0.7, 0.2, 0.03): dist≈0.01, reward≈0.96
    """
    obj: RigidObject = env.scene[object_cfg.name]
    pos = obj.data.root_pos_w - env.scene.env_origins
    
    # Optional height gate (to avoid rewarding cubes that haven't been picked up)
    h = pos[:, 2] - float(table_z)
    gate = (h > float(min_height_gate)).to(dtype=torch.float32)
    
    # Full 3D distance
    goal = pos.new_tensor(goal_pos).unsqueeze(0)
    dist_3d = torch.linalg.vector_norm(pos - goal, dim=1)
    
    reward = 1.0 - torch.tanh(dist_3d / float(std))
    return gate * reward


def lower_in_goal_region(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    goal_pos: tuple[float, float, float] = (0.7, 0.2, 0.0203),
    goal_half_extents_xy: tuple[float, float] = (0.08, 0.08),
    table_z: float = 0.0203,
    target_height: float = 0.08,
) -> torch.Tensor:
    """Reward for lowering the cube when in the goal XY region.
    
    Returns: in_goal_xy * (1 - h/target_height) where h is clamped to [0, target_height]
    
    This creates a GRADIENT toward the table in the goal region:
    - At transport height in goal: reward = 0
    - Halfway down in goal: reward = 0.5
    - On table in goal: reward = 1.0
    """
    obj: RigidObject = env.scene[object_cfg.name]
    pos = obj.data.root_pos_w - env.scene.env_origins
    
    # Check XY in goal region (slightly larger tolerance for approach)
    gx, gy = float(goal_pos[0]), float(goal_pos[1])
    hx, hy = float(goal_half_extents_xy[0]), float(goal_half_extents_xy[1])
    in_xy = torch.logical_and(
        torch.abs(pos[:, 0] - gx) <= hx,
        torch.abs(pos[:, 1] - gy) <= hy
    ).to(dtype=torch.float32)
    
    # Height above table
    h = pos[:, 2] - float(table_z)
    h = torch.clamp(h, min=0.0, max=float(target_height))
    
    # Reward for being lower (inverted height)
    lower_reward = 1.0 - (h / float(target_height))
    
    return in_xy * lower_reward


def gripper_open_when_placed(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    goal_pos: tuple[float, float, float] = (0.7, 0.2, 0.0203),
    goal_half_extents_xy: tuple[float, float] = (0.05, 0.05),
    table_z: float = 0.0203,
    place_height_tol: float = 0.03,
    robot_name: str = "robot",
) -> torch.Tensor:
    """Reward for opening gripper when cube is properly placed.
    
    Returns: placed * gripper_open_fraction
    
    This creates explicit reward for the final release action.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    pos = obj.data.root_pos_w - env.scene.env_origins
    
    # Check XY in goal
    gx, gy = float(goal_pos[0]), float(goal_pos[1])
    hx, hy = float(goal_half_extents_xy[0]), float(goal_half_extents_xy[1])
    in_xy = torch.logical_and(
        torch.abs(pos[:, 0] - gx) <= hx,
        torch.abs(pos[:, 1] - gy) <= hy
    )
    
    # Check height near table
    h = pos[:, 2] - float(table_z)
    height_ok = torch.abs(h) < float(place_height_tol)
    
    placed = torch.logical_and(in_xy, height_ok).to(dtype=torch.float32)
    
    # Gripper open fraction
    open_frac = _gripper_open_fraction(env, robot_name=robot_name)
    
    return placed * open_frac


class LiftToTransportThenGoalTerm(ManagerTermBase):
    """Combined lift + goal reward with smooth phase transition.
    
    Phase 1 (h < transport_height): Reward lifting
    Phase 2 (h >= transport_height): Reward 3D goal distance (including lowering)
    
    This ensures no reward for staying high once at transport height.
    """
    
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
    
    def reset(self, env_ids=None):
        pass  # No state needed
    
    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        object_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
        goal_pos: tuple[float, float, float] = (0.7, 0.2, 0.0203),
        table_z: float = 0.0203,
        transport_height: float = 0.06,
        goal_std: float = 0.15,
    ) -> torch.Tensor:
        obj: RigidObject = env.scene[object_cfg.name]
        pos = obj.data.root_pos_w - env.scene.env_origins
        h = pos[:, 2] - float(table_z)
        
        # Phase 1: Lift reward (capped at transport height)
        lift_reward = torch.clamp(h / float(transport_height), min=0.0, max=1.0)
        
        # Phase 2: 3D goal distance (active at all heights, but weighted)
        goal = pos.new_tensor(goal_pos).unsqueeze(0)
        dist_3d = torch.linalg.vector_norm(pos - goal, dim=1)
        goal_reward = 1.0 - torch.tanh(dist_3d / float(goal_std))
        
        # Phase transition: once lifted, goal tracking takes over
        above_transport = (h >= float(transport_height)).to(dtype=torch.float32)
        
        # Blend: lift_reward when low, goal_reward when high
        return (1.0 - above_transport) * lift_reward + above_transport * goal_reward





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
    return torch.full((env.num_envs,), float(value), device=env.device, dtype=torch.float32)



def object_is_lifted(
    env: "ManagerBasedRLEnv",
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    obj_z = obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return torch.where(obj_z > float(minimal_height), 1.0, 0.0)


def object_ee_distance_tanh(
    env: "ManagerBasedRLEnv",
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
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
    obj: RigidObject = env.scene[object_cfg.name]
    obj_pos = obj.data.root_pos_w - env.scene.env_origins
    
    lifted = obj_pos[:, 2] > float(minimal_height)
    
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
    obj: RigidObject = env.scene[object_cfg.name]
    pos = obj.data.root_pos_w - env.scene.env_origins
    
    gx, gy = float(goal_pos[0]), float(goal_pos[1])
    hx, hy = float(goal_half_extents_xy[0]), float(goal_half_extents_xy[1])
    in_xy = torch.logical_and(
        torch.abs(pos[:, 0] - gx) <= hx,
        torch.abs(pos[:, 1] - gy) <= hy
    )
    
    h = pos[:, 2] - float(table_z)
    height_ok = torch.abs(h) < float(place_height_tol)
    
    open_frac = _gripper_open_fraction(env, robot_name=robot_name)
    opened = open_frac > float(open_thresh)
    
    success = torch.logical_and(torch.logical_and(in_xy, height_ok), opened)
    return success.to(dtype=torch.float32)


def _gripper_open_fraction(env: "ManagerBasedRLEnv", robot_name: str = "robot") -> torch.Tensor:
    robot = env.scene[robot_name]
    if not hasattr(env.cfg, "gripper_joint_names") or not hasattr(env.cfg, "gripper_open_val"):
        return torch.zeros((env.num_envs,), device=env.device, dtype=torch.float32)
    joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
    if len(joint_ids) < 1:
        return torch.zeros((env.num_envs,), device=env.device, dtype=torch.float32)
    q = robot.data.joint_pos[:, joint_ids]
    open_val = float(env.cfg.gripper_open_val)
    open_val = max(open_val, 1e-6)
    opening = torch.mean(torch.abs(q), dim=1)
    return torch.clamp(opening / open_val, 0.0, 1.0)


def _cube_height_above_table(env: "ManagerBasedRLEnv", object_name: str = "cube_2", table_z: float = 0.0203) -> torch.Tensor:
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
    dist = ee_object_distance(env, ee_frame_cfg=ee_frame_cfg, object_cfg=object_cfg)
    return torch.exp(-dist / max(float(sigma), 1e-6))


def close_when_near(
    env: "ManagerBasedRLEnv",
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    robot_name: str = "robot",
    sigma: float = 0.05,
) -> torch.Tensor:
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
    g = object_grasped(
        env,
        robot_cfg=SceneEntityCfg(robot_name),
        ee_frame_cfg=SceneEntityCfg(ee_frame_name),
        object_cfg=SceneEntityCfg(object_name),
        diff_threshold=diff_threshold,
    ).to(dtype=torch.float32)
    return g


class GraspStartBonusTerm(ManagerTermBase):

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
        first_event = torch.logical_and(now, torch.logical_not(self._prev))
        first_event = torch.logical_and(first_event, torch.logical_not(self._given))
        event = first_event.to(dtype=torch.float32)
        self._given |= first_event
        self._prev = now
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
    table_z: float = 0.0203,
    transport_height: float = 0.06,
    lift_on: float = 0.03,
    switch_dist_xy: float = 0.10,
    switch_temp: float = 0.03,
    sigma_xy: float = 0.15,
    sigma_h: float = 0.03,
) -> torch.Tensor:
    g = grasp_hold(env, robot_name=robot_name, ee_frame_name=ee_frame_name, object_name=object_name, diff_threshold=diff_threshold)

    obj: RigidObject = env.scene[object_name]
    pos = obj.data.root_pos_w - env.scene.env_origins
    goal = pos.new_tensor(goal_pos).unsqueeze(0)
    d_xy = torch.linalg.vector_norm(pos[:, :2] - goal[:, :2], dim=1)  # (B,)

    h = _cube_height_above_table(env, object_name=object_name, table_z=table_z)  # (B,)

    lifted = (h > float(lift_on)).to(dtype=torch.float32)

    w = torch.sigmoid((d_xy - float(switch_dist_xy)) / max(float(switch_temp), 1e-6))
    h_des = w * float(transport_height)  # near goal -> 0

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
    hover_height: float = 0.05,
) -> torch.Tensor:
    g = grasp_hold(env, robot_name=robot_name, ee_frame_name=ee_frame_name, object_name=object_name, diff_threshold=diff_threshold)

    obj: RigidObject = env.scene[object_name]
    pos = obj.data.root_pos_w - env.scene.env_origins
    gx, gy = float(goal_pos[0]), float(goal_pos[1])
    hx, hy = float(goal_half_extents_xy[0]), float(goal_half_extents_xy[1])
    in_xy = torch.logical_and(torch.abs(pos[:, 0] - gx) <= hx, torch.abs(pos[:, 1] - gy) <= hy).to(dtype=torch.float32)

    h = _cube_height_above_table(env, object_name=object_name, table_z=table_z)
    return g * in_xy * torch.clamp(h - float(hover_height), min=0.0)


class HeightProgressWhenGraspedTerm(ManagerTermBase):

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
        return g * torch.clamp(rate, min=0.0, max=float(max_up_rate))


class GoalProgressAfterLiftWhenGraspedTerm(ManagerTermBase):

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
        return gate * torch.clamp(prog_rate, min=0.0, max=float(max_prog_rate))


class PlaceHeightErrorProgressInGoalTerm(ManagerTermBase):

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
    g = grasp_hold(env, robot_name=robot_name, ee_frame_name=ee_frame_name, object_name=object_name, diff_threshold=diff_threshold)
    in_goal = _in_goal_xy(env, object_name=object_name, goal_pos=goal_pos, goal_half_extents_xy=goal_half_extents_xy)
    h = _cube_height_above_table(env, object_name=object_name, table_z=table_z)
    viol = torch.relu(h - float(hover_thresh))
    return in_goal.to(torch.float32) * g * viol


class ReleaseEventBonusWhenReadyTerm(ManagerTermBase):

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
    in_goal = _in_goal_xy(env, object_name=object_name, goal_pos=goal_pos, goal_half_extents_xy=goal_half_extents_xy)
    h = _cube_height_above_table(env, object_name=object_name, table_z=table_z)
    open_frac = _gripper_open_fraction(env, robot_name=robot_name)
    bad = torch.logical_and(open_frac > float(open_thresh), torch.logical_and(h > float(height_thresh), torch.logical_not(in_goal)))
    return bad.to(dtype=torch.float32)


class EarlyReleaseInGoalPenaltyTerm(ManagerTermBase):

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
        g = grasp_hold(env, robot_name=robot_name, ee_frame_name=ee_frame_name, object_name=object_name, diff_threshold=diff_threshold)
        h = _cube_height_above_table(env, object_name=object_name, table_z=table_z)
        self._ever_lifted |= torch.logical_and(g > 0.5, h > float(lift_on))

        in_goal = _in_goal_xy(env, object_name=object_name, goal_pos=goal_pos, goal_half_extents_xy=goal_half_extents_xy)
        open_frac = _gripper_open_fraction(env, robot_name=robot_name)
        opened = open_frac > float(open_thresh)

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
    obj: RigidObject = env.scene[object_name]
    pos = obj.data.root_pos_w - env.scene.env_origins
    goal = pos.new_tensor(goal_pos).unsqueeze(0)
    d_goal_xy = torch.linalg.vector_norm(pos[:, :2] - goal[:, :2], dim=1)
    g_near = torch.exp(-d_goal_xy / max(float(sigma_goal), 1e-6))
    vz = obj.data.root_lin_vel_w[:, 2]
    slam = torch.relu(-vz - float(vz_thresh))
    return g_near * slam


class ActionSecondDifferenceL2Term(ManagerTermBase):

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        act_dim = int(sum(env.action_manager.action_term_dim))
        self._prev = torch.zeros((env.num_envs, act_dim), device=env.device, dtype=torch.float32)
        self._prev2 = torch.zeros((env.num_envs, act_dim), device=env.device, dtype=torch.float32)
        self._initialized = torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
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
        if not torch.all(self._initialized):
            missing = torch.logical_not(self._initialized)
            self._prev[missing] = a[missing].detach()
            self._prev2[missing] = a[missing].detach()
            self._initialized[missing] = True
            return torch.zeros((env.num_envs,), device=env.device, dtype=torch.float32)

        d2 = a - 2.0 * self._prev + self._prev2
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

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._init_xy = None

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        other_names = tuple(self.cfg.params.get("other_names", ("cube_1", "cube_3")))
        if self._init_xy is None:
            self._init_xy = torch.zeros((self._env.num_envs, len(other_names), 2), device=self._env.device)
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
    counts = torch.zeros((env.num_envs,), device=env.device, dtype=torch.float32)
    for name in other_names:
        in_goal = _in_goal_xy(env, object_name=name, goal_pos=goal_pos, goal_half_extents_xy=goal_half_extents_xy)
        counts += in_goal.to(dtype=torch.float32)
    return counts


class StableInGoalAfterLiftTerm(ManagerTermBase):

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
        h_ok = torch.exp(-torch.abs(h) / max(float(place_height_tol), 1e-6))
        obj: RigidObject = env.scene[object_name]
        v = torch.linalg.vector_norm(obj.data.root_lin_vel_w, dim=1)
        stable = torch.exp(-v / max(float(vel_sigma), 1e-6))
        shaped = in_goal.to(dtype=torch.float32) * h_ok * stable

        return shaped * self._ever_lifted.to(dtype=torch.float32)


class ReleaseWhenReadyAfterLiftTerm(ManagerTermBase):

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
    penalties = []
    thr = float(threshold)
    for name in sensor_names:
        sensor = env.scene.sensors[name]
        data = sensor.data

        forces = None
        if hasattr(data, "force_matrix_w_history"):
            forces = data.force_matrix_w_history
        elif hasattr(data, "force_matrix_w"):
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

        mags = torch.linalg.vector_norm(forces, dim=-1)  # (B, H, bodies, filters)
        mags = mags.max(dim=1).values  # (B, bodies, filters)
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
    robot = env.scene[robot_name]
    vel = robot.data.joint_vel
    limits = robot.data.soft_joint_vel_limits  # set from actuator velocity_limit (USD/PhysX if not overridden)

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
    dist = ee_object_distance(env, ee_frame_cfg=ee_frame_cfg, object_cfg=object_cfg)
    return -distance_scale * dist


def lift_shaping(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    min_height: float = 0.08,
    height_scale: float = 1.0,
) -> torch.Tensor:
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
    shaped = torch.tanh(h / max(float(height_scale), 1e-6))
    return shaped * grasp


def place_shaping(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    goal_pos: tuple[float, float, float] = (0.70, 0.20, 0.0203),
    distance_scale: float = 1.0,
) -> torch.Tensor:
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
    event = env.termination_manager.get_term(term_name).to(dtype=torch.float32)
    dt = float(getattr(env, "step_dt", 0.0) or 0.0)
    return event / max(dt, 1e-6)


class PickPlaceSuccessEventBonusTerm(ManagerTermBase):

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
    return reward * gate.to(dtype=reward.dtype)



def lift_height_reward(
    env: "ManagerBasedRLEnv",
    robot_name: str = "robot",
    ee_frame_name: str = "ee_frame",
    object_name: str = "cube_2",
    diff_threshold: float = 0.06,
    table_z: float = 0.0203,
    max_height: float = 0.15,
) -> torch.Tensor:
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
    
    lifted = h > float(min_lift)
    active = torch.logical_and(grasped, lifted)
    
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
    grasped = object_grasped(
        env,
        robot_cfg=SceneEntityCfg(robot_name),
        ee_frame_cfg=SceneEntityCfg(ee_frame_name),
        object_cfg=SceneEntityCfg(object_name),
        diff_threshold=diff_threshold,
    )
    
    obj: RigidObject = env.scene[object_name]
    pos = obj.data.root_pos_w - env.scene.env_origins  # (B, 3)
    
    goal = pos.new_tensor(goal_pos).unsqueeze(0)  # (1, 3)
    dist_sq = torch.sum((pos - goal) ** 2, dim=1)  # (B,)
    reward = torch.exp(-dist_sq / (float(sigma) ** 2))
    
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
    obj: RigidObject = env.scene[object_name]
    pos = obj.data.root_pos_w - env.scene.env_origins
    
    gx, gy = float(goal_pos[0]), float(goal_pos[1])
    hx, hy = float(goal_half_extents_xy[0]), float(goal_half_extents_xy[1])
    in_xy = torch.logical_and(
        torch.abs(pos[:, 0] - gx) <= hx,
        torch.abs(pos[:, 1] - gy) <= hy
    )
    
    h = pos[:, 2] - float(table_z)
    height_ok = torch.abs(h) < float(place_height_tol)
    
    open_frac = _gripper_open_fraction(env, robot_name=robot_name)
    opened = open_frac > float(open_thresh)
    
    success = torch.logical_and(torch.logical_and(in_xy, height_ok), opened)
    return success.to(dtype=torch.float32)



def lift_to_transport_capped(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    table_z: float = 0.0203,
    target_height: float = 0.08,
) -> torch.Tensor:
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
    obj: RigidObject = env.scene[object_cfg.name]
    pos = obj.data.root_pos_w - env.scene.env_origins
    
    h = pos[:, 2] - float(table_z)
    gate = (h > float(min_height_gate)).to(dtype=torch.float32)
    
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
    obj: RigidObject = env.scene[object_cfg.name]
    pos = obj.data.root_pos_w - env.scene.env_origins
    
    gx, gy = float(goal_pos[0]), float(goal_pos[1])
    hx, hy = float(goal_half_extents_xy[0]), float(goal_half_extents_xy[1])
    in_xy = torch.logical_and(
        torch.abs(pos[:, 0] - gx) <= hx,
        torch.abs(pos[:, 1] - gy) <= hy
    ).to(dtype=torch.float32)
    
    h = pos[:, 2] - float(table_z)
    h = torch.clamp(h, min=0.0, max=float(target_height))
    
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
    obj: RigidObject = env.scene[object_cfg.name]
    pos = obj.data.root_pos_w - env.scene.env_origins
    
    gx, gy = float(goal_pos[0]), float(goal_pos[1])
    hx, hy = float(goal_half_extents_xy[0]), float(goal_half_extents_xy[1])
    in_xy = torch.logical_and(
        torch.abs(pos[:, 0] - gx) <= hx,
        torch.abs(pos[:, 1] - gy) <= hy
    )
    
    h = pos[:, 2] - float(table_z)
    height_ok = torch.abs(h) < float(place_height_tol)
    
    placed = torch.logical_and(in_xy, height_ok).to(dtype=torch.float32)
    
    open_frac = _gripper_open_fraction(env, robot_name=robot_name)
    
    return placed * open_frac


class LiftToTransportThenGoalTerm(ManagerTermBase):
    
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
        
        lift_reward = torch.clamp(h / float(transport_height), min=0.0, max=1.0)
        
        goal = pos.new_tensor(goal_pos).unsqueeze(0)
        dist_3d = torch.linalg.vector_norm(pos - goal, dim=1)
        goal_reward = 1.0 - torch.tanh(dist_3d / float(goal_std))
        
        above_transport = (h >= float(transport_height)).to(dtype=torch.float32)
        
        return (1.0 - above_transport) * lift_reward + above_transport * goal_reward



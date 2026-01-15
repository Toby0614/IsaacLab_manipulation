

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .observations import object_grasped


def object_in_goal_and_released(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg,
    goal_pos: tuple[float, float, float],
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
    
    return torch.logical_and(torch.logical_and(in_xy, height_ok), opened)


class PickPlaceSuccessWithLiftHistoryTerm(ManagerTermBase):

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._ever_grasped = torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)
        self._ever_lifted = torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._ever_grasped[env_ids] = False
        self._ever_lifted[env_ids] = False

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        object_cfg: SceneEntityCfg,
        goal_pos: tuple[float, float, float],
        goal_half_extents_xy: tuple[float, float] = (0.05, 0.05),
        table_z: float = 0.0203,
        place_height_tol: float = 0.03,
        robot_name: str = "robot",
        ee_frame_name: str = "ee_frame",
        diff_threshold: float = 0.06,
        lift_height_thresh: float = 0.03,
        open_thresh: float = 0.5,
    ) -> torch.Tensor:
        
        grasped = object_grasped(
            env,
            robot_cfg=SceneEntityCfg(robot_name),
            ee_frame_cfg=SceneEntityCfg(ee_frame_name),
            object_cfg=object_cfg,
            diff_threshold=diff_threshold,
        )
        self._ever_grasped |= grasped
        
        obj: RigidObject = env.scene[object_cfg.name]
        pos = obj.data.root_pos_w - env.scene.env_origins
        h = pos[:, 2] - float(table_z)
        
        self._ever_lifted |= torch.logical_and(grasped, h > float(lift_height_thresh))
        
        gx, gy = float(goal_pos[0]), float(goal_pos[1])
        hx, hy = float(goal_half_extents_xy[0]), float(goal_half_extents_xy[1])
        in_xy = torch.logical_and(
            torch.abs(pos[:, 0] - gx) <= hx,
            torch.abs(pos[:, 1] - gy) <= hy
        )
        height_ok = torch.abs(h) < float(place_height_tol)
        
        open_frac = _gripper_open_fraction(env, robot_name=robot_name)
        opened = open_frac > float(open_thresh)
        
        history_ok = torch.logical_and(self._ever_grasped, self._ever_lifted)
        final_state_ok = torch.logical_and(torch.logical_and(in_xy, height_ok), opened)
        
        return torch.logical_and(history_ok, final_state_ok)


def object_near_goal(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg,
    goal_pos: tuple[float, float, float],
    xy_threshold: float = 0.05,
    z_threshold: float = 0.02,
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    obj_pos = obj.data.root_pos_w - env.scene.env_origins
    goal = obj_pos.new_tensor(goal_pos).unsqueeze(0)

    xy_dist = torch.linalg.vector_norm(obj_pos[:, :2] - goal[:, :2], dim=1)
    z_ok = torch.abs(obj_pos[:, 2] - goal[:, 2]) < z_threshold
    return torch.logical_and(xy_dist < xy_threshold, z_ok)


def cubes_stacked(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
    xy_threshold: float = 0.04,
    height_threshold: float = 0.005,
    height_diff: float = 0.0468,
    atol=0.0001,
    rtol=0.0001,
):
    robot: Articulation = env.scene[robot_cfg.name]
    cube_1: RigidObject = env.scene[cube_1_cfg.name]
    cube_2: RigidObject = env.scene[cube_2_cfg.name]
    cube_3: RigidObject = env.scene[cube_3_cfg.name]

    pos_diff_c12 = cube_1.data.root_pos_w - cube_2.data.root_pos_w
    pos_diff_c23 = cube_2.data.root_pos_w - cube_3.data.root_pos_w

    xy_dist_c12 = torch.norm(pos_diff_c12[:, :2], dim=1)
    xy_dist_c23 = torch.norm(pos_diff_c23[:, :2], dim=1)

    h_dist_c12 = torch.norm(pos_diff_c12[:, 2:], dim=1)
    h_dist_c23 = torch.norm(pos_diff_c23[:, 2:], dim=1)

    stacked = torch.logical_and(xy_dist_c12 < xy_threshold, xy_dist_c23 < xy_threshold)
    stacked = torch.logical_and(h_dist_c12 - height_diff < height_threshold, stacked)
    stacked = torch.logical_and(pos_diff_c12[:, 2] < 0.0, stacked)
    stacked = torch.logical_and(h_dist_c23 - height_diff < height_threshold, stacked)
    stacked = torch.logical_and(pos_diff_c23[:, 2] < 0.0, stacked)

    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
        surface_gripper = env.scene.surface_grippers["surface_gripper"]
        suction_cup_status = surface_gripper.state.view(-1)  # 1: closed, 0: closing, -1: open
        suction_cup_is_open = (suction_cup_status == -1)
        stacked = torch.logical_and(suction_cup_is_open, stacked)

    else:
        if hasattr(env.cfg, "gripper_joint_names"):
            gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
            assert len(gripper_joint_ids) == 2, "Terminations only support parallel gripper for now"

            stacked = torch.logical_and(
                torch.isclose(
                    robot.data.joint_pos[:, gripper_joint_ids[0]],
                    torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device),
                    atol=atol,
                    rtol=rtol,
                ),
                stacked,
            )
            stacked = torch.logical_and(
                torch.isclose(
                    robot.data.joint_pos[:, gripper_joint_ids[1]],
                    torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device),
                    atol=atol,
                    rtol=rtol,
                ),
                stacked,
            )
        else:
            raise ValueError("No gripper_joint_names found in environment config")

    return stacked


def object_in_goal_region(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg,
    goal_center: tuple[float, float, float],
    goal_half_extents_xy: tuple[float, float] = (0.05, 0.05),
    z_threshold: float = 0.03,
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    pos = obj.data.root_pos_w - env.scene.env_origins
    goal = pos.new_tensor(goal_center).unsqueeze(0)
    hx, hy = float(goal_half_extents_xy[0]), float(goal_half_extents_xy[1])
    in_xy = torch.logical_and(torch.abs(pos[:, 0] - goal[:, 0]) <= hx, torch.abs(pos[:, 1] - goal[:, 1]) <= hy)
    in_z = torch.abs(pos[:, 2] - goal[:, 2]) <= z_threshold
    return torch.logical_and(in_xy, in_z)


def other_objects_outside_goal_region(
    env: "ManagerBasedRLEnv",
    other_names: tuple[str, ...],
    goal_center: tuple[float, float, float],
    goal_half_extents_xy: tuple[float, float] = (0.05, 0.05),
    margin: float = 0.02,
) -> torch.Tensor:
    hx, hy = float(goal_half_extents_xy[0] + margin), float(goal_half_extents_xy[1] + margin)
    goal_x, goal_y = float(goal_center[0]), float(goal_center[1])

    ok = torch.ones((env.num_envs,), dtype=torch.bool, device=env.device)
    for name in other_names:
        obj: RigidObject = env.scene[name]
        pos = obj.data.root_pos_w - env.scene.env_origins
        in_xy = torch.logical_and(torch.abs(pos[:, 0] - goal_x) <= hx, torch.abs(pos[:, 1] - goal_y) <= hy)
        ok = torch.logical_and(ok, torch.logical_not(in_xy))
    return ok


def grasp_and_place_success(
    env: "ManagerBasedRLEnv",
    robot_name: str,
    ee_frame_name: str,
    object_name: str,
    goal_center: tuple[float, float, float],
    goal_half_extents_xy: tuple[float, float] = (0.05, 0.05),
    z_threshold: float = 0.03,
    other_names: tuple[str, ...] = ("cube_1", "cube_3"),
    other_margin: float = 0.02,
    diff_threshold: float = 0.06,
) -> torch.Tensor:
    grasped = object_grasped(
        env,
        robot_cfg=SceneEntityCfg(robot_name),
        ee_frame_cfg=SceneEntityCfg(ee_frame_name),
        object_cfg=SceneEntityCfg(object_name),
        diff_threshold=diff_threshold,
    )
    in_region = object_in_goal_region(
        env,
        object_cfg=SceneEntityCfg(object_name),
        goal_center=goal_center,
        goal_half_extents_xy=goal_half_extents_xy,
        z_threshold=z_threshold,
    )
    empty = other_objects_outside_goal_region(
        env,
        other_names=other_names,
        goal_center=goal_center,
        goal_half_extents_xy=goal_half_extents_xy,
        margin=other_margin,
    )
    return torch.logical_and(torch.logical_and(grasped, in_region), empty)


def place_and_release_success(
    env: "ManagerBasedRLEnv",
    robot_name: str,
    ee_frame_name: str,
    object_name: str,
    goal_center: tuple[float, float, float],
    goal_half_extents_xy: tuple[float, float] = (0.05, 0.05),
    z_threshold: float = 0.03,
    other_names: tuple[str, ...] = ("cube_1", "cube_3"),
    other_margin: float = 0.02,
    diff_threshold: float = 0.06,
) -> torch.Tensor:
    grasped = object_grasped(
        env,
        robot_cfg=SceneEntityCfg(robot_name),
        ee_frame_cfg=SceneEntityCfg(ee_frame_name),
        object_cfg=SceneEntityCfg(object_name),
        diff_threshold=diff_threshold,
    )
    released = torch.logical_not(grasped)
    in_region = object_in_goal_region(
        env,
        object_cfg=SceneEntityCfg(object_name),
        goal_center=goal_center,
        goal_half_extents_xy=goal_half_extents_xy,
        z_threshold=z_threshold,
    )
    empty = other_objects_outside_goal_region(
        env,
        other_names=other_names,
        goal_center=goal_center,
        goal_half_extents_xy=goal_half_extents_xy,
        margin=other_margin,
    )
    return torch.logical_and(torch.logical_and(in_region, empty), released)


def _gripper_open_fraction(env: "ManagerBasedRLEnv", robot_name: str = "robot") -> torch.Tensor:
    robot: Articulation = env.scene[robot_name]
    if not hasattr(env.cfg, "gripper_joint_names") or not hasattr(env.cfg, "gripper_open_val"):
        return torch.zeros((env.num_envs,), device=env.device, dtype=torch.float32)
    joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
    if len(joint_ids) < 1:
        return torch.zeros((env.num_envs,), device=env.device, dtype=torch.float32)
    q = robot.data.joint_pos[:, joint_ids]
    open_val = max(float(env.cfg.gripper_open_val), 1e-6)
    opening = torch.mean(torch.abs(q), dim=1)
    return torch.clamp(opening / open_val, 0.0, 1.0)


class FinalPlaceReleaseStableSuccessTerm(ManagerTermBase):

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._count = torch.zeros((env.num_envs,), dtype=torch.int32, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._count[env_ids] = 0

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        robot_name: str,
        object_name: str,
        goal_center: tuple[float, float, float],
        goal_half_extents_xy: tuple[float, float] = (0.05, 0.05),
        table_z: float = 0.0203,
        place_height_tol: float = 0.015,
        vel_thresh: float = 0.05,
        open_thresh: float = 0.8,
        hold_steps: int = 5,
    ) -> torch.Tensor:
        obj: RigidObject = env.scene[object_name]
        pos = obj.data.root_pos_w - env.scene.env_origins
        gx, gy = float(goal_center[0]), float(goal_center[1])
        hx, hy = float(goal_half_extents_xy[0]), float(goal_half_extents_xy[1])
        in_xy = torch.logical_and(torch.abs(pos[:, 0] - gx) <= hx, torch.abs(pos[:, 1] - gy) <= hy)

        h = (pos[:, 2] - float(table_z))
        height_ok = torch.abs(h) < float(place_height_tol)

        v = torch.linalg.vector_norm(obj.data.root_lin_vel_w, dim=1)
        stable = v < float(vel_thresh)

        open_frac = _gripper_open_fraction(env, robot_name=robot_name)
        opened = open_frac > float(open_thresh)

        good = torch.logical_and(torch.logical_and(torch.logical_and(in_xy, height_ok), stable), opened)

        self._count = torch.where(good, self._count + 1, torch.zeros_like(self._count))
        return self._count >= int(hold_steps)


class PickPlaceReleaseSuccessWithLiftHistoryTerm(ManagerTermBase):

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._ever_grasped = torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)
        self._ever_lifted = torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)
        self._count = torch.zeros((env.num_envs,), dtype=torch.int32, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._ever_grasped[env_ids] = False
        self._ever_lifted[env_ids] = False
        self._count[env_ids] = 0

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        robot_name: str,
        ee_frame_name: str,
        object_name: str,
        goal_center: tuple[float, float, float],
        goal_half_extents_xy: tuple[float, float] = (0.05, 0.05),
        table_z: float = 0.0203,
        place_height_tol: float = 0.015,
        vel_thresh: float = 0.05,
        open_thresh: float = 0.8,
        hold_steps: int = 5,
        diff_threshold: float = 0.06,
        lift_height_thresh: float = 0.03,
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
        lifted_now = torch.logical_and(grasped, h > float(lift_height_thresh))
        self._ever_lifted |= lifted_now

        gx, gy = float(goal_center[0]), float(goal_center[1])
        hx, hy = float(goal_half_extents_xy[0]), float(goal_half_extents_xy[1])
        in_xy = torch.logical_and(torch.abs(pos[:, 0] - gx) <= hx, torch.abs(pos[:, 1] - gy) <= hy)

        height_ok = torch.abs(h) < float(place_height_tol)
        v = torch.linalg.vector_norm(obj.data.root_lin_vel_w, dim=1)
        stable = v < float(vel_thresh)

        open_frac = _gripper_open_fraction(env, robot_name=robot_name)
        opened = open_frac > float(open_thresh)

        final_ok = torch.logical_and(torch.logical_and(torch.logical_and(in_xy, height_ok), stable), opened)
        success_ready = torch.logical_and(self._ever_grasped, self._ever_lifted)
        good = torch.logical_and(success_ready, final_ok)

        self._count = torch.where(good, self._count + 1, torch.zeros_like(self._count))
        return self._count >= int(hold_steps)


class SimplePickPlaceSuccessTerm(ManagerTermBase):

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._count = torch.zeros((env.num_envs,), dtype=torch.int32, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._count[env_ids] = 0

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        object_name: str,
        goal_pos: tuple[float, float, float],
        goal_half_extents_xy: tuple[float, float] = (0.05, 0.05),
        table_z: float = 0.0203,
        place_height_tol: float = 0.04,
        robot_name: str = "robot",
        open_thresh: float = 0.5,
        hold_steps: int = 5,
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
        
        good = torch.logical_and(torch.logical_and(in_xy, height_ok), opened)
        
        self._count = torch.where(good, self._count + 1, torch.zeros_like(self._count))
        return self._count >= int(hold_steps)


def simple_pickplace_success(
    env: "ManagerBasedRLEnv",
    object_name: str,
    goal_pos: tuple[float, float, float],
    goal_half_extents_xy: tuple[float, float] = (0.05, 0.05),
    table_z: float = 0.0203,
    place_height_tol: float = 0.04,
    robot_name: str = "robot",
    open_thresh: float = 0.5,
    hold_steps: int = 5,
) -> torch.Tensor:
    raise NotImplementedError("Use SimplePickPlaceSuccessTerm class directly")


class DropAfterLiftTerminationTerm(ManagerTermBase):

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._ever_lifted = torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)
        self._ungrasp_count = torch.zeros((env.num_envs,), dtype=torch.int32, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._ever_lifted[env_ids] = False
        self._ungrasp_count[env_ids] = 0

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        robot_name: str,
        ee_frame_name: str,
        object_name: str,
        goal_center: tuple[float, float, float],
        goal_half_extents_xy: tuple[float, float] = (0.05, 0.05),
        table_z: float = 0.0203,
        diff_threshold: float = 0.06,
        lift_height_thresh: float = 0.03,
        drop_height_thresh: float = 0.04,
        near_table_tol: float = 0.02,
        ungrasp_grace_steps: int = 2,
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

        self._ever_lifted |= torch.logical_and(grasped, h > float(lift_height_thresh))

        gx, gy = float(goal_center[0]), float(goal_center[1])
        hx, hy = float(goal_half_extents_xy[0]), float(goal_half_extents_xy[1])
        in_xy = torch.logical_and(torch.abs(pos[:, 0] - gx) <= hx, torch.abs(pos[:, 1] - gy) <= hy)
        near_table = torch.abs(h) <= float(near_table_tol)
        allowed_release_state = torch.logical_and(in_xy, near_table)

        dropped_now = torch.logical_and(
            self._ever_lifted,
            torch.logical_and(torch.logical_not(grasped), h > float(drop_height_thresh)),
        )
        dropped_now = torch.logical_and(dropped_now, torch.logical_not(allowed_release_state))

        self._ungrasp_count = torch.where(dropped_now, self._ungrasp_count + 1, torch.zeros_like(self._ungrasp_count))
        return self._ungrasp_count >= int(ungrasp_grace_steps)

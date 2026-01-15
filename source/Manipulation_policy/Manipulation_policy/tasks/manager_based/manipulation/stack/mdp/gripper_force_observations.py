

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gripper_force_obs(
    env: "ManagerBasedRLEnv",
    robot_name: str = "robot",
    normalize: bool = True,
    effort_limit: float = 70.0,
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_name]
    
    if not hasattr(env.cfg, "gripper_joint_names"):
        gripper_joint_names = ["panda_finger_.*"]
    else:
        gripper_joint_names = env.cfg.gripper_joint_names
    
    joint_ids, _ = robot.find_joints(gripper_joint_names)
    
    if len(joint_ids) < 1:
        return torch.zeros((env.num_envs, 2), device=env.device, dtype=torch.float32)
    
    applied_effort = robot.data.applied_torque[:, joint_ids]
    
    if applied_effort.shape[1] == 1:
        applied_effort = applied_effort.expand(-1, 2)
    elif applied_effort.shape[1] > 2:
        applied_effort = applied_effort[:, :2]
    
    force = torch.abs(applied_effort)
    
    if normalize:
        force = force / max(float(effort_limit), 1e-6)
        force = torch.clamp(force, 0.0, 2.0)  # Allow some overflow but bounded
    
    return force


def gripper_force_scalar(
    env: "ManagerBasedRLEnv",
    robot_name: str = "robot",
    normalize: bool = True,
    effort_limit: float = 70.0,
) -> torch.Tensor:
    force_2d = gripper_force_obs(env, robot_name, normalize, effort_limit)
    return torch.mean(force_2d, dim=1, keepdim=True)


def gripper_force_with_closure(
    env: "ManagerBasedRLEnv",
    robot_name: str = "robot",
    normalize: bool = True,
    effort_limit: float = 70.0,
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_name]
    
    force = gripper_force_scalar(env, robot_name, normalize, effort_limit)  # (B, 1)
    
    if hasattr(env.cfg, "gripper_joint_names") and hasattr(env.cfg, "gripper_open_val"):
        joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
        if len(joint_ids) >= 1:
            q = robot.data.joint_pos[:, joint_ids]
            open_val = max(float(env.cfg.gripper_open_val), 1e-6)
            closure = 1.0 - torch.clamp(torch.mean(torch.abs(q), dim=1, keepdim=True) / open_val, 0.0, 1.0)
        else:
            closure = torch.zeros((env.num_envs, 1), device=env.device, dtype=torch.float32)
    else:
        closure = torch.zeros((env.num_envs, 1), device=env.device, dtype=torch.float32)
    
    force_closure = force * closure
    
    return torch.cat([force, closure, force_closure], dim=1)


def gripper_contact_force_estimate(
    env: "ManagerBasedRLEnv",
    robot_name: str = "robot",
    object_name: str = "cube_2",
    ee_frame_name: str = "ee_frame",
    proximity_threshold: float = 0.06,
    normalize: bool = True,
    effort_limit: float = 70.0,
) -> torch.Tensor:
    from isaaclab.assets import RigidObject
    from isaaclab.sensors import FrameTransformer
    
    robot: Articulation = env.scene[robot_name]
    obj: RigidObject = env.scene[object_name]
    ee_frame: FrameTransformer = env.scene[ee_frame_name]
    
    force = gripper_force_scalar(env, robot_name, normalize, effort_limit)  # (B, 1)
    
    ee_pos = ee_frame.data.target_pos_w[:, 0, :]
    obj_pos = obj.data.root_pos_w
    dist = torch.linalg.vector_norm(obj_pos - ee_pos, dim=1, keepdim=True)
    
    in_contact = (dist < proximity_threshold).float()
    
    contact_force = force * in_contact
    
    return torch.cat([contact_force, in_contact], dim=1)


def gripper_grasp_force_indicator(
    env: "ManagerBasedRLEnv",
    robot_name: str = "robot",
    object_name: str = "cube_2",
    ee_frame_name: str = "ee_frame",
    proximity_threshold: float = 0.06,
    min_force_threshold: float = 0.1,
    normalize: bool = True,
    effort_limit: float = 70.0,
) -> torch.Tensor:
    force_closure = gripper_force_with_closure(env, robot_name, normalize, effort_limit)
    force_mag = force_closure[:, 0:1]  # avg force
    closure = force_closure[:, 1:2]    # closure fraction
    
    contact = gripper_contact_force_estimate(
        env, robot_name, object_name, ee_frame_name, proximity_threshold, normalize, effort_limit
    )
    in_contact = contact[:, 1:2]  # contact indicator
    
    grasp_quality = in_contact * closure * force_mag
    
    is_grasping = ((grasp_quality > min_force_threshold) & (in_contact > 0.5)).float()
    
    return torch.cat([force_mag, grasp_quality, is_grasping], dim=1)


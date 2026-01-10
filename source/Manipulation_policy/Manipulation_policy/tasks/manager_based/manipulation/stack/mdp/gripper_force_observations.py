# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gripper force/tensile sensor observation functions.

This module provides observation functions for gripper force sensing, simulating
a tensile sensor at the gripper. The force data can be derived from:
1. Joint applied torques (effort) on gripper finger joints
2. Gripper closure effort based on position error and motor effort

These observations can be added to the proprioceptive observation group to provide
tactile/force feedback without requiring physical force sensors in simulation.

Usage in environment config:
    gripper_force = ObsTerm(func=mdp.gripper_force_obs, params={"robot_name": "robot"})
    
Or via the ForceSensorEnvWrapper which automatically augments proprio observations.
"""

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
    """Gripper force observation from joint applied torques.
    
    This simulates a tensile/force sensor at the gripper by reading the
    applied torques on the gripper finger joints. Higher torque indicates
    stronger grasping force.
    
    Args:
        env: The environment instance
        robot_name: Name of the robot asset in the scene
        normalize: If True, normalize force by effort_limit to [0, ~1] range
        effort_limit: Max expected effort for normalization (N for Franka fingers)
        
    Returns:
        Tensor of shape (num_envs, 2): [left_finger_force, right_finger_force]
        
    Note:
        For Franka Panda, the gripper finger joints are:
        - panda_finger_joint1 (left finger)
        - panda_finger_joint2 (right finger)
        Typical max force ~70N per finger.
    """
    robot: Articulation = env.scene[robot_name]
    
    # Find gripper joint indices
    if not hasattr(env.cfg, "gripper_joint_names"):
        # Fallback for Franka Panda
        gripper_joint_names = ["panda_finger_.*"]
    else:
        gripper_joint_names = env.cfg.gripper_joint_names
    
    joint_ids, _ = robot.find_joints(gripper_joint_names)
    
    if len(joint_ids) < 1:
        # Return zeros if no gripper joints found
        return torch.zeros((env.num_envs, 2), device=env.device, dtype=torch.float32)
    
    # Get applied torque (effort) on gripper joints
    # applied_torque reflects the actual motor effort being applied
    applied_effort = robot.data.applied_torque[:, joint_ids]
    
    # Ensure we have 2 values (for 2-finger gripper)
    if applied_effort.shape[1] == 1:
        # Single joint - duplicate for both fingers
        applied_effort = applied_effort.expand(-1, 2)
    elif applied_effort.shape[1] > 2:
        # More than 2 - take first 2
        applied_effort = applied_effort[:, :2]
    
    # Take absolute value (force magnitude) 
    force = torch.abs(applied_effort)
    
    if normalize:
        # Normalize to [0, ~1] range for stable learning
        force = force / max(float(effort_limit), 1e-6)
        force = torch.clamp(force, 0.0, 2.0)  # Allow some overflow but bounded
    
    return force


def gripper_force_scalar(
    env: "ManagerBasedRLEnv",
    robot_name: str = "robot",
    normalize: bool = True,
    effort_limit: float = 70.0,
) -> torch.Tensor:
    """Single scalar gripper force observation (average of both fingers).
    
    This provides a single force value representing total grip strength,
    useful when you want a simpler observation space.
    
    Args:
        env: The environment instance
        robot_name: Name of the robot asset in the scene
        normalize: If True, normalize force by effort_limit
        effort_limit: Max expected effort for normalization
        
    Returns:
        Tensor of shape (num_envs, 1): average gripper force
    """
    force_2d = gripper_force_obs(env, robot_name, normalize, effort_limit)
    return torch.mean(force_2d, dim=1, keepdim=True)


def gripper_force_with_closure(
    env: "ManagerBasedRLEnv",
    robot_name: str = "robot",
    normalize: bool = True,
    effort_limit: float = 70.0,
) -> torch.Tensor:
    """Extended gripper force observation including closure state.
    
    This provides both force and closure information, which together
    indicate grasp quality:
    - High force + high closure = grasping something
    - High force + low closure = attempting to grasp but blocked
    - Low force + any closure = not actively grasping
    
    Args:
        env: The environment instance
        robot_name: Name of the robot asset in the scene
        normalize: If True, normalize values to [0, 1] range
        effort_limit: Max expected effort for normalization
        
    Returns:
        Tensor of shape (num_envs, 3): [avg_force, closure_fraction, force_closure_product]
    """
    robot: Articulation = env.scene[robot_name]
    
    # Get force observation
    force = gripper_force_scalar(env, robot_name, normalize, effort_limit)  # (B, 1)
    
    # Get gripper closure fraction
    if hasattr(env.cfg, "gripper_joint_names") and hasattr(env.cfg, "gripper_open_val"):
        joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
        if len(joint_ids) >= 1:
            q = robot.data.joint_pos[:, joint_ids]
            open_val = max(float(env.cfg.gripper_open_val), 1e-6)
            # How closed is the gripper (0 = fully open, 1 = fully closed)
            closure = 1.0 - torch.clamp(torch.mean(torch.abs(q), dim=1, keepdim=True) / open_val, 0.0, 1.0)
        else:
            closure = torch.zeros((env.num_envs, 1), device=env.device, dtype=torch.float32)
    else:
        closure = torch.zeros((env.num_envs, 1), device=env.device, dtype=torch.float32)
    
    # Force-closure product: high when both force and closure are high
    # This is a useful indicator of "actually grasping something"
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
    """Estimated contact force based on gripper effort and object proximity.
    
    This provides a "smart" force reading that only reports significant force
    when the gripper is actually near an object - similar to how a real force
    sensor would only report force when in contact.
    
    Args:
        env: The environment instance
        robot_name: Name of the robot asset
        object_name: Name of the target object for proximity check
        ee_frame_name: Name of the end-effector frame sensor
        proximity_threshold: Max distance to object for contact (meters)
        normalize: If True, normalize force values
        effort_limit: Max expected effort for normalization
        
    Returns:
        Tensor of shape (num_envs, 2): [contact_force, contact_indicator]
    """
    from isaaclab.assets import RigidObject
    from isaaclab.sensors import FrameTransformer
    
    robot: Articulation = env.scene[robot_name]
    obj: RigidObject = env.scene[object_name]
    ee_frame: FrameTransformer = env.scene[ee_frame_name]
    
    # Get base force
    force = gripper_force_scalar(env, robot_name, normalize, effort_limit)  # (B, 1)
    
    # Check proximity to object
    ee_pos = ee_frame.data.target_pos_w[:, 0, :]
    obj_pos = obj.data.root_pos_w
    dist = torch.linalg.vector_norm(obj_pos - ee_pos, dim=1, keepdim=True)
    
    # Contact indicator (1 if close and gripper has force)
    in_contact = (dist < proximity_threshold).float()
    
    # Contact force: only report force when near object
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
    """Binary-ish grasp force indicator with force magnitude.
    
    Provides a grasp quality indicator combining:
    - Whether gripper is near the object
    - Whether gripper is closed (not open)
    - How much force is being applied
    
    This is useful for policies that need to know "am I grasping properly?".
    
    Args:
        env: The environment instance
        robot_name: Name of the robot asset
        object_name: Name of the target object
        ee_frame_name: Name of the EE frame sensor
        proximity_threshold: Max distance for "in contact"
        min_force_threshold: Min force to count as "grasping" (normalized)
        normalize: If True, normalize force values
        effort_limit: Max expected effort for normalization
        
    Returns:
        Tensor of shape (num_envs, 3): [force_magnitude, grasp_quality, is_grasping]
    """
    # Get extended force+closure observation
    force_closure = gripper_force_with_closure(env, robot_name, normalize, effort_limit)
    force_mag = force_closure[:, 0:1]  # avg force
    closure = force_closure[:, 1:2]    # closure fraction
    
    # Get contact estimate
    contact = gripper_contact_force_estimate(
        env, robot_name, object_name, ee_frame_name, proximity_threshold, normalize, effort_limit
    )
    in_contact = contact[:, 1:2]  # contact indicator
    
    # Grasp quality: product of contact, closure, and force
    grasp_quality = in_contact * closure * force_mag
    
    # Binary-ish grasp indicator
    is_grasping = ((grasp_quality > min_force_threshold) & (in_contact > 0.5)).float()
    
    return torch.cat([force_mag, grasp_quality, is_grasping], dim=1)


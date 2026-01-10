# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Automatic phase detection for pick-and-place tasks.

Detects which manipulation phase each environment is in based on state observations.
Phases: reach, grasp, lift, transport, place

This enables phase-aware modality dropout as described in poe2.pdf.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class PickPlacePhaseDetector:
    """Detects manipulation phase for pick-and-place tasks.
    
    Phase definitions:
    - reach: Moving toward object, gripper open, object not grasped
    - grasp: Near object, gripper closing or closed, but not lifted
    - lift: Object grasped and lifted above threshold
    - transport: Object lifted and moving toward goal
    - place: Object at goal XY, lowering or releasing
    
    Usage:
        detector = PickPlacePhaseDetector(
            goal_pos=(0.70, 0.20, 0.0203),
            table_z=0.0203,
            lift_threshold=0.05,
        )
        
        # In environment step:
        phases = detector.detect_phases(env)
        env.dropout_manager.update_phases(phases)
    """
    
    def __init__(
        self,
        goal_pos: tuple[float, float, float] = (0.70, 0.20, 0.0203),
        table_z: float = 0.0203,
        lift_threshold: float = 0.05,
        grasp_dist_threshold: float = 0.06,
        goal_xy_radius: float = 0.10,
        transport_height_min: float = 0.04,
    ):
        """Initialize phase detector.
        
        Args:
            goal_pos: Target placement position (x, y, z)
            table_z: Table surface height
            lift_threshold: Height above table to count as "lifted"
            grasp_dist_threshold: Max EE-to-object distance to count as "grasped"
            goal_xy_radius: XY distance to goal to count as "at goal"
            transport_height_min: Min height during transport phase
        """
        self.goal_pos = goal_pos
        self.table_z = table_z
        self.lift_threshold = lift_threshold
        self.grasp_dist_threshold = grasp_dist_threshold
        self.goal_xy_radius = goal_xy_radius
        self.transport_height_min = transport_height_min
    
    def detect_phases(
        self,
        env: "ManagerBasedRLEnv",
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    ) -> list[str]:
        """Detect phase for each environment.
        
        Returns:
            List of phase names, one per environment
        """
        from isaaclab.assets import Articulation
        
        robot: Articulation = env.scene[robot_cfg.name]
        ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
        obj: RigidObject = env.scene[object_cfg.name]
        
        # Get state variables
        obj_pos = obj.data.root_pos_w - env.scene.env_origins  # (B, 3)
        ee_pos = ee_frame.data.target_pos_w[:, 0, :] - env.scene.env_origins  # (B, 3)
        
        # Distance from EE to object
        ee_to_obj_dist = torch.linalg.vector_norm(obj_pos - ee_pos, dim=1)  # (B,)
        
        # Object height above table
        obj_height = obj_pos[:, 2] - self.table_z  # (B,)
        
        # XY distance from object to goal
        goal_tensor = torch.tensor(self.goal_pos[:2], device=obj_pos.device, dtype=obj_pos.dtype)
        obj_to_goal_xy = torch.linalg.vector_norm(obj_pos[:, :2] - goal_tensor, dim=1)  # (B,)
        
        # Gripper state (closed = grasping)
        if hasattr(env.cfg, "gripper_joint_names"):
            gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
            gripper_closed = (
                torch.abs(
                    robot.data.joint_pos[:, gripper_joint_ids[0]]
                    - torch.tensor(env.cfg.gripper_open_val, device=robot.device, dtype=torch.float32)
                ) > env.cfg.gripper_threshold
            )  # (B,)
        else:
            # Fallback: assume closed if close to object
            gripper_closed = ee_to_obj_dist < self.grasp_dist_threshold
        
        # Phase detection logic per environment
        phases = []
        for i in range(env.num_envs):
            near_obj = ee_to_obj_dist[i] < self.grasp_dist_threshold
            is_lifted = obj_height[i] > self.lift_threshold
            at_goal_xy = obj_to_goal_xy[i] < self.goal_xy_radius
            in_transport_height = obj_height[i] > self.transport_height_min
            
            if at_goal_xy and is_lifted:
                phase = "place"
            elif is_lifted and in_transport_height:
                phase = "transport"
            elif is_lifted:
                phase = "lift"
            elif near_obj and gripper_closed[i]:
                phase = "grasp"
            else:
                phase = "reach"
            
            phases.append(phase)
        
        return phases
    
    def get_phase_statistics(self, phases: list[str]) -> dict[str, int]:
        """Get count of environments in each phase.
        
        Args:
            phases: List of phase names from detect_phases()
            
        Returns:
            Dictionary mapping phase name to count
        """
        stats = {}
        for phase in ["reach", "grasp", "lift", "transport", "place"]:
            stats[phase] = phases.count(phase)
        return stats


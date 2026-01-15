

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class PickPlacePhaseDetector:
    
    def __init__(
        self,
        goal_pos: tuple[float, float, float] = (0.21, 0.28, 0.0203),
        table_z: float = 0.0203,
        lift_threshold: float = 0.03,
        grasp_dist_threshold: float = 0.06,
        goal_xy_radius: float = 0.08,
        transport_height_min: float = 0.05,
    ):
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
        from isaaclab.assets import Articulation
        
        robot: Articulation = env.scene[robot_cfg.name]
        ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
        obj: RigidObject = env.scene[object_cfg.name]
        
        obj_pos = obj.data.root_pos_w - env.scene.env_origins  # (B, 3)
        ee_pos = ee_frame.data.target_pos_w[:, 0, :] - env.scene.env_origins  # (B, 3)
        
        ee_to_obj_dist = torch.linalg.vector_norm(obj_pos - ee_pos, dim=1)  # (B,)
        
        obj_height = obj_pos[:, 2] - self.table_z  # (B,)
        
        goal_tensor = torch.tensor(self.goal_pos[:2], device=obj_pos.device, dtype=obj_pos.dtype)
        obj_to_goal_xy = torch.linalg.vector_norm(obj_pos[:, :2] - goal_tensor, dim=1)  # (B,)
        
        if hasattr(env.cfg, "gripper_joint_names"):
            gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
            gripper_closed = (
                torch.abs(
                    robot.data.joint_pos[:, gripper_joint_ids[0]]
                    - torch.tensor(env.cfg.gripper_open_val, device=robot.device, dtype=torch.float32)
                ) > env.cfg.gripper_threshold
            )  # (B,)
        else:
            gripper_closed = ee_to_obj_dist < self.grasp_dist_threshold
        
        phases = []
        for i in range(env.num_envs):
            near_obj = ee_to_obj_dist[i] < self.grasp_dist_threshold
            is_lifted = obj_height[i] > self.lift_threshold
            at_goal_xy = obj_to_goal_xy[i] < self.goal_xy_radius
            in_transport_height = obj_height[i] > self.transport_height_min
            
            if at_goal_xy:
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
        stats = {}
        for phase in ["reach", "grasp", "lift", "transport", "place"]:
            stats[phase] = phases.count(phase)
        return stats


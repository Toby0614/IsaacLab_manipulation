

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

from isaaclab.managers import SceneEntityCfg

from . import observations as obs

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def rgbd_tensor_chw_with_dropout(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("wrist_cam"),
    depth_data_type: str = "distance_to_image_plane",
    depth_range: tuple[float, float] = (0.1, 2.0),
    depth_normalize: Literal["none", "range"] = "range",
) -> torch.Tensor:
    rgbd = obs.rgbd_tensor_chw(
        env,
        sensor_cfg=sensor_cfg,
        depth_data_type=depth_data_type,
        depth_range=depth_range,
        depth_normalize=depth_normalize,
    )
    
    if hasattr(env, 'dropout_manager') and env.dropout_manager.cfg.enabled:
        rgb = rgbd[:, :3, :, :]
        depth = rgbd[:, 3:4, :, :]
        
        rgb_drop, depth_drop = env.dropout_manager.apply_dropout(rgb, depth)
        
        rgbd = torch.cat([rgb_drop, depth_drop], dim=1)
    
    return rgbd


def rgb_tensor_chw_with_dropout(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("table_cam"),
) -> torch.Tensor:
    rgb = obs.rgb_tensor_chw(env, sensor_cfg=sensor_cfg)
    
    if hasattr(env, 'dropout_manager') and env.dropout_manager.cfg.enabled:
        dummy_depth = torch.zeros(rgb.shape[0], 1, rgb.shape[2], rgb.shape[3], device=rgb.device)
        rgb_drop, _ = env.dropout_manager.apply_dropout(rgb, dummy_depth)
        return rgb_drop
    
    return rgb


def multi_cam_tensor_chw_with_dropout(
    env: "ManagerBasedRLEnv",
    wrist_cam_cfg: SceneEntityCfg = SceneEntityCfg("wrist_cam"),
    table_cam_cfg: SceneEntityCfg = SceneEntityCfg("table_cam"),
    depth_data_type: str = "distance_to_image_plane",
    depth_range: tuple[float, float] = (0.1, 2.0),
    depth_normalize: Literal["none", "range"] = "range",
) -> torch.Tensor:
    wrist_rgbd = obs.rgbd_tensor_chw(
        env,
        sensor_cfg=wrist_cam_cfg,
        depth_data_type=depth_data_type,
        depth_range=depth_range,
        depth_normalize=depth_normalize,
    )
    
    table_rgb = obs.rgb_tensor_chw(env, sensor_cfg=table_cam_cfg)
    
    if table_rgb.shape[2:] != wrist_rgbd.shape[2:]:
        import torch.nn.functional as F
        wrist_h, wrist_w = wrist_rgbd.shape[2], wrist_rgbd.shape[3]
        table_rgb = F.interpolate(table_rgb, size=(wrist_h, wrist_w), mode="bilinear", align_corners=False)
    
    if hasattr(env, 'dropout_manager') and env.dropout_manager.cfg.enabled:
        wrist_rgb = wrist_rgbd[:, :3, :, :]
        wrist_depth = wrist_rgbd[:, 3:4, :, :]
        
        wrist_rgb_drop, wrist_depth_drop = env.dropout_manager.apply_dropout(wrist_rgb, wrist_depth)
        
        dummy_depth = torch.zeros(table_rgb.shape[0], 1, table_rgb.shape[2], table_rgb.shape[3], device=table_rgb.device)
        table_rgb_drop, _ = env.dropout_manager.apply_dropout(table_rgb, dummy_depth)
        
        wrist_rgbd = torch.cat([wrist_rgb_drop, wrist_depth_drop], dim=1)
        table_rgb = table_rgb_drop
    
    return torch.cat([wrist_rgbd, table_rgb], dim=1).contiguous()


def dropout_indicator_obs(env: "ManagerBasedRLEnv") -> torch.Tensor:
    if hasattr(env, 'dropout_manager'):
        return env.dropout_manager.get_dropout_indicator()
    else:
        return torch.zeros((env.num_envs, 1), device=env.device, dtype=torch.float32)


def pickplace_proprio_with_dropout_indicator(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    goal_pos: tuple[float, float, float] = (0.70, 0.20, 0.0203),
    table_z: float = 0.0203,
) -> torch.Tensor:
    proprio = obs.pickplace_proprio_vector(
        env,
        robot_cfg=robot_cfg,
        ee_frame_cfg=ee_frame_cfg,
        object_cfg=object_cfg,
        goal_pos=goal_pos,
        table_z=table_z,
    )
    
    if hasattr(env, 'dropout_manager') and env.dropout_manager.cfg.provide_dropout_indicator:
        indicator = dropout_indicator_obs(env)
        proprio = torch.cat([proprio, indicator], dim=1)
    
    return proprio


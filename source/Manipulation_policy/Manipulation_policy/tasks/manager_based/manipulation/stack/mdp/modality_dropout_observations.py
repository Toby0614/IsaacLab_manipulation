# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation functions with modality dropout support.

These are wrapper functions that call the existing observation functions
from observations.py and apply dropout via the ModalityDropoutManager.

Usage in environment config:
    Instead of:
        multi_cam = ObsTerm(func=mdp.multi_cam_tensor_chw, ...)
    
    Use:
        multi_cam = ObsTerm(func=mdp.multi_cam_tensor_chw_with_dropout, ...)
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

from isaaclab.managers import SceneEntityCfg

# Import existing observation functions
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
    """RGB-D observation with modality dropout support.
    
    This wraps the existing rgbd_tensor_chw function and applies dropout
    if a ModalityDropoutManager is attached to the environment.
    
    Returns:
        RGB-D tensor of shape (B, 4, H, W) with potential dropout applied
    """
    # Get clean observations
    rgbd = obs.rgbd_tensor_chw(
        env,
        sensor_cfg=sensor_cfg,
        depth_data_type=depth_data_type,
        depth_range=depth_range,
        depth_normalize=depth_normalize,
    )
    
    # Apply dropout if manager exists and is enabled
    if hasattr(env, 'dropout_manager') and env.dropout_manager.cfg.enabled:
        # Split RGB-D: (B, 4, H, W) -> (B, 3, H, W) + (B, 1, H, W)
        rgb = rgbd[:, :3, :, :]
        depth = rgbd[:, 3:4, :, :]
        
        # Apply dropout
        rgb_drop, depth_drop = env.dropout_manager.apply_dropout(rgb, depth)
        
        # Recombine
        rgbd = torch.cat([rgb_drop, depth_drop], dim=1)
    
    return rgbd


def rgb_tensor_chw_with_dropout(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("table_cam"),
) -> torch.Tensor:
    """RGB observation with modality dropout support.
    
    Returns:
        RGB tensor of shape (B, 3, H, W) with potential dropout applied
    """
    # Get clean observations
    rgb = obs.rgb_tensor_chw(env, sensor_cfg=sensor_cfg)
    
    # Apply dropout if manager exists
    if hasattr(env, 'dropout_manager') and env.dropout_manager.cfg.enabled:
        # Create dummy depth (won't be used, but manager expects both)
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
    """Multi-camera observation with modality dropout support.
    
    Applies dropout to both wrist camera (RGB-D) and table camera (RGB).
    
    Returns:
        Combined tensor of shape (B, 7, H, W): wrist RGB-D (4ch) + table RGB (3ch)
    """
    # Get clean observations for each camera separately
    wrist_rgbd = obs.rgbd_tensor_chw(
        env,
        sensor_cfg=wrist_cam_cfg,
        depth_data_type=depth_data_type,
        depth_range=depth_range,
        depth_normalize=depth_normalize,
    )
    
    table_rgb = obs.rgb_tensor_chw(env, sensor_cfg=table_cam_cfg)
    
    # Resize table_rgb to match wrist dimensions if needed
    if table_rgb.shape[2:] != wrist_rgbd.shape[2:]:
        import torch.nn.functional as F
        wrist_h, wrist_w = wrist_rgbd.shape[2], wrist_rgbd.shape[3]
        table_rgb = F.interpolate(table_rgb, size=(wrist_h, wrist_w), mode="bilinear", align_corners=False)
    
    # Apply dropout if manager exists
    if hasattr(env, 'dropout_manager') and env.dropout_manager.cfg.enabled:
        # Split wrist RGB-D
        wrist_rgb = wrist_rgbd[:, :3, :, :]
        wrist_depth = wrist_rgbd[:, 3:4, :, :]
        
        # Apply dropout to wrist camera
        wrist_rgb_drop, wrist_depth_drop = env.dropout_manager.apply_dropout(wrist_rgb, wrist_depth)
        
        # Apply dropout to table camera (RGB only, use dummy depth)
        dummy_depth = torch.zeros(table_rgb.shape[0], 1, table_rgb.shape[2], table_rgb.shape[3], device=table_rgb.device)
        table_rgb_drop, _ = env.dropout_manager.apply_dropout(table_rgb, dummy_depth)
        
        # Recombine
        wrist_rgbd = torch.cat([wrist_rgb_drop, wrist_depth_drop], dim=1)
        table_rgb = table_rgb_drop
    
    # Concatenate all channels
    return torch.cat([wrist_rgbd, table_rgb], dim=1).contiguous()


def dropout_indicator_obs(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Binary indicator of whether dropout is currently active.
    
    This can be added as an observation if cfg.provide_dropout_indicator is True,
    allowing the policy to know when vision is compromised (announced dropout mode).
    
    Returns:
        Tensor of shape (B, 1) with 1.0 if dropout active, 0.0 otherwise
    """
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
    """Proprio vector with optional dropout indicator appended.
    
    If env has dropout_manager with provide_dropout_indicator=True,
    appends a binary indicator to the proprio vector.
    
    Returns:
        Proprio vector with optional dropout indicator
    """
    # Get base proprio
    proprio = obs.pickplace_proprio_vector(
        env,
        robot_cfg=robot_cfg,
        ee_frame_cfg=ee_frame_cfg,
        object_cfg=object_cfg,
        goal_pos=goal_pos,
        table_z=table_z,
    )
    
    # Append dropout indicator if configured
    if hasattr(env, 'dropout_manager') and env.dropout_manager.cfg.provide_dropout_indicator:
        indicator = dropout_indicator_obs(env)
        proprio = torch.cat([proprio, indicator], dim=1)
    
    return proprio


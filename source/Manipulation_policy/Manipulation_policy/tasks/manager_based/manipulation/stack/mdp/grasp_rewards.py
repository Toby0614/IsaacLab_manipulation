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
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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


def gate_by_boolean(reward: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Multiply reward by a boolean (0/1) gate."""
    return reward * gate.to(dtype=reward.dtype)



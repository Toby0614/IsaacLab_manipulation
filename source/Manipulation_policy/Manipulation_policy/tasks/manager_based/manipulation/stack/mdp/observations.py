# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _apply_depth_postprocess(
    depth: torch.Tensor,
    depth_range: tuple[float, float] = (0.1, 2.0),
    depth_normalize: Literal["none", "range"] = "range",
) -> torch.Tensor:
    """Post-process depth-like tensors to be more RealSense-friendly.

    - Replaces inf with 0
    - Clamps to [min, max]
    - Optionally normalizes to [0, 1] (range normalization)
    """
    # Isaac returns inf for rays that miss; RealSense commonly uses 0 for invalid depth.
    depth = depth.clone()
    depth[depth == float("inf")] = 0.0

    dmin, dmax = float(depth_range[0]), float(depth_range[1])
    depth = torch.clamp(depth, min=0.0, max=dmax)
    # ensure we don't lift invalid zeros up to dmin
    valid = depth > 0.0
    depth = torch.where(valid, torch.clamp(depth, min=dmin, max=dmax), depth)

    if depth_normalize == "range":
        # map [dmin, dmax] -> [0, 1], keep invalid at 0
        denom = max(dmax - dmin, 1e-6)
        depth = torch.where(valid, (depth - dmin) / denom, depth)

    return depth


def _corrupt_rgb(
    rgb: torch.Tensor,
    *,
    modality_dropout_prob: float = 0.0,
    gaussian_noise_std: float = 0.0,
    cutout_prob: float = 0.0,
    cutout_size: tuple[int, int] = (80, 80),
) -> torch.Tensor:
    """Simple RGB corruption suitable for modality dropout experiments.

    Expects uint8 [0..255] or float image tensors; operates in-place on a clone and returns it.
    """
    x = rgb.clone()
    device = x.device

    # Whole-modality dropout
    if modality_dropout_prob > 0.0:
        drop_mask = (torch.rand((x.shape[0],), device=device) < modality_dropout_prob).view(-1, 1, 1, 1)
        x = torch.where(drop_mask, torch.zeros_like(x), x)

    # Additive Gaussian noise (works for uint8 and float; we clamp if uint8-like)
    if gaussian_noise_std > 0.0:
        noise = torch.randn_like(x.float()) * gaussian_noise_std
        x = x.float() + noise
        # If original was uint8 images, keep 0..255 range.
        x = torch.clamp(x, 0.0, 255.0)
        x = x.to(rgb.dtype) if rgb.dtype != torch.float32 else x

    # Cutout (random rectangular mask) — per-image
    if cutout_prob > 0.0:
        b, h, w = x.shape[0], x.shape[1], x.shape[2]
        ch = x.shape[3] if x.ndim == 4 else 1
        cut_h, cut_w = int(cutout_size[0]), int(cutout_size[1])
        cut_h = max(1, min(cut_h, h))
        cut_w = max(1, min(cut_w, w))
        do_cutout = torch.rand((b,), device=device) < cutout_prob
        if torch.any(do_cutout):
            ys = torch.randint(0, h - cut_h + 1, (b,), device=device)
            xs = torch.randint(0, w - cut_w + 1, (b,), device=device)
            for i in torch.where(do_cutout)[0].tolist():
                y0, x0 = int(ys[i].item()), int(xs[i].item())
                x[i, y0 : y0 + cut_h, x0 : x0 + cut_w, :ch] = 0

    return x


def _corrupt_depth(
    depth: torch.Tensor,
    *,
    modality_dropout_prob: float = 0.0,
    gaussian_noise_std: float = 0.0,
    speckle_prob: float = 0.0,
    quantize_mm: float = 1.0,
) -> torch.Tensor:
    """Depth corruption approximating RealSense-style artifacts.

    Assumes depth is float (meters) OR already normalized [0,1]. This function just injects generic
    noise/hole patterns; you control scaling/normalization elsewhere.
    """
    x = depth.clone().float()
    device = x.device

    # Whole-modality dropout
    if modality_dropout_prob > 0.0:
        drop_mask = (torch.rand((x.shape[0],), device=device) < modality_dropout_prob).view(-1, 1, 1, 1)
        x = torch.where(drop_mask, torch.zeros_like(x), x)

    # Random speckle holes
    if speckle_prob > 0.0:
        hole = torch.rand_like(x) < speckle_prob
        x = torch.where(hole, torch.zeros_like(x), x)

    # Additive Gaussian noise
    if gaussian_noise_std > 0.0:
        x = x + torch.randn_like(x) * gaussian_noise_std
        x = torch.clamp(x, min=0.0)

    # Quantize depth (approximate RealSense raw mm quantization). Only meaningful for meter-depth.
    if quantize_mm and quantize_mm > 0.0:
        # If values are in [0,1] normalized, quantization is a no-op-ish; still safe.
        mm = x * 1000.0
        mm = torch.round(mm / quantize_mm) * quantize_mm
        x = mm / 1000.0

    return x.to(depth.dtype) if depth.dtype != torch.float32 else x


def image(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("wrist_cam"),
    data_type: str = "rgb",
    normalize: bool = False,
    convert_perspective_to_orthogonal: bool = False,
    depth_range: tuple[float, float] = (0.1, 2.0),
    depth_normalize: Literal["none", "range"] = "range",
) -> torch.Tensor:
    """Read an image tensor from a camera-like sensor in the scene.

    This mirrors the common IsaacLab `image(...)` helper but lives locally so your task doesn't depend
    on `isaaclab_tasks`.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    images = sensor.data.output[data_type]

    # Optional: convert perspective depth to orthogonal depth if requested and if available.
    if (data_type == "distance_to_camera") and convert_perspective_to_orthogonal:
        images = math_utils.orthogonalize_perspective_depth(images, sensor.data.intrinsic_matrices)

    # Basic normalization behavior (optional)
    if normalize and data_type == "rgb":
        images = images.float() / 255.0
        mean_tensor = torch.mean(images, dim=(1, 2), keepdim=True)
        images = images - mean_tensor

    # Depth post-processing for depth-like outputs
    if ("distance_to" in data_type) or ("depth" in data_type):
        images = _apply_depth_postprocess(images, depth_range=depth_range, depth_normalize=depth_normalize)

    return images.clone()


def image_with_corruption(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("wrist_cam"),
    data_type: str = "rgb",
    normalize: bool = False,
    convert_perspective_to_orthogonal: bool = False,
    # depth post-processing
    depth_range: tuple[float, float] = (0.1, 2.0),
    depth_normalize: Literal["none", "range"] = "range",
    # corruption knobs (shared)
    modality_dropout_prob: float = 0.0,
    gaussian_noise_std: float = 0.0,
    # rgb knobs
    cutout_prob: float = 0.0,
    cutout_size: tuple[int, int] = (80, 80),
    # depth knobs
    speckle_prob: float = 0.0,
    quantize_mm: float = 1.0,
) -> torch.Tensor:
    """Same as :func:`image` but with simple modality dropout/corruption.

    Use this for Stage-B training: corrupt vision while adding force/contact observations.
    """
    img = image(
        env,
        sensor_cfg=sensor_cfg,
        data_type=data_type,
        normalize=normalize,
        convert_perspective_to_orthogonal=convert_perspective_to_orthogonal,
        depth_range=depth_range,
        depth_normalize=depth_normalize,
    )

    if data_type == "rgb":
        return _corrupt_rgb(
            img,
            modality_dropout_prob=modality_dropout_prob,
            gaussian_noise_std=gaussian_noise_std,
            cutout_prob=cutout_prob,
            cutout_size=cutout_size,
        )

    if ("distance_to" in data_type) or ("depth" in data_type):
        return _corrupt_depth(
            img,
            modality_dropout_prob=modality_dropout_prob,
            gaussian_noise_std=gaussian_noise_std,
            speckle_prob=speckle_prob,
            quantize_mm=quantize_mm,
        )

    # Default: no corruption
    return img


def cube_positions_in_world_frame(
    env: ManagerBasedRLEnv,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
) -> torch.Tensor:
    """The position of the cubes in the world frame."""
    cube_1: RigidObject = env.scene[cube_1_cfg.name]
    cube_2: RigidObject = env.scene[cube_2_cfg.name]
    cube_3: RigidObject = env.scene[cube_3_cfg.name]

    return torch.cat((cube_1.data.root_pos_w, cube_2.data.root_pos_w, cube_3.data.root_pos_w), dim=1)


def instance_randomize_cube_positions_in_world_frame(
    env: ManagerBasedRLEnv,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
) -> torch.Tensor:
    """The position of the cubes in the world frame."""
    if not hasattr(env, "rigid_objects_in_focus"):
        return torch.full((env.num_envs, 9), fill_value=-1)

    cube_1: RigidObjectCollection = env.scene[cube_1_cfg.name]
    cube_2: RigidObjectCollection = env.scene[cube_2_cfg.name]
    cube_3: RigidObjectCollection = env.scene[cube_3_cfg.name]

    cube_1_pos_w = []
    cube_2_pos_w = []
    cube_3_pos_w = []
    for env_id in range(env.num_envs):
        cube_1_pos_w.append(cube_1.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][0], :3])
        cube_2_pos_w.append(cube_2.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][1], :3])
        cube_3_pos_w.append(cube_3.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][2], :3])
    cube_1_pos_w = torch.stack(cube_1_pos_w)
    cube_2_pos_w = torch.stack(cube_2_pos_w)
    cube_3_pos_w = torch.stack(cube_3_pos_w)

    return torch.cat((cube_1_pos_w, cube_2_pos_w, cube_3_pos_w), dim=1)


def cube_orientations_in_world_frame(
    env: ManagerBasedRLEnv,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
):
    """The orientation of the cubes in the world frame."""
    cube_1: RigidObject = env.scene[cube_1_cfg.name]
    cube_2: RigidObject = env.scene[cube_2_cfg.name]
    cube_3: RigidObject = env.scene[cube_3_cfg.name]

    return torch.cat((cube_1.data.root_quat_w, cube_2.data.root_quat_w, cube_3.data.root_quat_w), dim=1)


def instance_randomize_cube_orientations_in_world_frame(
    env: ManagerBasedRLEnv,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
) -> torch.Tensor:
    """The orientation of the cubes in the world frame."""
    if not hasattr(env, "rigid_objects_in_focus"):
        return torch.full((env.num_envs, 9), fill_value=-1)

    cube_1: RigidObjectCollection = env.scene[cube_1_cfg.name]
    cube_2: RigidObjectCollection = env.scene[cube_2_cfg.name]
    cube_3: RigidObjectCollection = env.scene[cube_3_cfg.name]

    cube_1_quat_w = []
    cube_2_quat_w = []
    cube_3_quat_w = []
    for env_id in range(env.num_envs):
        cube_1_quat_w.append(cube_1.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][0], :4])
        cube_2_quat_w.append(cube_2.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][1], :4])
        cube_3_quat_w.append(cube_3.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][2], :4])
    cube_1_quat_w = torch.stack(cube_1_quat_w)
    cube_2_quat_w = torch.stack(cube_2_quat_w)
    cube_3_quat_w = torch.stack(cube_3_quat_w)

    return torch.cat((cube_1_quat_w, cube_2_quat_w, cube_3_quat_w), dim=1)


def object_obs(
    env: ManagerBasedRLEnv,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
):
    """
    Object observations (in world frame):
        cube_1 pos,
        cube_1 quat,
        cube_2 pos,
        cube_2 quat,
        cube_3 pos,
        cube_3 quat,
        gripper to cube_1,
        gripper to cube_2,
        gripper to cube_3,
        cube_1 to cube_2,
        cube_2 to cube_3,
        cube_1 to cube_3,
    """
    cube_1: RigidObject = env.scene[cube_1_cfg.name]
    cube_2: RigidObject = env.scene[cube_2_cfg.name]
    cube_3: RigidObject = env.scene[cube_3_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    cube_1_pos_w = cube_1.data.root_pos_w
    cube_1_quat_w = cube_1.data.root_quat_w

    cube_2_pos_w = cube_2.data.root_pos_w
    cube_2_quat_w = cube_2.data.root_quat_w

    cube_3_pos_w = cube_3.data.root_pos_w
    cube_3_quat_w = cube_3.data.root_quat_w

    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    gripper_to_cube_1 = cube_1_pos_w - ee_pos_w
    gripper_to_cube_2 = cube_2_pos_w - ee_pos_w
    gripper_to_cube_3 = cube_3_pos_w - ee_pos_w

    cube_1_to_2 = cube_1_pos_w - cube_2_pos_w
    cube_2_to_3 = cube_2_pos_w - cube_3_pos_w
    cube_1_to_3 = cube_1_pos_w - cube_3_pos_w

    return torch.cat(
        (
            cube_1_pos_w - env.scene.env_origins,
            cube_1_quat_w,
            cube_2_pos_w - env.scene.env_origins,
            cube_2_quat_w,
            cube_3_pos_w - env.scene.env_origins,
            cube_3_quat_w,
            gripper_to_cube_1,
            gripper_to_cube_2,
            gripper_to_cube_3,
            cube_1_to_2,
            cube_2_to_3,
            cube_1_to_3,
        ),
        dim=1,
    )


def instance_randomize_object_obs(
    env: ManagerBasedRLEnv,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
):
    """
    Object observations (in world frame):
        cube_1 pos,
        cube_1 quat,
        cube_2 pos,
        cube_2 quat,
        cube_3 pos,
        cube_3 quat,
        gripper to cube_1,
        gripper to cube_2,
        gripper to cube_3,
        cube_1 to cube_2,
        cube_2 to cube_3,
        cube_1 to cube_3,
    """
    if not hasattr(env, "rigid_objects_in_focus"):
        return torch.full((env.num_envs, 9), fill_value=-1)

    cube_1: RigidObjectCollection = env.scene[cube_1_cfg.name]
    cube_2: RigidObjectCollection = env.scene[cube_2_cfg.name]
    cube_3: RigidObjectCollection = env.scene[cube_3_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    cube_1_pos_w = []
    cube_2_pos_w = []
    cube_3_pos_w = []
    cube_1_quat_w = []
    cube_2_quat_w = []
    cube_3_quat_w = []
    for env_id in range(env.num_envs):
        cube_1_pos_w.append(cube_1.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][0], :3])
        cube_2_pos_w.append(cube_2.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][1], :3])
        cube_3_pos_w.append(cube_3.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][2], :3])
        cube_1_quat_w.append(cube_1.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][0], :4])
        cube_2_quat_w.append(cube_2.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][1], :4])
        cube_3_quat_w.append(cube_3.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][2], :4])
    cube_1_pos_w = torch.stack(cube_1_pos_w)
    cube_2_pos_w = torch.stack(cube_2_pos_w)
    cube_3_pos_w = torch.stack(cube_3_pos_w)
    cube_1_quat_w = torch.stack(cube_1_quat_w)
    cube_2_quat_w = torch.stack(cube_2_quat_w)
    cube_3_quat_w = torch.stack(cube_3_quat_w)

    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    gripper_to_cube_1 = cube_1_pos_w - ee_pos_w
    gripper_to_cube_2 = cube_2_pos_w - ee_pos_w
    gripper_to_cube_3 = cube_3_pos_w - ee_pos_w

    cube_1_to_2 = cube_1_pos_w - cube_2_pos_w
    cube_2_to_3 = cube_2_pos_w - cube_3_pos_w
    cube_1_to_3 = cube_1_pos_w - cube_3_pos_w

    return torch.cat(
        (
            cube_1_pos_w - env.scene.env_origins,
            cube_1_quat_w,
            cube_2_pos_w - env.scene.env_origins,
            cube_2_quat_w,
            cube_3_pos_w - env.scene.env_origins,
            cube_3_quat_w,
            gripper_to_cube_1,
            gripper_to_cube_2,
            gripper_to_cube_3,
            cube_1_to_2,
            cube_2_to_3,
            cube_1_to_3,
        ),
        dim=1,
    )


def ee_frame_pos(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_pos = ee_frame.data.target_pos_w[:, 0, :] - env.scene.env_origins[:, 0:3]

    return ee_frame_pos


def ee_frame_quat(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_quat = ee_frame.data.target_quat_w[:, 0, :]

    return ee_frame_quat


def gripper_pos(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Obtain the versatile gripper position of both Gripper and Suction Cup.
    """
    robot: Articulation = env.scene[robot_cfg.name]

    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
        # Handle multiple surface grippers by concatenating their states
        gripper_states = []
        for gripper_name, surface_gripper in env.scene.surface_grippers.items():
            gripper_states.append(surface_gripper.state.view(-1, 1))

        if len(gripper_states) == 1:
            return gripper_states[0]
        else:
            return torch.cat(gripper_states, dim=1)

    else:
        if hasattr(env.cfg, "gripper_joint_names"):
            gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
            assert len(gripper_joint_ids) == 2, "Observation gripper_pos only support parallel gripper for now"
            finger_joint_1 = robot.data.joint_pos[:, gripper_joint_ids[0]].clone().unsqueeze(1)
            finger_joint_2 = -1 * robot.data.joint_pos[:, gripper_joint_ids[1]].clone().unsqueeze(1)
            return torch.cat((finger_joint_1, finger_joint_2), dim=1)
        else:
            raise NotImplementedError("[Error] Cannot find gripper_joint_names in the environment config")


def object_grasped(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    diff_threshold: float = 0.06,
) -> torch.Tensor:
    """Check if an object is grasped by the specified robot."""

    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    object_pos = object.data.root_pos_w
    end_effector_pos = ee_frame.data.target_pos_w[:, 0, :]
    pose_diff = torch.linalg.vector_norm(object_pos - end_effector_pos, dim=1)

    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
        surface_gripper = env.scene.surface_grippers["surface_gripper"]
        suction_cup_status = surface_gripper.state.view(-1, 1)  # 1: closed, 0: closing, -1: open
        suction_cup_is_closed = (suction_cup_status == 1).to(torch.float32)
        grasped = torch.logical_and(suction_cup_is_closed, pose_diff < diff_threshold)

    else:
        if hasattr(env.cfg, "gripper_joint_names"):
            gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
            assert len(gripper_joint_ids) == 2, "Observations only support parallel gripper for now"

            grasped = torch.logical_and(
                pose_diff < diff_threshold,
                torch.abs(
                    robot.data.joint_pos[:, gripper_joint_ids[0]]
                    - torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device)
                )
                > env.cfg.gripper_threshold,
            )
            grasped = torch.logical_and(
                grasped,
                torch.abs(
                    robot.data.joint_pos[:, gripper_joint_ids[1]]
                    - torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device)
                )
                > env.cfg.gripper_threshold,
            )

    return grasped


def object_stacked(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    upper_object_cfg: SceneEntityCfg,
    lower_object_cfg: SceneEntityCfg,
    xy_threshold: float = 0.05,
    height_threshold: float = 0.005,
    height_diff: float = 0.0468,
) -> torch.Tensor:
    """Check if an object is stacked by the specified robot."""

    robot: Articulation = env.scene[robot_cfg.name]
    upper_object: RigidObject = env.scene[upper_object_cfg.name]
    lower_object: RigidObject = env.scene[lower_object_cfg.name]

    pos_diff = upper_object.data.root_pos_w - lower_object.data.root_pos_w
    height_dist = torch.linalg.vector_norm(pos_diff[:, 2:], dim=1)
    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)

    stacked = torch.logical_and(xy_dist < xy_threshold, (height_dist - height_diff) < height_threshold)

    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
        surface_gripper = env.scene.surface_grippers["surface_gripper"]
        suction_cup_status = surface_gripper.state.view(-1, 1)  # 1: closed, 0: closing, -1: open
        suction_cup_is_open = (suction_cup_status == -1).to(torch.float32)
        stacked = torch.logical_and(suction_cup_is_open, stacked)

    else:
        if hasattr(env.cfg, "gripper_joint_names"):
            gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
            assert len(gripper_joint_ids) == 2, "Observations only support parallel gripper for now"
            stacked = torch.logical_and(
                torch.isclose(
                    robot.data.joint_pos[:, gripper_joint_ids[0]],
                    torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device),
                    atol=1e-4,
                    rtol=1e-4,
                ),
                stacked,
            )
            stacked = torch.logical_and(
                torch.isclose(
                    robot.data.joint_pos[:, gripper_joint_ids[1]],
                    torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device),
                    atol=1e-4,
                    rtol=1e-4,
                ),
                stacked,
            )
        else:
            raise ValueError("No gripper_joint_names found in environment config")

    return stacked


def contact_force_magnitudes(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("gripper_contact"),
    reduce: Literal["sum", "max", "none"] = "sum",
) -> torch.Tensor:
    """Contact force magnitude(s) from a ContactSensor-like sensor.

    This is written defensively because IsaacLab sensor field names can vary by version.
    Expected sensor outputs (one of):
    - sensor.data.net_forces_w: (..., 3)
    - sensor.data.force_matrix_w: (..., 3) or (..., 6)

    Returns:
    - If reduce="sum" or "max": shape (num_envs, 1)
    - If reduce="none": shape (num_envs, K) where K is the number of contact bodies/frames reported
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    data = sensor.data

    forces = None
    if hasattr(data, "net_forces_w"):
        forces = data.net_forces_w
    elif hasattr(data, "force_matrix_w"):
        forces = data.force_matrix_w

    if forces is None:
        raise AttributeError(
            f"Contact sensor '{sensor_cfg.name}' does not expose known force fields. "
            "Expected `net_forces_w` or `force_matrix_w` on `sensor.data`."
        )

    # Normalize force tensor to (..., 3) by taking first 3 components if needed.
    if forces.shape[-1] > 3:
        forces = forces[..., :3]

    # Compute magnitude per reported body/frame
    mags = torch.linalg.vector_norm(forces, dim=-1)

    # Common shapes are (num_envs, num_bodies) or (num_envs, num_bodies, history)
    # Reduce any trailing dims besides (num_envs, K)
    if mags.ndim > 2:
        mags = mags.reshape(mags.shape[0], -1)

    if reduce == "none":
        return mags
    if reduce == "max":
        return mags.max(dim=1, keepdim=True).values
    # default: sum
    return mags.sum(dim=1, keepdim=True)


def cube_poses_in_base_frame(
    env: ManagerBasedRLEnv,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    return_key: Literal["pos", "quat", None] = None,
) -> torch.Tensor:
    """The position and orientation of the cubes in the robot base frame."""

    cube_1: RigidObject = env.scene[cube_1_cfg.name]
    cube_2: RigidObject = env.scene[cube_2_cfg.name]
    cube_3: RigidObject = env.scene[cube_3_cfg.name]

    pos_cube_1_world = cube_1.data.root_pos_w
    pos_cube_2_world = cube_2.data.root_pos_w
    pos_cube_3_world = cube_3.data.root_pos_w

    quat_cube_1_world = cube_1.data.root_quat_w
    quat_cube_2_world = cube_2.data.root_quat_w
    quat_cube_3_world = cube_3.data.root_quat_w

    robot: Articulation = env.scene[robot_cfg.name]
    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w

    pos_cube_1_base, quat_cube_1_base = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, pos_cube_1_world, quat_cube_1_world
    )
    pos_cube_2_base, quat_cube_2_base = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, pos_cube_2_world, quat_cube_2_world
    )
    pos_cube_3_base, quat_cube_3_base = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, pos_cube_3_world, quat_cube_3_world
    )

    pos_cubes_base = torch.cat((pos_cube_1_base, pos_cube_2_base, pos_cube_3_base), dim=1)
    quat_cubes_base = torch.cat((quat_cube_1_base, quat_cube_2_base, quat_cube_3_base), dim=1)

    if return_key == "pos":
        return pos_cubes_base
    elif return_key == "quat":
        return quat_cubes_base
    elif return_key is None:
        return torch.cat((pos_cubes_base, quat_cubes_base), dim=1)


def object_abs_obs_in_base_frame(
    env: ManagerBasedRLEnv,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """
    Object Abs observations (in base frame): remove the relative observations, and add abs gripper pos and quat in robot base frame
        cube_1 pos,
        cube_1 quat,
        cube_2 pos,
        cube_2 quat,
        cube_3 pos,
        cube_3 quat,
        gripper pos,
        gripper quat,
    """
    cube_1: RigidObject = env.scene[cube_1_cfg.name]
    cube_2: RigidObject = env.scene[cube_2_cfg.name]
    cube_3: RigidObject = env.scene[cube_3_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w

    cube_1_pos_w = cube_1.data.root_pos_w
    cube_1_quat_w = cube_1.data.root_quat_w

    cube_2_pos_w = cube_2.data.root_pos_w
    cube_2_quat_w = cube_2.data.root_quat_w

    cube_3_pos_w = cube_3.data.root_pos_w
    cube_3_quat_w = cube_3.data.root_quat_w

    pos_cube_1_base, quat_cube_1_base = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, cube_1_pos_w, cube_1_quat_w
    )
    pos_cube_2_base, quat_cube_2_base = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, cube_2_pos_w, cube_2_quat_w
    )
    pos_cube_3_base, quat_cube_3_base = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, cube_3_pos_w, cube_3_quat_w
    )

    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    ee_quat_w = ee_frame.data.target_quat_w[:, 0, :]
    ee_pos_base, ee_quat_base = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)

    return torch.cat(
        (
            pos_cube_1_base,
            quat_cube_1_base,
            pos_cube_2_base,
            quat_cube_2_base,
            pos_cube_3_base,
            quat_cube_3_base,
            ee_pos_base,
            ee_quat_base,
        ),
        dim=1,
    )


def ee_frame_pose_in_base_frame(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    return_key: Literal["pos", "quat", None] = None,
) -> torch.Tensor:
    """
    The end effector pose in the robot base frame.
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    ee_frame_quat_w = ee_frame.data.target_quat_w[:, 0, :]

    robot: Articulation = env.scene[robot_cfg.name]
    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w
    ee_pos_in_base, ee_quat_in_base = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, ee_frame_pos_w, ee_frame_quat_w
    )

    if return_key == "pos":
        return ee_pos_in_base
    elif return_key == "quat":
        return ee_quat_in_base
    elif return_key is None:
        return torch.cat((ee_pos_in_base, ee_quat_in_base), dim=1)

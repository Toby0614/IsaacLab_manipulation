
from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
import torch.nn.functional as F
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
    depth = depth.clone()
    depth[depth == float("inf")] = 0.0

    dmin, dmax = float(depth_range[0]), float(depth_range[1])
    depth = torch.clamp(depth, min=0.0, max=dmax)
    valid = depth > 0.0
    depth = torch.where(valid, torch.clamp(depth, min=dmin, max=dmax), depth)

    if depth_normalize == "range":
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
    x = rgb.clone()
    device = x.device

    if modality_dropout_prob > 0.0:
        drop_mask = (torch.rand((x.shape[0],), device=device) < modality_dropout_prob).view(-1, 1, 1, 1)
        x = torch.where(drop_mask, torch.zeros_like(x), x)

    if gaussian_noise_std > 0.0:
        noise = torch.randn_like(x.float()) * gaussian_noise_std
        x = x.float() + noise
        x = torch.clamp(x, 0.0, 255.0)
        x = x.to(rgb.dtype) if rgb.dtype != torch.float32 else x

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
    x = depth.clone().float()
    device = x.device

    if modality_dropout_prob > 0.0:
        drop_mask = (torch.rand((x.shape[0],), device=device) < modality_dropout_prob).view(-1, 1, 1, 1)
        x = torch.where(drop_mask, torch.zeros_like(x), x)

    if speckle_prob > 0.0:
        hole = torch.rand_like(x) < speckle_prob
        x = torch.where(hole, torch.zeros_like(x), x)

    if gaussian_noise_std > 0.0:
        x = x + torch.randn_like(x) * gaussian_noise_std
        x = torch.clamp(x, min=0.0)

    if quantize_mm and quantize_mm > 0.0:
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
    sensor = env.scene.sensors[sensor_cfg.name]
    images = sensor.data.output[data_type]

    if (data_type == "distance_to_camera") and convert_perspective_to_orthogonal:
        images = math_utils.orthogonalize_perspective_depth(images, sensor.data.intrinsic_matrices)

    if normalize and data_type == "rgb":
        images = images.float() / 255.0
        mean_tensor = torch.mean(images, dim=(1, 2), keepdim=True)
        images = images - mean_tensor

    if ("distance_to" in data_type) or ("depth" in data_type):
        images = _apply_depth_postprocess(images, depth_range=depth_range, depth_normalize=depth_normalize)

    return images.clone()


def image_feature_vector(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("wrist_cam"),
    data_type: str = "rgb",
    out_hw: tuple[int, int] = (64, 64),
    rgb_to_grayscale: bool = False,
    depth_range: tuple[float, float] = (0.1, 2.0),
    depth_normalize: Literal["none", "range"] = "range",
) -> torch.Tensor:
    img = image(
        env,
        sensor_cfg=sensor_cfg,
        data_type=data_type,
        normalize=False,
        convert_perspective_to_orthogonal=False,
        depth_range=depth_range,
        depth_normalize=depth_normalize,
    )

    if data_type == "rgb":
        x = img.float() / 255.0 if img.dtype == torch.uint8 else img.float()
        x = x.permute(0, 3, 1, 2)
        if rgb_to_grayscale:
            r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
            x = 0.2989 * r + 0.5870 * g + 0.1140 * b
    else:
        x = img.float()
        if x.ndim == 3:
            x = x.unsqueeze(-1)
        x = x.permute(0, 3, 1, 2)

    oh, ow = int(out_hw[0]), int(out_hw[1])
    x = F.interpolate(x, size=(oh, ow), mode="bilinear", align_corners=False)

    x = x.reshape(x.shape[0], -1)
    return x


def rgbd_tensor_chw(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("wrist_cam"),
    depth_data_type: str = "distance_to_image_plane",
    depth_range: tuple[float, float] = (0.1, 2.0),
    depth_normalize: Literal["none", "range"] = "range",
) -> torch.Tensor:
    rgb = image(env, sensor_cfg=sensor_cfg, data_type="rgb", normalize=False)
    depth = image(
        env,
        sensor_cfg=sensor_cfg,
        data_type=depth_data_type,
        normalize=False,
        depth_range=depth_range,
        depth_normalize=depth_normalize,
    )

    if rgb.dtype == torch.uint8:
        rgb_f = rgb.float() / 255.0
    else:
        rgb_f = rgb.float()

    rgb_chw = rgb_f.permute(0, 3, 1, 2)  # (B,3,H,W)

    if depth.ndim == 3:
        depth = depth.unsqueeze(-1)
    depth_chw = depth.float().permute(0, 3, 1, 2)  # (B,1,H,W)

    return torch.cat([rgb_chw, depth_chw], dim=1).contiguous()


def rgb_tensor_chw(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("table_cam"),
) -> torch.Tensor:
    rgb = image(env, sensor_cfg=sensor_cfg, data_type="rgb", normalize=False)

    if rgb.dtype == torch.uint8:
        rgb_f = rgb.float() / 255.0
    else:
        rgb_f = rgb.float()

    rgb_chw = rgb_f.permute(0, 3, 1, 2)  # (B,3,H,W)

    return rgb_chw.contiguous()


def multi_cam_tensor_chw(
    env,
    wrist_cam_cfg: SceneEntityCfg = SceneEntityCfg("wrist_cam"),
    table_cam_cfg: SceneEntityCfg = SceneEntityCfg("table_cam"),
    depth_data_type: str = "distance_to_image_plane",
    depth_range: tuple[float, float] = (0.1, 2.0),
    depth_normalize: Literal["none", "range"] = "range",
) -> torch.Tensor:
    wrist_rgbd = rgbd_tensor_chw(
        env,
        sensor_cfg=wrist_cam_cfg,
        depth_data_type=depth_data_type,
        depth_range=depth_range,
        depth_normalize=depth_normalize,
    )  # (B, 4, H, W)
    
    table_rgb = rgb_tensor_chw(env, sensor_cfg=table_cam_cfg)  # (B, 3, H, W)
    
    wrist_h, wrist_w = wrist_rgbd.shape[2], wrist_rgbd.shape[3]
    table_h, table_w = table_rgb.shape[2], table_rgb.shape[3]
    
    if table_h != wrist_h or table_w != wrist_w:
        table_rgb = F.interpolate(table_rgb, size=(wrist_h, wrist_w), mode="bilinear", align_corners=False)
    
    return torch.cat([wrist_rgbd, table_rgb], dim=1).contiguous()


def grasp_proprio_vector(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
) -> torch.Tensor:
    from isaaclab.envs.mdp import joint_pos_rel, joint_vel_rel, last_action

    robot_joint_pos = joint_pos_rel(env, asset_cfg=robot_cfg)  # (B, nJ)
    robot_joint_vel = joint_vel_rel(env, asset_cfg=robot_cfg)  # (B, nJ)
    act = last_action(env)  # (B, nA)

    ee_pos = ee_frame_pos(env, ee_frame_cfg=ee_frame_cfg)  # (B, 3)
    ee_quat = ee_frame_quat(env, ee_frame_cfg=ee_frame_cfg)  # (B, 4)
    grip = gripper_pos(env, robot_cfg=robot_cfg)  # (B, 1) or (B,2)

    obj: RigidObject = env.scene[object_cfg.name]
    obj_pos = obj.data.root_pos_w - env.scene.env_origins  # (B, 3)

    rel = obj_pos - ee_pos

    return torch.cat([act, robot_joint_pos, robot_joint_vel, ee_pos, ee_quat, grip, rel], dim=1)


def pickplace_proprio_vector(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    goal_pos: tuple[float, float, float] = (0.70, 0.20, 0.0203),
    table_z: float = 0.0203,
) -> torch.Tensor:
    from isaaclab.envs.mdp import joint_pos_rel, joint_vel_rel, last_action

    robot_joint_pos = joint_pos_rel(env, asset_cfg=robot_cfg)  # (B, nJ)
    robot_joint_vel = joint_vel_rel(env, asset_cfg=robot_cfg)  # (B, nJ)
    act = last_action(env)  # (B, nA)

    ee_pos = ee_frame_pos(env, ee_frame_cfg=ee_frame_cfg)  # (B, 3)
    ee_quat = ee_frame_quat(env, ee_frame_cfg=ee_frame_cfg)  # (B, 4)
    grip = gripper_pos(env, robot_cfg=robot_cfg)  # (B, 1) or (B,2)

    obj: RigidObject = env.scene[object_cfg.name]
    obj_pos = obj.data.root_pos_w - env.scene.env_origins  # (B, 3)
    
    ee_to_obj = obj_pos - ee_pos  # (B, 3)
    
    goal = torch.tensor(goal_pos, device=env.device, dtype=torch.float32)
    goal = goal.unsqueeze(0).expand(env.num_envs, -1)  # (B, 3)
    
    obj_to_goal = goal - obj_pos  # (B, 3)
    obj_to_goal_xy_dist = torch.linalg.vector_norm(obj_to_goal[:, :2], dim=1, keepdim=True)  # (B, 1)
    
    ee_to_goal = goal - ee_pos  # (B, 3)
    
    obj_height = (obj_pos[:, 2:3] - table_z)  # (B, 1)

    return torch.cat([
        act,                    # (B, nA) - last action
        robot_joint_pos,        # (B, nJ) - joint positions
        robot_joint_vel,        # (B, nJ) - joint velocities  
        ee_pos,                 # (B, 3) - end-effector position
        ee_quat,                # (B, 4) - end-effector orientation
        grip,                   # (B, 1-2) - gripper state
        ee_to_obj,              # (B, 3) - vector from EE to object (for reaching)
        goal,                   # (B, 3) - GOAL POSITION (critical!)
        obj_to_goal,            # (B, 3) - vector from object to goal (for placing)
        obj_to_goal_xy_dist,    # (B, 1) - XY distance to goal
        ee_to_goal,             # (B, 3) - vector from EE to goal
        obj_height,             # (B, 1) - object height above table
    ], dim=1)


def image_with_corruption(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("wrist_cam"),
    data_type: str = "rgb",
    normalize: bool = False,
    convert_perspective_to_orthogonal: bool = False,
    depth_range: tuple[float, float] = (0.1, 2.0),
    depth_normalize: Literal["none", "range"] = "range",
    modality_dropout_prob: float = 0.0,
    gaussian_noise_std: float = 0.0,
    cutout_prob: float = 0.0,
    cutout_size: tuple[int, int] = (80, 80),
    speckle_prob: float = 0.0,
    quantize_mm: float = 1.0,
) -> torch.Tensor:
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

    return img


def cube_positions_in_world_frame(
    env: ManagerBasedRLEnv,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
) -> torch.Tensor:
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
    robot: Articulation = env.scene[robot_cfg.name]

    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
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
            finger_joint_2 = torch.abs(robot.data.joint_pos[:, gripper_joint_ids[1]]).clone().unsqueeze(1)
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

    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    object_pos = object.data.root_pos_w
    end_effector_pos = ee_frame.data.target_pos_w[:, 0, :]
    pose_diff = torch.linalg.vector_norm(object_pos - end_effector_pos, dim=1)

    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
        surface_gripper = env.scene.surface_grippers["surface_gripper"]
        suction_cup_status = surface_gripper.state.view(-1)  # 1: closed, 0: closing, -1: open
        suction_cup_is_closed = (suction_cup_status == 1)
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

    robot: Articulation = env.scene[robot_cfg.name]
    upper_object: RigidObject = env.scene[upper_object_cfg.name]
    lower_object: RigidObject = env.scene[lower_object_cfg.name]

    pos_diff = upper_object.data.root_pos_w - lower_object.data.root_pos_w
    height_dist = torch.linalg.vector_norm(pos_diff[:, 2:], dim=1)
    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)

    stacked = torch.logical_and(xy_dist < xy_threshold, (height_dist - height_diff) < height_threshold)

    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
        surface_gripper = env.scene.surface_grippers["surface_gripper"]
        suction_cup_status = surface_gripper.state.view(-1)  # 1: closed, 0: closing, -1: open
        suction_cup_is_open = (suction_cup_status == -1)
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


def target_cube_lin_ang_vel(
    env: "ManagerBasedRLEnv",
    object_name: str = "cube_2",
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_name]
    return torch.cat([obj.data.root_lin_vel_w, obj.data.root_ang_vel_w], dim=1)


def contact_force_magnitudes(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("gripper_contact"),
    reduce: Literal["sum", "max", "none"] = "sum",
) -> torch.Tensor:
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

    if forces.shape[-1] > 3:
        forces = forces[..., :3]

    mags = torch.linalg.vector_norm(forces, dim=-1)

    if mags.ndim > 2:
        mags = mags.reshape(mags.shape[0], -1)

    if reduce == "none":
        return mags
    if reduce == "max":
        return mags.max(dim=1, keepdim=True).values
    return mags.sum(dim=1, keepdim=True)


def cube_poses_in_base_frame(
    env: ManagerBasedRLEnv,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    return_key: Literal["pos", "quat", None] = None,
) -> torch.Tensor:

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




def goal_position(
    env: "ManagerBasedRLEnv",
    goal_pos: tuple[float, float, float] = (0.70, 0.20, 0.0203),
) -> torch.Tensor:
    goal = torch.tensor(goal_pos, device=env.device, dtype=torch.float32)
    return goal.unsqueeze(0).expand(env.num_envs, -1)


def target_cube_position(
    env: "ManagerBasedRLEnv",
    object_name: str = "cube_2",
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_name]
    pos = obj.data.root_pos_w - env.scene.env_origins
    return pos


def ee_to_goal_vector(
    env: "ManagerBasedRLEnv",
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    goal_pos: tuple[float, float, float] = (0.70, 0.20, 0.0203),
) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos = ee_frame.data.target_pos_w[:, 0, :] - env.scene.env_origins
    goal = ee_pos.new_tensor(goal_pos).unsqueeze(0)
    return goal - ee_pos


def cube_to_goal_vector(
    env: "ManagerBasedRLEnv",
    object_name: str = "cube_2",
    goal_pos: tuple[float, float, float] = (0.70, 0.20, 0.0203),
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_name]
    pos = obj.data.root_pos_w - env.scene.env_origins
    goal = pos.new_tensor(goal_pos).unsqueeze(0)
    return goal - pos


def cube_to_goal_distance_xy(
    env: "ManagerBasedRLEnv",
    object_name: str = "cube_2",
    goal_pos: tuple[float, float, float] = (0.70, 0.20, 0.0203),
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_name]
    pos = obj.data.root_pos_w - env.scene.env_origins
    goal = pos.new_tensor(goal_pos).unsqueeze(0)
    dist = torch.linalg.vector_norm(pos[:, :2] - goal[:, :2], dim=1, keepdim=True)
    return dist


def cube_in_goal_region(
    env: "ManagerBasedRLEnv",
    object_name: str = "cube_2",
    goal_pos: tuple[float, float, float] = (0.70, 0.20, 0.0203),
    goal_half_extents_xy: tuple[float, float] = (0.025, 0.025),
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_name]
    pos = obj.data.root_pos_w - env.scene.env_origins
    gx, gy = float(goal_pos[0]), float(goal_pos[1])
    hx, hy = float(goal_half_extents_xy[0]), float(goal_half_extents_xy[1])
    in_x = torch.abs(pos[:, 0] - gx) <= hx
    in_y = torch.abs(pos[:, 1] - gy) <= hy
    in_goal = torch.logical_and(in_x, in_y).to(dtype=torch.float32, device=env.device)
    return in_goal.unsqueeze(-1)


def target_cube_height_above_table(
    env: "ManagerBasedRLEnv",
    object_name: str = "cube_2",
    table_z: float = 0.0203,
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_name]
    z = obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    h = z - float(table_z)
    return h.unsqueeze(-1)


def gripper_open_fraction(
    env: "ManagerBasedRLEnv",
    robot_name: str = "robot",
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_name]
    if not hasattr(env.cfg, "gripper_joint_names") or not hasattr(env.cfg, "gripper_open_val"):
        return torch.zeros((env.num_envs, 1), device=env.device, dtype=torch.float32)
    joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
    if len(joint_ids) < 1:
        return torch.zeros((env.num_envs, 1), device=env.device, dtype=torch.float32)
    q = robot.data.joint_pos[:, joint_ids]
    open_val = max(float(env.cfg.gripper_open_val), 1e-6)
    opening = torch.mean(torch.abs(q), dim=1, keepdim=True)
    return torch.clamp(opening / open_val, 0.0, 1.0)

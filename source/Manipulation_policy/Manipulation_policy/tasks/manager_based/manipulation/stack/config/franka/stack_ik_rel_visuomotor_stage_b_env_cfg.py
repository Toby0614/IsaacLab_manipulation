# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stage-B env cfg: corrupted vision (RGB-D) + proprio (+ later: force/contact)."""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ... import mdp
from . import stack_ik_rel_visuomotor_env_cfg


@configclass
class ObservationsCfg:
    """Observation specifications for Stage-B.

    Notes:
    - Uses `mdp.image_with_corruption` for RGB-D to implement modality dropout / corruption.
    - Keep non-vision terms identical to Stage-A so you can transfer weights cleanly.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object = ObsTerm(func=mdp.object_obs)
        cube_positions = ObsTerm(func=mdp.cube_positions_in_world_frame)
        cube_orientations = ObsTerm(func=mdp.cube_orientations_in_world_frame)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)

        # --- Corrupted Vision (RGB-D) ---
        # You can tune these knobs per modality:
        # - modality_dropout_prob: probability of dropping the entire frame (sets to 0)
        # - gaussian_noise_std: RGB noise in pixel units; depth noise in meters (if not normalized)
        # - cutout_* affects RGB only; speckle/quantize affects depth only
        table_rgb = ObsTerm(
            func=mdp.image_with_corruption,
            params={
                "sensor_cfg": SceneEntityCfg("table_cam"),
                "data_type": "rgb",
                "normalize": False,
                "modality_dropout_prob": 0.2,
                "gaussian_noise_std": 8.0,
                "cutout_prob": 0.25,
                "cutout_size": (120, 120),
            },
        )
        table_depth = ObsTerm(
            func=mdp.image_with_corruption,
            params={
                "sensor_cfg": SceneEntityCfg("table_cam"),
                "data_type": "distance_to_image_plane",
                "normalize": False,
                "depth_range": (0.1, 2.0),
                "depth_normalize": "range",
                "modality_dropout_prob": 0.2,
                "gaussian_noise_std": 0.002,  # ~2mm
                "speckle_prob": 0.01,
                "quantize_mm": 1.0,
            },
        )
        wrist_rgb = ObsTerm(
            func=mdp.image_with_corruption,
            params={
                "sensor_cfg": SceneEntityCfg("wrist_cam"),
                "data_type": "rgb",
                "normalize": False,
                "modality_dropout_prob": 0.2,
                "gaussian_noise_std": 8.0,
                "cutout_prob": 0.25,
                "cutout_size": (120, 120),
            },
        )
        wrist_depth = ObsTerm(
            func=mdp.image_with_corruption,
            params={
                "sensor_cfg": SceneEntityCfg("wrist_cam"),
                "data_type": "distance_to_image_plane",
                "normalize": False,
                "depth_range": (0.1, 2.0),
                "depth_normalize": "range",
                "modality_dropout_prob": 0.2,
                "gaussian_noise_std": 0.002,
                "speckle_prob": 0.01,
                "quantize_mm": 1.0,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class FrankaCubeStackVisuomotorStageBEnvCfg(stack_ik_rel_visuomotor_env_cfg.FrankaCubeStackVisuomotorEnvCfg):
    """Stage-B: inherits the same scene/cameras, swaps in corrupted RGB-D observations."""

    observations: ObservationsCfg = ObservationsCfg()



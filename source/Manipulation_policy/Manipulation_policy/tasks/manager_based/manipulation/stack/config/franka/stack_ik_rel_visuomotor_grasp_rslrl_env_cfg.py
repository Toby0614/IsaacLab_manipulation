# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A minimal RSL-RL-friendly grasp env (cube_2) with proprio + RGB-D features.

Key point:
- We *downsample + flatten* RGB and depth into vectors so the default MLP actor-critic in RSL-RL can train.
  This is the simplest bring-up path. Later you can swap to a CNN+fusion policy.
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ... import mdp
from . import stack_ik_rel_visuomotor_env_cfg


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # --- proprio / state ---
        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)

        # --- minimal object hint (optional but helps early learning) ---
        cube_positions = ObsTerm(func=mdp.cube_positions_in_world_frame)

        # --- vision (RGB-D) as feature vectors ---
        # Using wrist camera only for grasp.
        wrist_rgb_feat = ObsTerm(
            func=mdp.image_feature_vector,
            params={
                "sensor_cfg": SceneEntityCfg("wrist_cam"),
                "data_type": "rgb",
                "out_hw": (64, 64),
                "rgb_to_grayscale": True,
            },
        )
        wrist_depth_feat = ObsTerm(
            func=mdp.image_feature_vector,
            params={
                "sensor_cfg": SceneEntityCfg("wrist_cam"),
                "data_type": "distance_to_image_plane",
                "out_hw": (64, 64),
                "depth_range": (0.1, 2.0),
                "depth_normalize": "range",
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            # Important: make a single flat vector so default RSL-RL MLP can consume it.
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class FrankaCubeStackVisuomotorGraspRslRlEnvCfg(stack_ik_rel_visuomotor_env_cfg.FrankaCubeStackVisuomotorEnvCfg):
    """Same scene/actions/rewards as the visuomotor env, but RSL-RL-friendly observations."""

    observations: ObservationsCfg = ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Reduce render cost while still having RGB-D.
        self.scene.wrist_cam.height = 128
        self.scene.wrist_cam.width = 128



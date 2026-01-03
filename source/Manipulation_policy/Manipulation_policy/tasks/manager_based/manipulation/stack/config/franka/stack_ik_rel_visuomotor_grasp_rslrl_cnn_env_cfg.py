# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL CNN grasp env (cube_2): proprio vector + RGB-D image tensor.

IMPORTANT (matches RSL-RL's ActorCriticCNN expectations):
- The environment must expose observation *groups* as top-level keys in the observation dict.
- ActorCriticCNN takes an `obs_groups` mapping that lists which observation groups are used by the actor/critic.

So we expose two top-level observation groups:
- "proprio": (B, D) 2D tensor
- "rgbd": (B, 4, H, W) 4D tensor
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
    class ProprioCfg(ObsGroup):
        proprio = ObsTerm(
            func=mdp.grasp_proprio_vector,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cube_2"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            # single 2D tensor output (B, D)
            self.concatenate_terms = True

    @configclass
    class RgbdCfg(ObsGroup):
        rgbd = ObsTerm(
            func=mdp.rgbd_tensor_chw,
            params={
                "sensor_cfg": SceneEntityCfg("wrist_cam"),
                "depth_data_type": "distance_to_image_plane",
                "depth_range": (0.1, 2.0),
                "depth_normalize": "range",
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            # Single 4D tensor output (B, C, H, W).
            # IMPORTANT: must be a tensor, not a nested dict/tensordict, for ActorCriticCNN.
            self.concatenate_terms = True

    # top-level observation groups (keys used by RSL-RL `obs_groups`)
    proprio: ProprioCfg = ProprioCfg()
    rgbd: RgbdCfg = RgbdCfg()


@configclass
class FrankaCubeStackVisuomotorGraspRslRlCnnEnvCfg(stack_ik_rel_visuomotor_env_cfg.FrankaCubeStackVisuomotorEnvCfg):
    """Same scene/actions/rewards/terminations as visuomotor grasp, with CNN-friendly observations."""

    observations: ObservationsCfg = ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Keep wrist camera modest for throughput; your base env sets 640x480.
        self.scene.wrist_cam.height = 128
        self.scene.wrist_cam.width = 128



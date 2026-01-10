# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Example: Franka Pick-and-Place Environment with Modality Dropout.

This config demonstrates how to enable the modality dropout system
without modifying your existing working environment.

Key differences from pickplace_env_cfg.py:
1. Observations use dropout-aware wrappers
2. ModalityDropoutManager is initialized in __post_init__
3. Manager state is updated in reset and step callbacks

Usage:
    # Training with dropout:
    python scripts/rsl_rl/train.py --task=Isaac-Franka-PickPlace-Dropout-Direct-v0
    
    # Evaluation with deterministic dropout:
    python scripts/rsl_rl/play.py --task=Isaac-Franka-PickPlace-Dropout-Direct-v0
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ... import mdp
from .pickplace_env_cfg import FrankaPickPlaceEnvCfg, GOAL_POS, TABLE_Z

# Import modality dropout components
from ...mdp.modality_dropout_cfg import (
    HardDropoutTrainingCfg,
    PhaseBasedDropoutCfg,
    EvalDropoutCfg,
)
from ...mdp.modality_dropout_manager import ModalityDropoutManager
from ...mdp.phase_detector import PickPlacePhaseDetector


# =============================================================================
# OBSERVATIONS WITH DROPOUT
# =============================================================================
@configclass
class ObservationsWithDropoutCfg:
    """Observations with modality dropout enabled.
    
    Uses dropout-aware observation wrappers instead of standard ones.
    """

    @configclass
    class ProprioCfg(ObsGroup):
        """Proprioceptive observations (same as base config)."""

        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)
        gripper_open_frac = ObsTerm(func=mdp.gripper_open_fraction, params={"robot_name": "robot"})
        object = ObsTerm(func=mdp.object_obs)
        cube2_lin_ang_vel = ObsTerm(func=mdp.target_cube_lin_ang_vel, params={"object_name": "cube_2"})

        # Goal information
        goal_position = ObsTerm(func=mdp.goal_position, params={"goal_pos": GOAL_POS})
        cube_to_goal = ObsTerm(func=mdp.cube_to_goal_vector, params={"object_name": "cube_2", "goal_pos": GOAL_POS})
        cube_to_goal_dist = ObsTerm(func=mdp.cube_to_goal_distance_xy, params={"object_name": "cube_2", "goal_pos": GOAL_POS})
        cube_height = ObsTerm(func=mdp.target_cube_height_above_table, params={"object_name": "cube_2", "table_z": TABLE_Z})
        cube_in_goal_xy = ObsTerm(
            func=mdp.cube_in_goal_region,
            params={"object_name": "cube_2", "goal_pos": GOAL_POS, "goal_half_extents_xy": (0.05, 0.05)},
        )
        
        # Optional: Add dropout indicator if using announced dropout mode
        # dropout_indicator = ObsTerm(func=mdp.dropout_indicator_obs)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class MultiCamCfg(ObsGroup):
        """Multi-camera observations WITH DROPOUT APPLIED.
        
        KEY CHANGE: Uses multi_cam_tensor_chw_with_dropout instead of multi_cam_tensor_chw
        """

        multi_cam = ObsTerm(
            func=mdp.multi_cam_tensor_chw_with_dropout,  # <-- Dropout-aware wrapper
            params={
                "wrist_cam_cfg": SceneEntityCfg("wrist_cam"),
                "table_cam_cfg": SceneEntityCfg("table_cam"),
                "depth_data_type": "distance_to_image_plane",
                "depth_range": (0.1, 2.0),
                "depth_normalize": "range",
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    proprio: ProprioCfg = ProprioCfg()
    multi_cam: MultiCamCfg = MultiCamCfg()


# =============================================================================
# ENVIRONMENT CONFIG WITH DROPOUT
# =============================================================================
@configclass
class FrankaPickPlaceWithDropoutEnvCfg(FrankaPickPlaceEnvCfg):
    """Franka Pick-and-Place with Modality Dropout.
    
    This config extends the base pickplace config and adds:
    1. Dropout-aware observations
    2. ModalityDropoutManager initialization
    3. Phase detector for phase-aware dropout (optional)
    """

    # Override observations to use dropout-aware versions
    observations: ObservationsWithDropoutCfg = ObservationsWithDropoutCfg()

    def __post_init__(self):
        super().__post_init__()
        
        # =================================================================
        # MODALITY DROPOUT CONFIGURATION
        # =================================================================
        # Choose one of the preset configurations or create your own:
        
        # Option 1: Hard dropout training (recommended for M3 baseline)
        self.dropout_cfg = HardDropoutTrainingCfg()
        
        # Option 2: Phase-based dropout for systematic evaluation
        # self.dropout_cfg = PhaseBasedDropoutCfg()
        
        # Option 3: Deterministic dropout for evaluation
        # self.dropout_cfg = EvalDropoutCfg()
        
        # Option 4: Custom configuration
        # from ...mdp.modality_dropout_cfg import ModalityDropoutCfg
        # self.dropout_cfg = ModalityDropoutCfg(
        #     enabled=True,
        #     dropout_mode="hard",
        #     dropout_probability=0.02,
        #     dropout_duration_range=(10, 50),
        # )
        
        # =================================================================
        # PHASE DETECTION (Optional, for phase-aware dropout)
        # =================================================================
        # If using phase-aware dropout, configure the phase detector
        if self.dropout_cfg.phase_aware:
            self.phase_detector_cfg = {
                "goal_pos": GOAL_POS,
                "table_z": TABLE_Z,
                "lift_threshold": 0.05,
                "grasp_dist_threshold": 0.06,
                "goal_xy_radius": 0.10,
            }


# =============================================================================
# REGISTRATION (for gym.make)
# =============================================================================
import gymnasium as gym

# Register the environment
gym.register(
    id="Isaac-Franka-PickPlace-Dropout-Direct-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FrankaPickPlaceWithDropoutEnvCfg,
        "rsl_rl_cfg_entry_point": f"{__name__.rsplit('.', 1)[0]}.agents.rsl_rl_ppo_cnn_cfg:FrankaPickPlacePPORunnerCfg",
    },
)


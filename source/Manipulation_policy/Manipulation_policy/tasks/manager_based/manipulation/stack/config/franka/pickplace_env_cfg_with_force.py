# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka Pick-and-Place Environment with Gripper Force Sensing (Tensile Sensor).

This config extends the base pick-and-place environment to include gripper force
observations WITHOUT modifying the reward system. The force data is added to the
proprioceptive observation group alongside existing robot state observations.

Key Features:
1. Same reward system as base pickplace_env_cfg.py
2. Same scene setup, actions, terminations
3. ADDED: Gripper force sensing in proprio observations
4. Force derived from joint applied torques (simulated tensile sensor)

Usage:
    # Training with force sensing:
    python scripts/rsl_rl/train.py --task=Isaac-Franka-PickPlace-Force-v0
    
    # Or use the wrapper approach in train.py:
    python scripts/rsl_rl/train.py --task=Isaac-Franka-PickPlace-v0 --force_sensing
    
The force observation modes:
- "scalar": Single average force value (1 dim) - simplest
- "per_finger": Per-finger forces (2 dims)
- "with_closure": Force + closure + product (3 dims) - recommended
- "grasp_indicator": Force + quality + is_grasping (3 dims) - most informative
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ... import mdp
from .pickplace_env_cfg import (
    FrankaPickPlaceEnvCfg,
    GOAL_POS,
    GOAL_HALF_EXTENTS_XY,
    TABLE_Z,
    LIFT_HEIGHT,
    GRASP_DIFF_THRESH_REW,
)

# Import force sensor components
from ...mdp.gripper_force_observations import (
    gripper_force_obs,
    gripper_force_scalar,
    gripper_force_with_closure,
    gripper_contact_force_estimate,
    gripper_grasp_force_indicator,
)


# =============================================================================
# OBSERVATIONS WITH FORCE SENSING
# =============================================================================
@configclass
class ObservationsWithForceCfg:
    """Observations with gripper force sensing (tensile sensor) added to proprio.
    
    This extends the base observations by adding force data to proprioception.
    The multi_cam observations remain unchanged.
    """

    @configclass
    class ProprioCfg(ObsGroup):
        """Proprioceptive observations WITH force sensing.
        
        Same as base config but with added gripper force observations.
        Force observations provide information about grasping force,
        similar to a tensile sensor on a real gripper.
        """

        # === Base proprio observations (unchanged) ===
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
            params={"object_name": "cube_2", "goal_pos": GOAL_POS, "goal_half_extents_xy": GOAL_HALF_EXTENTS_XY},
        )

        # === NEW: Gripper Force Sensing (Tensile Sensor) ===
        # Option 1: Simple scalar force (1 dim) - choose ONE of these options
        # gripper_force = ObsTerm(func=gripper_force_scalar, params={"robot_name": "robot"})
        
        # Option 2: Per-finger forces (2 dims)
        # gripper_force = ObsTerm(func=gripper_force_obs, params={"robot_name": "robot"})
        
        # Option 3: Force + closure info (3 dims) - RECOMMENDED
        gripper_force = ObsTerm(
            func=gripper_force_with_closure,
            params={
                "robot_name": "robot",
                "normalize": True,
                "effort_limit": 70.0,  # Franka finger max effort ~70N
            },
        )
        
        # Option 4: Grasp quality indicator (3 dims) - most informative
        # gripper_force = ObsTerm(
        #     func=gripper_grasp_force_indicator,
        #     params={
        #         "robot_name": "robot",
        #         "object_name": "cube_2",
        #         "ee_frame_name": "ee_frame",
        #         "proximity_threshold": 0.06,
        #     },
        # )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class MultiCamCfg(ObsGroup):
        """Multi-camera observations (unchanged from base config)."""

        multi_cam = ObsTerm(
            func=mdp.multi_cam_tensor_chw_with_dropout,
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
# ENVIRONMENT CONFIG WITH FORCE SENSING
# =============================================================================
@configclass
class FrankaPickPlaceWithForceEnvCfg(FrankaPickPlaceEnvCfg):
    """Franka Pick-and-Place with Gripper Force Sensing.
    
    This config extends the base pickplace config to add:
    1. Gripper force observations in proprio (simulated tensile sensor)
    
    The reward system, scene setup, and terminations are UNCHANGED.
    Only observations are modified to include force data.
    
    Force sensing options (configure in ObservationsWithForceCfg.ProprioCfg):
    - gripper_force_scalar: Average force (1 dim)
    - gripper_force_obs: Per-finger force (2 dims)
    - gripper_force_with_closure: Force + closure info (3 dims) [DEFAULT]
    - gripper_grasp_force_indicator: Force + grasp quality (3 dims)
    """

    # Override observations to use force-sensing version
    observations: ObservationsWithForceCfg = ObservationsWithForceCfg()

    def __post_init__(self):
        # Call parent post_init to set up scene, actions, etc.
        super().__post_init__()
        
        # NOTE: Rewards and terminations are inherited unchanged from base config
        # This ensures the task behavior and learning objective remain identical


# =============================================================================
# ALTERNATIVE: Scalar Force Config (minimal overhead)
# =============================================================================
@configclass
class ObservationsWithScalarForceCfg(ObservationsWithForceCfg):
    """Observations with minimal force sensing (single scalar)."""
    
    @configclass
    class ProprioCfg(ObservationsWithForceCfg.ProprioCfg):
        """Proprio with scalar force only."""
        
        # Override to use scalar force
        gripper_force = ObsTerm(
            func=gripper_force_scalar,
            params={"robot_name": "robot", "normalize": True, "effort_limit": 70.0},
        )

    proprio: ProprioCfg = ProprioCfg()


@configclass
class FrankaPickPlaceWithScalarForceEnvCfg(FrankaPickPlaceEnvCfg):
    """Pick-and-Place with single scalar force observation (minimal overhead)."""
    observations: ObservationsWithScalarForceCfg = ObservationsWithScalarForceCfg()
    
    def __post_init__(self):
        super().__post_init__()


# =============================================================================
# ALTERNATIVE: Grasp Indicator Config (most informative)
# =============================================================================
@configclass
class ObservationsWithGraspIndicatorCfg(ObservationsWithForceCfg):
    """Observations with grasp quality indicator."""
    
    @configclass
    class ProprioCfg(ObservationsWithForceCfg.ProprioCfg):
        """Proprio with grasp indicator."""
        
        gripper_force = ObsTerm(
            func=gripper_grasp_force_indicator,
            params={
                "robot_name": "robot",
                "object_name": "cube_2",
                "ee_frame_name": "ee_frame",
                "proximity_threshold": 0.06,
                "min_force_threshold": 0.1,
                "normalize": True,
                "effort_limit": 70.0,
            },
        )

    proprio: ProprioCfg = ProprioCfg()


@configclass
class FrankaPickPlaceWithGraspIndicatorEnvCfg(FrankaPickPlaceEnvCfg):
    """Pick-and-Place with grasp quality indicator (most informative)."""
    observations: ObservationsWithGraspIndicatorCfg = ObservationsWithGraspIndicatorCfg()
    
    def __post_init__(self):
        super().__post_init__()


# =============================================================================
# REGISTRATION (for gym.make)
# =============================================================================
import gymnasium as gym

# Main force-sensing environment (with_closure mode - 3 dims)
gym.register(
    id="Isaac-Franka-PickPlace-Force-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FrankaPickPlaceWithForceEnvCfg,
        "rsl_rl_cfg_entry_point": f"{__name__.rsplit('.', 1)[0]}.agents.rsl_rl_ppo_cnn_cfg:FrankaPickPlacePPORunnerCfg",
    },
)

# Scalar force variant (1 dim - minimal)
gym.register(
    id="Isaac-Franka-PickPlace-Force-Scalar-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FrankaPickPlaceWithScalarForceEnvCfg,
        "rsl_rl_cfg_entry_point": f"{__name__.rsplit('.', 1)[0]}.agents.rsl_rl_ppo_cnn_cfg:FrankaPickPlacePPORunnerCfg",
    },
)

# Grasp indicator variant (3 dims - most informative)
gym.register(
    id="Isaac-Franka-PickPlace-Force-GraspIndicator-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FrankaPickPlaceWithGraspIndicatorEnvCfg,
        "rsl_rl_cfg_entry_point": f"{__name__.rsplit('.', 1)[0]}.agents.rsl_rl_ppo_cnn_cfg:FrankaPickPlacePPORunnerCfg",
    },
)


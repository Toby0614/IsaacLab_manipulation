

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

from ...mdp.gripper_force_observations import (
    gripper_force_obs,
    gripper_force_scalar,
    gripper_force_with_closure,
    gripper_contact_force_estimate,
    gripper_grasp_force_indicator,
)


@configclass
class ObservationsWithForceCfg:

    @configclass
    class ProprioCfg(ObsGroup):

        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)
        gripper_open_frac = ObsTerm(func=mdp.gripper_open_fraction, params={"robot_name": "robot"})
        object = ObsTerm(func=mdp.object_obs)
        cube2_lin_ang_vel = ObsTerm(func=mdp.target_cube_lin_ang_vel, params={"object_name": "cube_2"})

        goal_position = ObsTerm(func=mdp.goal_position, params={"goal_pos": GOAL_POS})
        cube_to_goal = ObsTerm(func=mdp.cube_to_goal_vector, params={"object_name": "cube_2", "goal_pos": GOAL_POS})
        cube_to_goal_dist = ObsTerm(func=mdp.cube_to_goal_distance_xy, params={"object_name": "cube_2", "goal_pos": GOAL_POS})
        cube_height = ObsTerm(func=mdp.target_cube_height_above_table, params={"object_name": "cube_2", "table_z": TABLE_Z})
        cube_in_goal_xy = ObsTerm(
            func=mdp.cube_in_goal_region,
            params={"object_name": "cube_2", "goal_pos": GOAL_POS, "goal_half_extents_xy": GOAL_HALF_EXTENTS_XY},
        )

        
        
        gripper_force = ObsTerm(
            func=gripper_force_with_closure,
            params={
                "robot_name": "robot",
                "normalize": True,
                "effort_limit": 70.0,  # Franka finger max effort ~70N
            },
        )
        

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class MultiCamCfg(ObsGroup):

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


@configclass
class FrankaPickPlaceWithForceEnvCfg(FrankaPickPlaceEnvCfg):

    observations: ObservationsWithForceCfg = ObservationsWithForceCfg()

    def __post_init__(self):
        super().__post_init__()
        


@configclass
class ObservationsWithScalarForceCfg(ObservationsWithForceCfg):
    
    @configclass
    class ProprioCfg(ObservationsWithForceCfg.ProprioCfg):
        
        gripper_force = ObsTerm(
            func=gripper_force_scalar,
            params={"robot_name": "robot", "normalize": True, "effort_limit": 70.0},
        )

    proprio: ProprioCfg = ProprioCfg()


@configclass
class FrankaPickPlaceWithScalarForceEnvCfg(FrankaPickPlaceEnvCfg):
    observations: ObservationsWithScalarForceCfg = ObservationsWithScalarForceCfg()
    
    def __post_init__(self):
        super().__post_init__()


@configclass
class ObservationsWithGraspIndicatorCfg(ObservationsWithForceCfg):
    
    @configclass
    class ProprioCfg(ObservationsWithForceCfg.ProprioCfg):
        
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
    observations: ObservationsWithGraspIndicatorCfg = ObservationsWithGraspIndicatorCfg()
    
    def __post_init__(self):
        super().__post_init__()


import gymnasium as gym

gym.register(
    id="Isaac-Franka-PickPlace-Force-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FrankaPickPlaceWithForceEnvCfg,
        "rsl_rl_cfg_entry_point": f"{__name__.rsplit('.', 1)[0]}.agents.rsl_rl_ppo_cnn_cfg:FrankaPickPlacePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Franka-PickPlace-Force-Scalar-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FrankaPickPlaceWithScalarForceEnvCfg,
        "rsl_rl_cfg_entry_point": f"{__name__.rsplit('.', 1)[0]}.agents.rsl_rl_ppo_cnn_cfg:FrankaPickPlacePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Franka-PickPlace-Force-GraspIndicator-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FrankaPickPlaceWithGraspIndicatorEnvCfg,
        "rsl_rl_cfg_entry_point": f"{__name__.rsplit('.', 1)[0]}.agents.rsl_rl_ppo_cnn_cfg:FrankaPickPlacePPORunnerCfg",
    },
)


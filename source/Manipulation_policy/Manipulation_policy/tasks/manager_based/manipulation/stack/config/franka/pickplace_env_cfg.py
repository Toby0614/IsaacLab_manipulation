

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import FrameTransformerCfg, TiledCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from ... import mdp
from ...mdp import franka_stack_events
from ...mdp import grasp_rewards
from ...stack_env_cfg import StackEnvCfg

from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG


GOAL_POS = (0.21, 0.28, 0.0203)
GOAL_HALF_EXTENTS_XY = (0.05, 0.05)  # 10cm x 10cm goal region
TABLE_Z = 0.0203  # Table surface height
LIFT_HEIGHT = 0.07  # 7cm above table to count as "lifted"
GRASP_DIFF_THRESH_REW = 0.03


@configclass
class EventCfg:

    init_franka_arm_pose = EventTerm(
        func=franka_stack_events.set_default_joint_pose,
        mode="reset",
        params={
            "default_pose": [0.0444, -0.1894, -0.1107, -2.5148, 0.0044, 2.3775, 0.6952, 0.0400, 0.0400],
        },
    )

    randomize_franka_joint_state = EventTerm(
        func=franka_stack_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.02,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    randomize_cube_positions = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.40, 0.60),
                "y": (-0.10, 0.10),
                "z": (TABLE_Z, TABLE_Z),
                "yaw": (-0.8, 0.8),
            },
            "min_separation": 0.10,
            "asset_cfgs": [SceneEntityCfg("cube_1"), SceneEntityCfg("cube_2"), SceneEntityCfg("cube_3")],
        },
    )



@configclass
class ObservationsCfg:

    @configclass
    class ProprioCfg(ObsGroup):

        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)
        gripper_open_frac = ObsTerm(func=mdp.gripper_open_fraction, params={"robot_name": "robot"})
        goal_position = ObsTerm(func=mdp.goal_position, params={"goal_pos": GOAL_POS})
        cube_position = ObsTerm(func=mdp.target_cube_position, params={"object_name": "cube_2"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class MultiCamCfg(ObsGroup):

        multi_cam = ObsTerm(
            func=mdp.multi_cam_tensor_chw_with_dropout,  # ← CHANGED: dropout-aware version
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
class RewardsCfg:

    reaching_object = RewTerm(
        func=grasp_rewards.object_ee_distance_tanh,
        weight=3.0,  # Slightly increased to bootstrap approach
        params={
            "object_cfg": SceneEntityCfg("cube_2"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "std": 0.10,
        },
    )

    grasp_bonus = RewTerm(
        func=grasp_rewards.GraspStartBonusTerm,
        weight=15.0,  # Increased to make grasp milestone clear
        params={
            "robot_name": "robot",
            "ee_frame_name": "ee_frame",
            "object_name": "cube_2",
            "diff_threshold": GRASP_DIFF_THRESH_REW,
        },
    )

    grasp_hold = RewTerm(
        func=grasp_rewards.grasp_hold,
        weight=0.3,  # REDUCED to prevent "just hold" local optimum
        params={
            "robot_name": "robot",
            "ee_frame_name": "ee_frame",
            "object_name": "cube_2",
            "diff_threshold": GRASP_DIFF_THRESH_REW,
        },
    )

    close_when_near = RewTerm(
        func=grasp_rewards.close_when_near,
        weight=1.5,
        params={
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "object_cfg": SceneEntityCfg("cube_2"),
            "robot_name": "robot",
            "sigma": 0.05,
        },
    )

    lift_progress = RewTerm(
        func=grasp_rewards.HeightProgressWhenGraspedTerm,
        weight=20.0,  # REDUCED from 30 - was dominating goal approach
        params={
            "robot_name": "robot",
            "ee_frame_name": "ee_frame",
            "object_name": "cube_2",
            "diff_threshold": GRASP_DIFF_THRESH_REW,
            "table_z": TABLE_Z,
            "max_up_rate": 0.25,
        },
    )

    lift_capped = RewTerm(
        func=grasp_rewards.lift_to_transport_height,
        weight=4.0,  # REDUCED from 10 - was drowning out goal signal
        params={
            "robot_name": "robot",
            "ee_frame_name": "ee_frame",
            "object_name": "cube_2",
            "diff_threshold": GRASP_DIFF_THRESH_REW,
            "table_z": TABLE_Z,
            "target_height": LIFT_HEIGHT,  # 6cm
        },
    )

    goal_3d_distance = RewTerm(
        func=grasp_rewards.object_to_goal_distance_3d_when_grasped,
        weight=12.0,  # Strong signal to drive toward goal
        params={
            "robot_name": "robot",
            "ee_frame_name": "ee_frame",
            "object_name": "cube_2",
            "diff_threshold": GRASP_DIFF_THRESH_REW,
            "goal_pos": GOAL_POS,  # (0.70, 0.20, 0.0203) - Z at table level!
            "sigma": 0.15,
        },
    )

    lower_in_goal = RewTerm(
        func=grasp_rewards.lower_in_goal_region,
        weight=6.0,
        params={
            "object_cfg": SceneEntityCfg("cube_2"),
            "goal_pos": GOAL_POS,
            "goal_half_extents_xy": (0.08, 0.08),  # Slightly larger for approach
            "table_z": TABLE_Z,
            "target_height": LIFT_HEIGHT,
        },
    )

    stable_in_goal = RewTerm(
        func=grasp_rewards.StableInGoalAfterLiftTerm,
        weight=15.0,
        params={
            "robot_name": "robot",
            "ee_frame_name": "ee_frame",
            "object_name": "cube_2",
            "diff_threshold": GRASP_DIFF_THRESH_REW,
            "lift_height_thresh": 0.03,
            "goal_pos": GOAL_POS,
            "goal_half_extents_xy": GOAL_HALF_EXTENTS_XY,
            "table_z": TABLE_Z,
            "place_height_tol": 0.025,  # Slightly looser
            "vel_sigma": 0.20,  # Slightly more forgiving
        },
    )

    release_when_ready = RewTerm(
        func=grasp_rewards.ReleaseWhenReadyAfterLiftTerm,
        weight=20.0,
        params={
            "robot_name": "robot",
            "ee_frame_name": "ee_frame",
            "object_name": "cube_2",
            "diff_threshold": GRASP_DIFF_THRESH_REW,
            "lift_height_thresh": 0.03,
            "goal_pos": GOAL_POS,
            "goal_half_extents_xy": GOAL_HALF_EXTENTS_XY,
            "table_z": TABLE_Z,
            "place_height_tol": 0.025,
            "vel_thresh": 0.10,
        },
    )

    success_bonus = RewTerm(
        func=grasp_rewards.PickPlaceSuccessEventBonusTerm,
        weight=250.0,  # Large one-time bonus
        params={
            "robot_name": "robot",
            "ee_frame_name": "ee_frame",
            "object_name": "cube_2",
            "diff_threshold": GRASP_DIFF_THRESH_REW,
            "lift_height_thresh": 0.03,
            "goal_center": GOAL_POS,
            "goal_half_extents_xy": GOAL_HALF_EXTENTS_XY,
            "table_z": TABLE_Z,
            "place_height_tol": 0.035,  # Slightly more forgiving
            "vel_thresh": 0.12,
            "open_thresh": 0.55,
        },
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-5e-5)

    action_temporal_consistency = RewTerm(func=grasp_rewards.ActionSecondDifferenceL2Term, weight=-1e-4)
    
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-5)


    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2e-6)

    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-2e-7)

    object_motion = RewTerm(
        func=grasp_rewards.object_motion_l2_when_grasped,
        weight=-0.12,
        params={
            "robot_name": "robot",
            "ee_frame_name": "ee_frame",
            "object_name": "cube_2",
            "diff_threshold": GRASP_DIFF_THRESH_REW,
            "table_z": TABLE_Z,
            "lift_on": 0.02,
            "ang_vel_scale": 0.15,
        },
    )

    lift_straight_up = RewTerm(
        func=grasp_rewards.object_xy_speed_during_lift_penalty,
        weight=-0.15,
        params={
            "robot_name": "robot",
            "ee_frame_name": "ee_frame",
            "object_name": "cube_2",
            "diff_threshold": GRASP_DIFF_THRESH_REW,
            "table_z": TABLE_Z,
            "lift_on": 0.01,
            "lift_target": LIFT_HEIGHT,
        },
    )

    slam_near_goal = RewTerm(
        func=grasp_rewards.slam_penalty_near_goal,
        weight=-1.0,
        params={
            "object_name": "cube_2",
            "goal_pos": GOAL_POS,
            "sigma_goal": 0.18,
            "vz_thresh": 0.18,
        },
    )


@configclass
class TerminationsCfg:

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube_2")},
    )

    dropped_after_lift = DoneTerm(
        func=mdp.DropAfterLiftTerminationTerm,
        params={
            "robot_name": "robot",
            "ee_frame_name": "ee_frame",
            "object_name": "cube_2",
            "goal_center": GOAL_POS,
            "goal_half_extents_xy": GOAL_HALF_EXTENTS_XY,
            "table_z": TABLE_Z,
            "diff_threshold": 0.06,
            "lift_height_thresh": 0.03,
            "drop_height_thresh": 0.04,
            "near_table_tol": 0.02,
            "ungrasp_grace_steps": 2,
        },
    )

    success_grasp = DoneTerm(
        func=mdp.PickPlaceSuccessWithLiftHistoryTerm,
        params={
            "object_cfg": SceneEntityCfg("cube_2"),
            "goal_pos": GOAL_POS,
            "goal_half_extents_xy": GOAL_HALF_EXTENTS_XY,
            "table_z": TABLE_Z,
            "place_height_tol": 0.03,
            "robot_name": "robot",
            "ee_frame_name": "ee_frame",
            "diff_threshold": 0.06,
            "lift_height_thresh": 0.03,
            "open_thresh": 0.5,
        },
    )


@configclass
class FrankaPickPlaceEnvCfg(StackEnvCfg):

    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 7.0

        self.events = EventCfg()

        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]

        self.gripper_joint_names = ["panda_finger_.*"]
        self.gripper_open_val = 0.04
        self.gripper_threshold = 0.005

        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )

        cube_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        )

        self.scene.cube_1 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_1",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0.0, TABLE_Z], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=cube_properties,
                semantic_tags=[("class", "cube_1")],
            ),
        )
        self.scene.cube_2 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_2",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.55, 0.05, TABLE_Z], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=cube_properties,
                semantic_tags=[("class", "cube_2")],
            ),
        )
        self.scene.cube_3 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_3",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.60, -0.1, TABLE_Z], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/green_block.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=cube_properties,
                semantic_tags=[("class", "cube_3")],
            ),
        )

        self.scene.table.spawn.semantic_tags = [("class", "table")]
        self.scene.plane.semantic_tags = [("class", "ground")]

        goal_size = (2.0 * GOAL_HALF_EXTENTS_XY[0], 2.0 * GOAL_HALF_EXTENTS_XY[1], 0.002)
        self.scene.goal_region = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/GoalRegion",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[GOAL_POS[0], GOAL_POS[1], GOAL_POS[2]], rot=[1, 0, 0, 0]),
            spawn=sim_utils.CuboidCfg(
                size=goal_size,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.15, 0.05),  # Bright red-orange
                    emissive_color=(0.6, 0.1, 0.0),    # Red glow
                    roughness=0.3,
                    metallic=0.0,
                    opacity=0.45,  # Slightly more opaque for better visibility
                ),
                rigid_props=RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True,
                    solver_position_iteration_count=1,
                    solver_velocity_iteration_count=0,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            ),
        )
        self.scene.goal_region.spawn.semantic_tags = [("class", "goal_region")]

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.1034]),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
                    name="tool_rightfinger",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
                    name="tool_leftfinger",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
                ),
            ],
        )

        self.scene.table_cam = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/table_cam",
            update_period=0.0,
            height=64,
            width=64,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 2.5)
            ),
            offset=TiledCameraCfg.OffsetCfg(
                pos=(1.0, 0.0, 0.4), rot=(0.35355, -0.61237, -0.61237, 0.35355), convention="ros"
            ),
        )

        self.scene.wrist_cam = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
            update_period=0.0,
            height=64,
            width=64,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01, 2.0)
            ),
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.13, 0.0, -0.15), rot=(-0.70614, 0.03701, 0.03701, -0.70614), convention="ros"
            ),
        )

        self.image_obs_list = ["wrist_cam", "table_cam"]

        self.sim.use_fabric = True
        
        self.sim.device = "cuda"
        
        self.sim.render_interval = self.decimation
        
        self.scene.replicate_physics = True


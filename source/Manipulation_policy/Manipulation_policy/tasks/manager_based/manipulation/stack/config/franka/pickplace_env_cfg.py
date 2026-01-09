# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka Pick-and-Place Environment Configuration.

This is the SINGLE consolidated environment config for pick-and-place task.
Combines: scene setup, IK actions, cameras, rewards, terminations, and CNN observations.

Task: Pick up cube_2 and place it at the goal position.
"""

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


# =============================================================================
# TASK CONFIGURATION
# =============================================================================
# NOTE: Table is centered at (0.5, 0.0) (see `stack_env_cfg.py`), and cubes reset around:
#   x ∈ [0.40, 0.60], y ∈ [-0.10, 0.10]
# Fixed goal (kept simple; no goal randomization).
# Updated goal position: moved -0.45 in X, +0.5 in Y from previous (0.66, -0.22)
# New position: (0.21, 0.28) - on the left side of the table, forward from robot
GOAL_POS = (0.21, 0.28, 0.0203)
GOAL_HALF_EXTENTS_XY = (0.05, 0.05)  # 10cm x 10cm goal region
TABLE_Z = 0.0203  # Table surface height
# Slightly higher lift target to encourage a cleaner "lift up" before transport.
LIFT_HEIGHT = 0.07  # 7cm above table to count as "lifted"
# Grasp heuristic parameters (used by mdp.object_grasped)
# NOTE: mdp.object_grasped is a proximity heuristic, not a true contact grasp detector.
# If this threshold is too large, the policy can "farm" grasp reward by hovering near the cube with a closed gripper.
GRASP_DIFF_THRESH_REW = 0.03


# =============================================================================
# EVENTS (Reset randomization)
# =============================================================================
@configclass
class EventCfg:
    """Configuration for reset events and domain randomization.
    
    NOTE: Visual domain randomization (lighting, textures) is DISABLED by default
    because it uses the Replicator API which is extremely slow (causes 10-100x slowdown).
    Enable only for final sim2real fine-tuning, not during initial training.
    """

    # Robot initial pose
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

    # Cube randomization
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

    # =========================================================================
    # VISUAL DOMAIN RANDOMIZATION - DISABLED FOR PERFORMANCE
    # =========================================================================
    # These use Replicator API with synchronous USD operations = EXTREMELY SLOW.
    # Enable only for sim2real fine-tuning after initial policy training.
    # =========================================================================
    # randomize_light = EventTerm(...)
    # randomize_table_visual_material = EventTerm(...)


# =============================================================================
# OBSERVATIONS (CNN-compatible: proprio + multi_cam)
# =============================================================================
@configclass
class ObservationsCfg:
    """Observations split into proprio (vector) and multi_cam (images) for CNN policy."""

    @configclass
    class ProprioCfg(ObsGroup):
        """Proprioceptive observations (concatenated vector)."""

        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)
        gripper_open_frac = ObsTerm(func=mdp.gripper_open_fraction, params={"robot_name": "robot"})
        object = ObsTerm(func=mdp.object_obs)
        # Target cube velocities improve stability and help credit assignment for place/release.
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

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class MultiCamCfg(ObsGroup):
        """Combined camera observations: wrist RGB-D (4ch) + table RGB (3ch) = 7 channels."""

        multi_cam = ObsTerm(
            func=mdp.multi_cam_tensor_chw,
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
# REWARDS - REDESIGNED FOR BETTER GOAL APPROACH
# =============================================================================
# Key changes based on analysis of stalled training:
# 1. Reduced lift reward dominance (was causing "hold high forever" behavior)
# 2. Added 3D goal distance reward (so lowering toward table is rewarded!)
# 3. Simplified carry phase (removed complex sigmoid switching)
# 4. Added object velocity penalty during transport (reduces drops)
# =============================================================================
@configclass
class RewardsCfg:
    """Simplified staged reward structure.
    
    Design principles (from PDF + analysis):
    - 3D goal distance naturally rewards lowering (goal Z = table level)
    - Lift reward CAPS at transport height (no incentive to stay high)
    - Goal approach always active when grasped (not height-gated!)
    - Dense shaping throughout, sparse bonuses at key milestones
    
    Phases: reach → grasp → lift (capped) → goal_approach (3D) → place → release
    """

    # =========================================================================
    # PHASE 1: REACH (always active)
    # =========================================================================
    # Dense EE-to-object distance. Works well in current training.
    reaching_object = RewTerm(
        func=grasp_rewards.object_ee_distance_tanh,
        weight=3.0,  # Slightly increased to bootstrap approach
        params={
            "object_cfg": SceneEntityCfg("cube_2"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "std": 0.10,
        },
    )

    # =========================================================================
    # PHASE 2: GRASP (always active)
    # =========================================================================
    # One-time bonus for first grasp (prevents grasp farming without lifting)
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

    # Small per-step hold reward (kept small so lift/goal take over)
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

    # Close gripper when near (bootstraps grasp discovery)
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

    # =========================================================================
    # PHASE 3: LIFT (gated by grasp, CAPPED at transport height)
    # =========================================================================
    # Potential-based height progress - helps escape "grasp but never lift"
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

    # Capped lift reward: saturates at transport height!
    # KEY: Once at transport height, this gives constant reward - no incentive to go higher
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

    # =========================================================================
    # PHASE 4: GOAL APPROACH - 3D DISTANCE (THE KEY FIX!)
    # =========================================================================
    # Uses FULL 3D distance to goal. Goal Z = table level.
    # This naturally rewards LOWERING the cube toward the table!
    # - Hovering at 15cm above goal: low reward (far from goal in Z)
    # - Lowering to 3cm: higher reward (closer to goal)
    # - Placed at goal: maximum reward
    # NO HEIGHT GATE - active as soon as grasped!
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

    # Explicit lowering reward in goal region (reinforces the 3D distance signal)
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

    # =========================================================================
    # PHASE 5: PLACE & STABILIZE (gated by ever_lifted)
    # =========================================================================
    # Reward for being stable (low velocity) in goal region near table
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

    # =========================================================================
    # PHASE 6: RELEASE (gated by ever_lifted)
    # =========================================================================
    # Reward for opening gripper when properly placed
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

    # =========================================================================
    # PHASE 7: SUCCESS BONUS (one-time)
    # =========================================================================
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

    # =========================================================================
    # REGULARIZATION & PENALTIES
    # =========================================================================
    # Action rate penalty (smooth actions)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-5e-5)

    # Temporal consistency / action smoothness (2nd-order). This targets "wobbling" better than action_rate alone.
    action_temporal_consistency = RewTerm(func=grasp_rewards.ActionSecondDifferenceL2Term, weight=-1e-4)
    
    # Joint velocity penalty (prevents jerky motions that cause drops)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-5)

    # -------------------------------------------------------------------------
    # EXTRA ANTI-WOBBLE TERMS (ADDED ON TOP of the working reward system)
    # -------------------------------------------------------------------------
    # These are intentionally small and mostly act as "regularizers":
    # - discourage high-frequency joint oscillations (joint acceleration / torque)
    # - discourage shaking the object while carrying
    # - discourage slamming down near the goal during placement

    # Penalize joint accelerations (targets the vibration you observed).
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2e-6)

    # Penalize applied torques (reduces aggressive IK corrections / oscillations).
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-2e-7)

    # Penalize object motion once grasped + minimally lifted (reduces "wobble while carrying").
    # Weight reduced from -0.25 to -0.12 to avoid over-penalizing normal transport velocity
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

    # Penalize sideways motion during the lift band (encourages "lift straight up").
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

    # Penalize high downward velocity near the goal (encourages gentle placement).
    # Weight reduced from -2.0 to -1.0 to allow reasonable placement speed while still discouraging slamming
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


# =============================================================================
# TERMINATIONS
# =============================================================================
@configclass
class TerminationsCfg:
    """Termination conditions."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube_2")},
    )

    # Optional but useful: end episodes on mid-air drops after the robot has lifted the cube at least once.
    # This speeds up learning for simple pick-and-place without punishing valid release at the goal.
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

    # Success: REQUIRES lift history (prevents counting pushing as success)
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


# =============================================================================
# MAIN ENVIRONMENT CONFIG
# =============================================================================
@configclass
class FrankaPickPlaceEnvCfg(StackEnvCfg):
    """Complete Franka Pick-and-Place Environment Configuration.
    
    This is the ONLY env config needed for the pick-and-place task.
    """

    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Episode settings
        # Requested: increase from ~5s -> 7s to give the policy more time to complete smoother place.
        self.episode_length_s = 7.0

        # Events
        self.events = EventCfg()

        # =====================================================================
        # ROBOT SETUP
        # =====================================================================
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]

        # Gripper settings (required for grasp detection)
        self.gripper_joint_names = ["panda_finger_.*"]
        self.gripper_open_val = 0.04
        self.gripper_threshold = 0.005

        # IK actions
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

        # =====================================================================
        # SCENE SETUP (Cubes)
        # =====================================================================
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

        # Semantic tags
        self.scene.table.spawn.semantic_tags = [("class", "table")]
        self.scene.plane.semantic_tags = [("class", "ground")]

        # =====================================================================
        # GOAL REGION VISUALIZATION (colored box you can see)
        # =====================================================================
        # This is a kinematic, collision-disabled cuboid placed at the fixed goal.
        # Bright red/orange color for high visibility
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

        # =====================================================================
        # END EFFECTOR FRAME
        # =====================================================================
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

        # =====================================================================
        # CAMERAS (TiledCamera for CNN - 64x64)
        # =====================================================================
        # Table camera: RGB only (3 channels)
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

        # Wrist camera: RGB-D (4 channels)
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

        # =====================================================================
        # PERFORMANCE SETTINGS (CRITICAL FOR CAMERA-BASED TRAINING)
        # =====================================================================
        # Use fabric for fast USD operations
        self.sim.use_fabric = True
        
        # Device: use "cuda" (not "cuda:0") for best Replicator/RTX compatibility
        self.sim.device = "cuda"
        
        # CRITICAL: render_interval MUST be >= decimation to avoid multiple renders per step!
        # The base class sets this but it can get overridden. Explicitly set it here.
        self.sim.render_interval = self.decimation
        
        # Enable physics replication for faster scene creation
        # (Only works when NOT using per-env visual randomization)
        self.scene.replicate_physics = True


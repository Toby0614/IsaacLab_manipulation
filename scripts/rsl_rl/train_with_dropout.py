#!/usr/bin/env python
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Training script with modality dropout support.

This is a modified version of train.py that adds modality dropout capability
via environment wrapping. Your existing training code doesn't need to change!

Usage:
    # Train with hard dropout (M3 baseline):
    python scripts/rsl_rl/train_with_dropout.py --task=Isaac-Franka-PickPlace-Direct-v0 --dropout_mode=hard
    
    # Train with soft dropout (noise):
    python scripts/rsl_rl/train_with_dropout.py --task=Isaac-Franka-PickPlace-Direct-v0 --dropout_mode=soft
    
    # Train with phase-aware dropout:
    python scripts/rsl_rl/train_with_dropout.py --task=Isaac-Franka-PickPlace-Direct-v0 --dropout_mode=phase
    
    # Train without dropout (M1 baseline):
    python scripts/rsl_rl/train_with_dropout.py --task=Isaac-Franka-PickPlace-Direct-v0 --dropout_mode=none
"""

import argparse
import sys
import os

# Add Isaac Lab to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import cli_args for RSL-RL arguments
import cli_args


def add_dropout_args(parser: argparse.ArgumentParser):
    """Add modality dropout arguments to parser."""
    
    dropout_group = parser.add_argument_group("modality_dropout", "Modality dropout configuration")
    
    dropout_group.add_argument(
        "--dropout_mode",
        type=str,
        default="none",
        choices=["none", "hard", "soft", "mixed", "phase", "eval"],
        help="Dropout mode: 'none' (disabled), 'hard' (blackout), 'soft' (noise), "
             "'mixed' (both), 'phase' (phase-aware), 'eval' (deterministic for testing)",
    )
    
    dropout_group.add_argument(
        "--dropout_prob",
        type=float,
        default=None,
        help="Override dropout probability (chance per step to start new dropout event). "
             "Default depends on mode.",
    )
    
    dropout_group.add_argument(
        "--dropout_duration_min",
        type=int,
        default=None,
        help="Minimum dropout duration in steps (at 20Hz: 10 steps = 0.5s)",
    )
    
    dropout_group.add_argument(
        "--dropout_duration_max",
        type=int,
        default=None,
        help="Maximum dropout duration in steps (at 20Hz: 60 steps = 3.0s)",
    )
    
    dropout_group.add_argument(
        "--enable_phase_detection",
        action="store_true",
        help="Enable automatic phase detection (for phase-aware dropout)",
    )
    
    dropout_group.add_argument(
        "--dropout_rgb",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Apply dropout to RGB channels (default: True)",
    )
    
    dropout_group.add_argument(
        "--dropout_depth",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Apply dropout to depth channel (default: True)",
    )
    
    return parser


def create_dropout_config(args):
    """Create dropout configuration from command line arguments."""
    
    # Import here to avoid issues if module not found
    from source.Manipulation_policy.Manipulation_policy.tasks.manager_based.manipulation.stack.mdp.modality_dropout_cfg import (
        ModalityDropoutCfg,
        HardDropoutTrainingCfg,
        SoftDropoutTrainingCfg,
        MixedDropoutTrainingCfg,
        PhaseBasedDropoutCfg,
        EvalDropoutCfg,
    )
    
    # Select base config
    if args.dropout_mode == "none":
        cfg = ModalityDropoutCfg(enabled=False)
    elif args.dropout_mode == "hard":
        cfg = HardDropoutTrainingCfg()
    elif args.dropout_mode == "soft":
        cfg = SoftDropoutTrainingCfg()
    elif args.dropout_mode == "mixed":
        cfg = MixedDropoutTrainingCfg()
    elif args.dropout_mode == "phase":
        cfg = PhaseBasedDropoutCfg()
    elif args.dropout_mode == "eval":
        cfg = EvalDropoutCfg()
    else:
        raise ValueError(f"Unknown dropout mode: {args.dropout_mode}")
    
    # Apply overrides
    if args.dropout_prob is not None:
        cfg.dropout_probability = args.dropout_prob
    
    if args.dropout_duration_min is not None and args.dropout_duration_max is not None:
        cfg.dropout_duration_range = (args.dropout_duration_min, args.dropout_duration_max)
    
    cfg.dropout_rgb = args.dropout_rgb
    cfg.dropout_depth = args.dropout_depth
    
    return cfg


def wrap_env_with_dropout(env, args):
    """Wrap environment with dropout if enabled."""
    
    if args.dropout_mode == "none":
        print("[INFO] Modality dropout DISABLED")
        return env
    
    # Import dropout components
    from source.Manipulation_policy.Manipulation_policy.tasks.manager_based.manipulation.stack.mdp.dropout_env_wrapper import (
        VecEnvDropoutWrapper,
    )
    
    # Create dropout config
    dropout_cfg = create_dropout_config(args)
    
    # Determine if phase detection is needed
    enable_phase_detection = args.enable_phase_detection or dropout_cfg.phase_aware
    
    # Phase detector kwargs (customize for your task)
    phase_detector_kwargs = {
        "goal_pos": (0.21, 0.28, 0.0203),  # Your goal position
        "table_z": 0.0203,
        "lift_threshold": 0.05,
        "grasp_dist_threshold": 0.06,
        "goal_xy_radius": 0.10,
    }
    
    # Wrap environment
    env = VecEnvDropoutWrapper(
        env,
        dropout_cfg=dropout_cfg,
        enable_phase_detection=enable_phase_detection,
        phase_detector_kwargs=phase_detector_kwargs if enable_phase_detection else None,
    )
    
    print(f"[INFO] Modality dropout ENABLED:")
    print(f"  - Mode: {dropout_cfg.dropout_mode}")
    print(f"  - Probability: {dropout_cfg.dropout_probability:.3f}")
    print(f"  - Duration range: {dropout_cfg.dropout_duration_range} steps")
    print(f"  - Phase aware: {dropout_cfg.phase_aware}")
    print(f"  - RGB dropout: {dropout_cfg.dropout_rgb}")
    print(f"  - Depth dropout: {dropout_cfg.dropout_depth}")
    
    return env


def main():
    """Main training function with dropout support."""
    
    # Parse arguments (but don't launch Isaac Sim yet)
    parser = argparse.ArgumentParser(description="Train RL agent with modality dropout.")
    
    # Add training arguments
    parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
    parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
    parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
    parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
    
    # Add RSL-RL arguments
    cli_args.add_rsl_rl_args(parser)
    
    # Add dropout arguments
    add_dropout_args(parser)
    
    # Add AppLauncher arguments
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    
    args_cli, hydra_args = parser.parse_known_args()
    
    # Always enable cameras for vision-based tasks
    _task = getattr(args_cli, "task", "") or ""
    if any(k in _task for k in ["Visuomotor", "Camera", "RGB", "Depth", "PickPlace"]):
        args_cli.enable_cameras = True
    
    # Clear sys.argv for Hydra
    sys.argv = [sys.argv[0]] + hydra_args
    
    # Launch Isaac Sim
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    
    # Now import everything else (after Isaac Sim is launched)
    import gymnasium as gym
    import torch
    from datetime import datetime
    
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab.envs import ManagerBasedRLEnvCfg, DirectRLEnvCfg
    from isaaclab.utils.dict import print_dict
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils.hydra import hydra_task_config
    
    import Manipulation_policy.tasks  # noqa: F401
    
    # Get config via Hydra
    @hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
    def get_config(env_cfg, agent_cfg):
        return env_cfg, agent_cfg
    
    env_cfg, agent_cfg = get_config()
    
    # Apply CLI overrides
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    env_cfg.seed = agent_cfg.seed
    
    # Setup logging
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    env_cfg.log_dir = log_dir
    
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    print(f"[INFO] Run directory: {log_dir}")
    
    # Create environment
    print(f"[INFO] Creating environment: {args_cli.task}")
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    # Wrap with dropout if enabled
    env = wrap_env_with_dropout(env, args_cli)
    
    # Wrap for RSL-RL
    env = RslRlVecEnvWrapper(env)
    
    # Create runner and train
    print(f"[INFO] Creating PPO runner...")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    
    print(f"[INFO] Starting training for {agent_cfg.max_iterations} iterations...")
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    
    # Close
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()


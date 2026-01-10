# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--export_policy",
    action="store_true",
    default=False,
    help=(
        "Export the loaded policy to JIT/ONNX before playing. "
        "Note: for some CNN+fusion policies, ONNX export can fail due to normalizer/actor input shape mismatch."
    ),
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# modality dropout arguments (optional, for robustness evaluation/visualization)
parser.add_argument(
    "--dropout_mode",
    type=str,
    default="none",
    choices=["none", "hard", "soft", "mixed"],
    help="Modality dropout mode during play: none (disabled), hard (blackout), soft (noise), mixed (both).",
)
parser.add_argument("--dropout_prob", type=float, default=None, help="Dropout probability per step (override preset).")
parser.add_argument("--dropout_duration_min", type=int, default=None, help="Min dropout duration (steps).")
parser.add_argument("--dropout_duration_max", type=int, default=None, help="Max dropout duration (steps).")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
# (defensive) strip stray tokens that Hydra's override grammar can't parse (e.g., standalone "\")
def _sanitize_hydra_overrides(overrides: list[str]) -> list[str]:
    sanitized: list[str] = []
    for a in overrides:
        if a is None:
            continue
        s = str(a)
        if s.strip() == "":
            continue
        if s.strip("\\").strip() == "":
            continue
        sanitized.append(s)
    return sanitized


hydra_args = _sanitize_hydra_overrides(hydra_args)
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
import glob

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import Manipulation_policy.tasks  # noqa: F401


def _resolve_checkpoint_arg(checkpoint_arg: str) -> str:
    """Resolve a user-provided checkpoint argument to a concrete .pt file.

    Users sometimes pass a *run directory* (e.g., '/tmp/M3_mix' or a log folder)
    instead of a specific model file. RSL-RL expects a file path for torch.load().
    """
    if checkpoint_arg is None:
        raise ValueError("checkpoint_arg is None")

    p = retrieve_file_path(checkpoint_arg)
    if os.path.isfile(p):
        return p

    if not os.path.isdir(p):
        raise FileNotFoundError(f"Checkpoint path does not exist: {p}")

    # Common patterns in this repo/logs
    candidates: list[str] = []
    for pat in ("model_*.pt", "model*.pt", "*.pt"):
        candidates.extend(glob.glob(os.path.join(p, pat)))

    # Filter out non-files just in case and sort by mtime (newest first)
    candidates = [c for c in candidates if os.path.isfile(c)]
    candidates.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    if not candidates:
        raise FileNotFoundError(
            f"Checkpoint argument '{checkpoint_arg}' resolved to directory '{p}', but no .pt files were found inside."
        )

    chosen = candidates[0]
    print(f"[INFO] Resolved checkpoint directory '{p}' -> '{chosen}'")
    return chosen


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = _resolve_checkpoint_arg(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for modality dropout (evaluation during play)
    if args_cli.dropout_mode != "none":
        try:
            from Manipulation_policy.tasks.manager_based.manipulation.stack.mdp.dropout_env_wrapper import (
                VecEnvDropoutWrapper,
            )
            from Manipulation_policy.tasks.manager_based.manipulation.stack.mdp.modality_dropout_cfg import (
                HardDropoutTrainingCfg,
                SoftDropoutTrainingCfg,
                MixedDropoutTrainingCfg,
            )

            if args_cli.dropout_mode == "hard":
                dropout_cfg = HardDropoutTrainingCfg()
            elif args_cli.dropout_mode == "soft":
                dropout_cfg = SoftDropoutTrainingCfg()
            elif args_cli.dropout_mode == "mixed":
                dropout_cfg = MixedDropoutTrainingCfg()

            # Apply CLI overrides
            if args_cli.dropout_prob is not None:
                dropout_cfg.dropout_probability = args_cli.dropout_prob
            if args_cli.dropout_duration_min is not None and args_cli.dropout_duration_max is not None:
                dropout_cfg.dropout_duration_range = (args_cli.dropout_duration_min, args_cli.dropout_duration_max)

            env = VecEnvDropoutWrapper(env, dropout_cfg)

            print("=" * 80)
            print("[INFO] Modality Dropout ENABLED during play:")
            print(f"  Mode: {dropout_cfg.dropout_mode}")
            print(f"  Probability: {dropout_cfg.dropout_probability:.3f}")
            print(f"  Duration range: {dropout_cfg.dropout_duration_range} steps")
            print(f"  RGB dropout: {dropout_cfg.dropout_rgb}")
            print(f"  Depth dropout: {dropout_cfg.dropout_depth}")
            print("=" * 80)
        except Exception as e:
            print(f"[ERROR] Failed to initialize dropout wrapper for play: {e}")
            print("[WARNING] Continuing without dropout during play.")

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit (optional)
    if args_cli.export_policy:
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        try:
            export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        except Exception as e:
            print(f"[WARN] Failed to export policy as JIT. Continuing without export. Error: {e}")
        try:
            export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")
        except Exception as e:
            print(f"[WARN] Failed to export policy as ONNX. Continuing without export. Error: {e}")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

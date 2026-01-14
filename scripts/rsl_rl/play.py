# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import re

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

# Force sensing (for M2/M4 policies)
parser.add_argument(
    "--force_sensing", action="store_true", default=False,
    help="Enable gripper force sensing (required for M2/M4 policies)."
)
parser.add_argument(
    "--force_mode", type=str, default="grasp_indicator",
    choices=["scalar", "per_finger", "with_closure", "contact_estimate", "grasp_indicator"],
    help="Force sensing mode (must match training config)."
)

# =============================================================================
# EVALUATION MODE: Systematic dropout robustness evaluation (Variant 1 & 2)
# =============================================================================
parser.add_argument(
    "--eval_dropout",
    action="store_true",
    default=False,
    help="Run systematic dropout evaluation (exits automatically when done).",
)
parser.add_argument(
    "--eval_variant",
    type=str,
    default="both",
    choices=["1", "2", "both"],
    help="Evaluation variant: 1 (phase-based), 2 (time-based), or both.",
)
parser.add_argument(
    "--eval_phases",
    type=str,
    default="reach,grasp,lift,transport,place",
    help="Comma-separated phases for Variant 1 evaluation.",
)
parser.add_argument(
    "--eval_onset_steps",
    type=str,
    default="10,25,50,75,100,125,150",
    help="Comma-separated onset steps for Variant 2 evaluation.",
)
parser.add_argument(
    "--eval_durations",
    type=str,
    default="5,10,20,40,60,80,100",
    help="Comma-separated dropout durations (steps) for evaluation.",
)
parser.add_argument(
    "--eval_episodes",
    type=int,
    default=100,
    help="Number of episodes per evaluation condition.",
)
parser.add_argument(
    "--eval_output_dir",
    type=str,
    default="results/dropout_eval",
    help="Directory to save evaluation results.",
)
parser.add_argument(
    "--policy_name",
    type=str,
    default=None,
    help="Policy name for evaluation results (auto-detected if not provided).",
)

# =============================================================================
# EVALUATION MODE: Systematic POSE CORRUPTION robustness evaluation
# =============================================================================
parser.add_argument(
    "--eval_pose_corruption",
    action="store_true",
    default=False,
    help="Run systematic pose-corruption evaluation on oracle cube_position (exits automatically when done).",
)
parser.add_argument(
    "--pose_eval_variant",
    type=str,
    default="both",
    choices=["1", "2", "both"],
    help="Pose-eval variant: 1 (phase-based), 2 (time-based), or both.",
)
parser.add_argument(
    "--pose_eval_phases",
    type=str,
    default="reach,grasp,lift,transport,place",
    help="Comma-separated phases for Variant 1 pose evaluation.",
)
parser.add_argument(
    "--pose_eval_onset_steps",
    type=str,
    default="10,25,50,75,100,125,150",
    help="Comma-separated onset steps for Variant 2 pose evaluation.",
)
parser.add_argument(
    "--pose_eval_durations",
    type=str,
    default="5,10,20,40,60,80,100",
    help="Comma-separated corruption durations (steps) for pose evaluation.",
)
parser.add_argument(
    "--pose_eval_modes",
    type=str,
    default="hard,freeze",
    help="Comma-separated pose corruption modes to evaluate (e.g., 'hard,freeze,noise,delay').",
)
parser.add_argument(
    "--pose_eval_episodes",
    type=int,
    default=200,
    help="Number of episodes per evaluation condition for pose corruption.",
)
parser.add_argument(
    "--pose_eval_output_dir",
    type=str,
    default="results/pose_eval",
    help="Directory to save pose-corruption evaluation results.",
)
parser.add_argument("--pose_eval_delay_steps", type=int, default=5, help="Delay steps used when pose mode is 'delay'.")
parser.add_argument("--pose_eval_noise_std", type=float, default=0.01, help="Pose noise std (meters) used for 'noise'.")
parser.add_argument("--pose_eval_drift_noise_std", type=float, default=0.001, help="Pose drift noise std (meters) used for 'noise'.")

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
        # Strip bash-style comment tokens that sometimes sneak into multi-line scripts.
        # Hydra override grammar cannot parse a raw '#' token.
        if s.lstrip().startswith("#"):
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
import json
import numpy as np
from datetime import datetime
from dataclasses import dataclass, asdict

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


# =============================================================================
# EVALUATION DATA STRUCTURES
# =============================================================================

@dataclass
class EvalResult:
    """Single evaluation condition result."""
    policy_name: str
    variant: int
    condition: str  # phase name or onset step
    duration: int
    success_rate: float
    mean_episode_length: float
    std_episode_length: float
    num_episodes: int
    dropout_triggered_rate: float
    # Pose-corruption specific diagnostics (kept optional-ish via defaults so dropout eval doesn't break)
    corruption_step_rate: float = 0.0
    corruption_episode_rate: float = 0.0


@dataclass
class EvalSummary:
    """Summary of evaluation run."""
    policy_name: str
    variant: int
    conditions: list
    durations: list
    success_matrix: list  # 2D: [condition x duration]
    timestamp: str
    seed: int
    num_episodes_per_condition: int


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def run_evaluation_episodes(
    env,
    policy,
    policy_nn,
    num_episodes: int,
    device: str,
) -> dict:
    """Run evaluation episodes and collect metrics."""
    num_envs = env.unwrapped.num_envs
    
    successes = []
    episode_lengths = []
    dropout_triggered = []
    
    episodes_completed = 0
    obs = env.get_observations()
    env_episode_step = torch.zeros(num_envs, dtype=torch.int32, device=device)
    
    while episodes_completed < num_episodes:
        with torch.inference_mode():
            actions = policy(obs)
            # IsaacLab wrappers may follow either Gymnasium API:
            #   obs, reward, terminated, truncated, info
            # or the older VecEnv-style API:
            #   obs, reward, done, info
            step_out = env.step(actions)
            if isinstance(step_out, (tuple, list)) and len(step_out) == 5:
                obs, rewards, terminated, truncated, info = step_out
                done = terminated | truncated
            elif isinstance(step_out, (tuple, list)) and len(step_out) == 4:
                obs, rewards, done, info = step_out
                if not torch.is_tensor(done):
                    done = torch.as_tensor(done, device=device)
                done = done.to(dtype=torch.bool)

                # Best-effort split of `done` into terminated vs truncated using timeout signals in info.
                time_outs = None
                if isinstance(info, dict):
                    for k in ("time_outs", "timeouts", "time_out"):
                        if k in info:
                            time_outs = info[k]
                            break
                    if time_outs is None and isinstance(info.get("extras"), dict):
                        for k in ("time_outs", "timeouts", "time_out"):
                            if k in info["extras"]:
                                time_outs = info["extras"][k]
                                break
                if time_outs is not None:
                    if not torch.is_tensor(time_outs):
                        time_outs = torch.as_tensor(time_outs, device=device)
                    truncated = time_outs.to(dtype=torch.bool)
                    terminated = done & ~truncated
                else:
                    truncated = torch.zeros_like(done, dtype=torch.bool)
                    terminated = done
            else:
                raise ValueError(f"Unexpected env.step() return of length {len(step_out) if isinstance(step_out, (tuple, list)) else type(step_out)}")
            env_episode_step += 1
            
            done_indices = done.nonzero(as_tuple=False).squeeze(-1)
            
            for idx in done_indices.cpu().tolist():
                if episodes_completed >= num_episodes:
                    break
                
                # Determine success.
                #
                # IMPORTANT:
                # - In IsaacLab, `terminated=True` can mean *any* termination condition fired
                #   (success OR failure), while `truncated=True` usually means timeout.
                # - For robustness eval we want true task success, not "episode ended".
                success = False
                try:
                    base_unwrapped = env.unwrapped
                    if hasattr(base_unwrapped, "termination_manager"):
                        # `success_grasp` is the success termination term configured in
                        # `.../config/franka/pickplace_env_cfg.py` for this task.
                        term = base_unwrapped.termination_manager.get_term("success_grasp")
                        if torch.is_tensor(term):
                            success = bool(term[idx].item())
                        else:
                            success = bool(term)
                    else:
                        success = bool(terminated[idx].item())
                except Exception:
                    success = bool(terminated[idx].item())
                successes.append(float(success))
                episode_lengths.append(env_episode_step[idx].item())
                
                # Check dropout trigger
                base_env = env
                while hasattr(base_env, 'env'):
                    if hasattr(base_env, 'dropout_manager'):
                        break
                    base_env = base_env.env
                
                triggered = False
                if hasattr(base_env, 'dropout_manager'):
                    triggered = base_env.dropout_manager.dropout_triggered_count[idx].item() > 0
                dropout_triggered.append(float(triggered))
                
                episodes_completed += 1
            
            if len(done_indices) > 0:
                env_episode_step[done_indices] = 0
            
            policy_nn.reset(done)
    
    return {
        "success_rate": np.mean(successes) if successes else 0.0,
        "mean_episode_length": np.mean(episode_lengths) if episode_lengths else 0.0,
        "std_episode_length": np.std(episode_lengths) if episode_lengths else 0.0,
        "num_episodes": len(successes),
        "dropout_triggered_rate": np.mean(dropout_triggered) if dropout_triggered else 0.0,
    }


def run_variant1_eval_shared(
    env,
    policy,
    policy_nn,
    policy_name: str,
    phases: list,
    durations: list,
    num_episodes: int,
    device: str,
    seed: int,
) -> tuple:
    """Run Variant 1 (phase-based) evaluation using shared environment."""
    from Manipulation_policy.tasks.manager_based.manipulation.stack.mdp.eval_dropout_wrapper import (
        Variant1PhaseDropoutManager,
    )
    from Manipulation_policy.tasks.manager_based.manipulation.stack.mdp.eval_dropout_cfg import (
        Variant1PhaseDropoutCfg,
    )
    from Manipulation_policy.tasks.manager_based.manipulation.stack.mdp.phase_detector import (
        PickPlacePhaseDetector,
    )
    
    results = []
    success_matrix = []
    
    print(f"\n{'='*60}")
    print(f"VARIANT 1 (Phase-Based) Evaluation: {policy_name}")
    print(f"Phases: {phases}, Durations: {durations}")
    print(f"{'='*60}")
    
    # Create phase detector
    phase_detector = PickPlacePhaseDetector()
    
    print("  Starting evaluation grid...", flush=True)
    
    for phase in phases:
        phase_results = []
        
        for duration in durations:
            print(f"  Phase={phase}, Duration={duration}...", end=" ", flush=True)
            
            # Create NEW dropout manager with this config
            cfg = Variant1PhaseDropoutCfg(
                target_phase=phase,
                dropout_duration_steps=duration,
            )
            dropout_manager = Variant1PhaseDropoutManager(
                cfg=cfg,
                num_envs=env.unwrapped.num_envs,
                device=device,
            )
            
            # Attach to environment
            env.unwrapped.dropout_manager = dropout_manager
            
            # Run evaluation with phase detection
            metrics = run_evaluation_episodes_v1(
                env, policy, policy_nn, num_episodes, device, 
                dropout_manager, phase_detector
            )
            
            result = EvalResult(
                policy_name=policy_name,
                variant=1,
                condition=phase,
                duration=duration,
                **metrics,
            )
            results.append(result)
            phase_results.append(metrics["success_rate"])
            
            print(f"Success: {metrics['success_rate']:.1%}, Dropout triggered: {metrics['dropout_triggered_rate']:.1%}")
        
        success_matrix.append(phase_results)
    
    summary = EvalSummary(
        policy_name=policy_name,
        variant=1,
        conditions=phases,
        durations=durations,
        success_matrix=success_matrix,
        timestamp=datetime.now().isoformat(),
        seed=seed,
        num_episodes_per_condition=num_episodes,
    )
    
    return summary, results


def apply_dropout_to_obs(obs, dropout_manager, device: str):
    """Manually apply dropout (blackout) to camera/visual observations.
    
    Handles both dict/TensorDict and tensor observations.
    For dict: looks for keys containing camera-related terms or 4D image tensors
    For tensor: applies dropout if tensor looks like images (4D)
    """
    if not dropout_manager.dropout_active.any():
        return obs
    
    mask = dropout_manager.dropout_active  # (B,)
    
    def apply_blackout(tensor: torch.Tensor, force: bool = False) -> torch.Tensor:
        """Apply blackout to a tensor.
        
        Args:
            tensor: Input tensor
            force: If True, always apply blackout to 4D tensors (for explicit camera keys)
                   If False, only apply if tensor looks like typical image data
        """
        if not torch.is_tensor(tensor):
            return tensor
        if tensor.ndim != 4:
            return tensor
        
        # If not forced, check if this looks like image data (B, C, H, W) or (B, H, W, C)
        # Multi-camera setups can have many channels (e.g., 7 = 2 cams × 3 RGB + 1 depth)
        if not force:
            b, d1, d2, d3 = tensor.shape
            # Heuristic: spatial dims (H, W) are usually larger than channel dim
            # If d2 and d3 are both >= 16, it's likely (B, C, H, W) format
            is_image = (d2 >= 16 and d3 >= 16) or (d1 <= 4) or (d3 <= 4)
            if not is_image:
                return tensor
        
        val_out = tensor.clone()
        broadcast_mask = mask.view(-1, 1, 1, 1)
        val_out = torch.where(broadcast_mask, torch.zeros_like(val_out), val_out)
        return val_out
    
    # Handle dict-like observations (including TensorDict)
    # TensorDict from tensordict library behaves like a dict
    from collections.abc import Mapping
    if isinstance(obs, Mapping):
        # Expanded list of camera-related key substrings
        # 'cam' matches 'multi_cam', 'camera' matches 'wrist_camera', etc.
        camera_keywords = ['cam', 'camera', 'image', 'rgb', 'depth', 'visual', 'pixels', 'img']
        
        # Try to preserve TensorDict type if possible
        try:
            from tensordict import TensorDict
            is_tensordict = isinstance(obs, TensorDict)
        except ImportError:
            is_tensordict = False
        
        obs_out = {}
        for key in obs.keys():
            val = obs[key]
            key_lower = key.lower()
            
            # Check if key suggests camera/visual data
            is_camera_key = any(kw in key_lower for kw in camera_keywords)
            
            if is_camera_key and torch.is_tensor(val) and val.ndim == 4:
                # Explicit camera key - force blackout regardless of channel count
                obs_out[key] = apply_blackout(val, force=True)
            elif torch.is_tensor(val) and val.ndim == 4:
                # Unknown key but 4D tensor - use heuristic to check if it's image-like
                obs_out[key] = apply_blackout(val, force=False)
            else:
                obs_out[key] = val
        
        # Convert back to TensorDict if input was TensorDict
        if is_tensordict:
            return TensorDict(obs_out, batch_size=obs.batch_size)
        return obs_out
    
    # Handle tensor observations (might be concatenated)
    elif torch.is_tensor(obs):
        return apply_blackout(obs)
    
    return obs


def run_evaluation_episodes_v1(
    env,
    policy,
    policy_nn,
    num_episodes: int,
    device: str,
    dropout_manager,
    phase_detector,
) -> dict:
    """Run evaluation episodes for Variant 1 with phase detection."""
    num_envs = env.unwrapped.num_envs
    
    successes = []
    episode_lengths = []
    dropout_triggered = []
    
    episodes_completed = 0
    
    # Reset dropout manager for this condition
    all_env_ids = torch.arange(num_envs, device=device)
    dropout_manager.reset(all_env_ids)
    
    # Get current observations (don't reset env - causes inference mode issues)
    obs = env.get_observations()
    
    # Debug: show observation structure (once per variant)
    if not hasattr(run_evaluation_episodes_v1, '_obs_structure_printed'):
        run_evaluation_episodes_v1._obs_structure_printed = True
        print(f"  [DEBUG] Observation type: {type(obs).__name__}")
        if hasattr(obs, 'keys'):
            camera_keywords = ['cam', 'camera', 'image', 'rgb', 'depth', 'visual', 'pixels', 'img']
            for key in obs.keys():
                val = obs[key]
                shape = val.shape if torch.is_tensor(val) else "N/A"
                is_camera = any(kw in key.lower() for kw in camera_keywords) or (torch.is_tensor(val) and val.ndim == 4)
                marker = " <-- WILL BLACKOUT" if is_camera else ""
                print(f"    Key: '{key}', Shape: {shape}{marker}")
    
    env_episode_step = torch.zeros(num_envs, dtype=torch.int32, device=device)
    
    while episodes_completed < num_episodes:
        with torch.inference_mode():
            # Update phases and dropout state BEFORE getting actions
            try:
                phases = phase_detector.detect_phases(env.unwrapped)
                dropout_manager.update_phases(phases)
            except Exception:
                pass
            dropout_manager.step()
            
            # Apply dropout to observations
            obs_with_dropout = apply_dropout_to_obs(obs, dropout_manager, device)
            
            actions = policy(obs_with_dropout)
            step_out = env.step(actions)
            
            if isinstance(step_out, (tuple, list)) and len(step_out) == 5:
                obs, rewards, terminated, truncated, info = step_out
                done = terminated | truncated
            elif isinstance(step_out, (tuple, list)) and len(step_out) == 4:
                obs, rewards, done, info = step_out
                if not torch.is_tensor(done):
                    done = torch.as_tensor(done, device=device)
                done = done.to(dtype=torch.bool)
                truncated = torch.zeros_like(done, dtype=torch.bool)
                terminated = done
            else:
                raise ValueError(f"Unexpected env.step() return")
            
            env_episode_step += 1
            
            done_indices = done.nonzero(as_tuple=False).squeeze(-1)
            
            for idx in done_indices.cpu().tolist():
                if episodes_completed >= num_episodes:
                    break
                
                # See `run_evaluation_episodes` for rationale: only count true task success.
                success = False
                try:
                    if hasattr(env.unwrapped, "termination_manager"):
                        term = env.unwrapped.termination_manager.get_term("success_grasp")
                        if torch.is_tensor(term):
                            success = bool(term[idx].item())
                        else:
                            success = bool(term)
                    else:
                        success = bool(terminated[idx].item())
                except Exception:
                    success = bool(terminated[idx].item())
                successes.append(float(success))
                episode_lengths.append(env_episode_step[idx].item())
                
                triggered = dropout_manager.dropout_triggered_count[idx].item() > 0
                dropout_triggered.append(float(triggered))
                
                episodes_completed += 1
            
            if len(done_indices) > 0:
                env_episode_step[done_indices] = 0
                dropout_manager.reset(done_indices)
            
            policy_nn.reset(done)
    
    return {
        "success_rate": np.mean(successes) if successes else 0.0,
        "mean_episode_length": np.mean(episode_lengths) if episode_lengths else 0.0,
        "std_episode_length": np.std(episode_lengths) if episode_lengths else 0.0,
        "num_episodes": len(successes),
        "dropout_triggered_rate": np.mean(dropout_triggered) if dropout_triggered else 0.0,
    }


def _infer_force_dim(force_mode: str) -> int:
    force_dims = {
        "scalar": 1,
        "per_finger": 2,
        "with_closure": 3,
        "contact_estimate": 2,
        "grasp_indicator": 3,
    }
    return int(force_dims.get(force_mode, 0))


def apply_pose_corruption_to_obs(obs, pose_manager, force_dim: int, env_unwrapped=None):
    """Corrupt the cube *world position* embedded inside obs['proprio'].

    IMPORTANT:
    - The actual `proprio` vector for `Isaac-Franka-PickPlace-v0` is a concatenation of many ObsTerms
      (see `.../config/franka/pickplace_env_cfg*.py`). The cube position is NOT guaranteed to be the
      last 3 dims.
    - Older code assumed "cube_position is at the end", which silently corrupted the wrong features
      and produced misleading "Pose corrupted: 0%" diagnostics.

    This function therefore supports two modes:
    - If the unwrapped env exposes a cached `__pose_eval_cube_pos_indices`, we use it.
    - Otherwise we fall back to the legacy "last 3 dims" heuristic (for older checkpoints/envs).
    """
    if obs is None:
        return obs
    if not hasattr(obs, "keys") or "proprio" not in obs:
        return obs

    proprio = obs["proprio"]
    if not torch.is_tensor(proprio) or proprio.ndim != 2:
        return obs

    if proprio.shape[1] < (3 + max(force_dim, 0)):
        return obs

    # NOTE: We infer indices lazily because `proprio` ordering depends on the env config and wrappers.
    cube_indices = getattr(env_unwrapped, "__pose_eval_cube_pos_indices", None) if env_unwrapped is not None else None

    # Legacy fallback: assume cube pos is right before force dims (if any).
    if cube_indices is None:
        if force_dim > 0:
            cube_slice = slice(-(force_dim + 3), -force_dim)
        else:
            cube_slice = slice(-3, None)
        cube_pos = proprio[:, cube_slice]
        cube_pos_corrupt = pose_manager.apply(cube_pos)
        obs["proprio"] = (
            torch.cat([proprio[:, : cube_slice.start], cube_pos_corrupt, proprio[:, cube_slice.stop :]], dim=1)
            if force_dim > 0
            else torch.cat([proprio[:, :-3], cube_pos_corrupt], dim=1)
        )
        return obs

    # New robust path: overwrite the identified cube position columns (xyz).
    idx = torch.as_tensor(cube_indices, device=proprio.device, dtype=torch.long)  # (3,)
    cube_pos = proprio.index_select(dim=1, index=idx)  # (B,3)
    cube_pos_corrupt = pose_manager.apply(cube_pos)
    proprio_out = proprio.clone()
    proprio_out[:, idx] = cube_pos_corrupt
    obs["proprio"] = proprio_out
    return obs


def _infer_cube_pos_indices(env_unwrapped, proprio: torch.Tensor) -> list[int] | None:
    """Infer which 3 columns in `proprio` correspond to the cube world position (x,y,z).

    We do this by matching columns against the oracle cube position computed from the scene.
    Returns a list of 3 column indices [ix, iy, iz], or None if not confidently identified.
    """
    try:
        obj = env_unwrapped.scene["cube_2"]
        cube_pos = obj.data.root_pos_w - env_unwrapped.scene.env_origins  # (B,3)
        if not torch.is_tensor(cube_pos) or cube_pos.ndim != 2 or cube_pos.shape[1] != 3:
            return None
        if cube_pos.shape[0] != proprio.shape[0]:
            return None

        # Use a small subset for speed (matching is very sharp: exact float equality often holds).
        b = int(min(64, proprio.shape[0]))
        p = proprio[:b].float()
        c = cube_pos[:b].float()

        # Compute per-column mean absolute error vs each cube coordinate.
        # err[d, j] = mean_i |p[i, j] - c[i, d]|
        err = torch.empty((3, p.shape[1]), device=p.device, dtype=torch.float32)
        for d in range(3):
            err[d] = (p - c[:, d : d + 1]).abs().mean(dim=0)

        # Pick best column per dimension and validate with a tolerance.
        best = torch.argmin(err, dim=1)  # (3,)
        best_err = err[torch.arange(3, device=p.device), best]

        # Tolerance: positions are in meters; 1e-3 is 1mm (already strict).
        if torch.any(best_err > 1e-3):
            return None

        idxs = [int(best[0].item()), int(best[1].item()), int(best[2].item())]
        return idxs
    except Exception:
        return None


def run_evaluation_episodes_pose_v1(
    env,
    policy,
    policy_nn,
    num_episodes: int,
    device: str,
    pose_manager,
    phase_detector,
    force_dim: int,
    seed: int,
) -> dict:
    """Run evaluation episodes for pose corruption Variant 1 (phase-triggered)."""
    num_envs = env.unwrapped.num_envs

    successes = []
    episode_lengths = []
    triggered = []
    corrupted_step_rates = []

    # -------------------------------------------------------------------------
    # IMPORTANT (speed + correctness with large num_envs):
    #
    # When num_envs >> num_episodes, collecting the *first* N completed episodes is biased toward
    # "fast finishers". For faster evaluation with many envs (e.g., 1600), we instead:
    # - sample a fixed random subset of env_ids of size `num_episodes`
    # - collect exactly ONE episode from each sampled env
    # This removes completion-time bias while still benefiting from parallel simulation throughput.
    # -------------------------------------------------------------------------
    target_env_count = int(min(num_episodes, num_envs))
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed) + 1337)
    perm = torch.randperm(num_envs, generator=gen)
    target_env_ids = perm[:target_env_count].to(device=device, dtype=torch.long)
    is_target_env = torch.zeros(num_envs, dtype=torch.bool, device=device)
    is_target_env[target_env_ids] = True
    collected_for_env = torch.zeros(num_envs, dtype=torch.bool, device=device)
    episodes_completed = 0

    # Reset env at the start of EACH condition so "trigger on entering phase" is meaningful.
    # This fixes the poe4.pdf issue where conditions could begin mid-episode.
    try:
        reset_out = env.reset()
        if isinstance(reset_out, tuple) and len(reset_out) >= 1:
            obs = reset_out[0]
        else:
            obs = reset_out
    except Exception:
        # Fallback: if env.reset() isn't supported by the wrapper stack, keep old behavior.
        obs = env.get_observations()

    # Reset recurrent states defensively.
    try:
        policy_nn.reset(torch.ones((num_envs,), device=device, dtype=torch.bool))
    except Exception:
        pass

    all_env_ids = torch.arange(num_envs, device=device)
    pose_manager.reset(all_env_ids)
    env_episode_step = torch.zeros(num_envs, dtype=torch.int32, device=device)
    env_corrupted_steps = torch.zeros(num_envs, dtype=torch.int32, device=device)

    while episodes_completed < num_episodes:
        with torch.inference_mode():
            # Update phases and pose manager BEFORE action
            try:
                phases = phase_detector.detect_phases(env.unwrapped)
                pose_manager.update_phases(phases)
            except Exception:
                pass
            pose_manager.step()

            # Count timestep-level corruption (before action; corresponds to corrupted observation this step).
            env_corrupted_steps += pose_manager.event_active.to(dtype=torch.int32)

            # Lazily infer and cache cube-pos indices (once per env) for correct corruption.
            if not hasattr(env.unwrapped, "__pose_eval_cube_pos_indices"):
                inferred = None
                if hasattr(obs, "keys") and "proprio" in obs and torch.is_tensor(obs["proprio"]) and obs["proprio"].ndim == 2:
                    inferred = _infer_cube_pos_indices(env.unwrapped, obs["proprio"])
                if inferred is not None:
                    setattr(env.unwrapped, "__pose_eval_cube_pos_indices", inferred)
                    print(f"  [INFO] pose-eval: inferred cube_position indices in obs['proprio']: {inferred}", flush=True)
                else:
                    setattr(env.unwrapped, "__pose_eval_cube_pos_indices", None)
                    print("  [WARN] pose-eval: could not infer cube_position indices; falling back to legacy slice.", flush=True)

            obs_corrupt = apply_pose_corruption_to_obs(obs, pose_manager, force_dim=force_dim, env_unwrapped=env.unwrapped)
            actions = policy(obs_corrupt)
            step_out = env.step(actions)

            if isinstance(step_out, (tuple, list)) and len(step_out) == 5:
                obs, rewards, terminated, truncated, info = step_out
                done = terminated | truncated
            elif isinstance(step_out, (tuple, list)) and len(step_out) == 4:
                obs, rewards, done, info = step_out
                if not torch.is_tensor(done):
                    done = torch.as_tensor(done, device=device)
                done = done.to(dtype=torch.bool)
                truncated = torch.zeros_like(done, dtype=torch.bool)
                terminated = done
            else:
                raise ValueError("Unexpected env.step() return")

            env_episode_step += 1
            done_indices = done.nonzero(as_tuple=False).squeeze(-1)

            for idx in done_indices.cpu().tolist():
                if episodes_completed >= target_env_count:
                    break
                if not bool(is_target_env[idx].item()):
                    continue
                if bool(collected_for_env[idx].item()):
                    continue

                # success via termination_manager term (task success)
                success = False
                try:
                    if hasattr(env.unwrapped, "termination_manager"):
                        term = env.unwrapped.termination_manager.get_term("success_grasp")
                        success = bool(term[idx].item()) if torch.is_tensor(term) else bool(term)
                    else:
                        success = bool(terminated[idx].item())
                except Exception:
                    success = bool(terminated[idx].item())

                successes.append(float(success))
                episode_lengths.append(env_episode_step[idx].item())
                did_trigger = float(pose_manager.triggered_count[idx].item() > 0)
                triggered.append(did_trigger)
                # timestep-level corruption fraction for this episode
                ep_steps = max(int(env_episode_step[idx].item()), 1)
                ep_corr = int(env_corrupted_steps[idx].item())
                corrupted_step_rates.append(float(ep_corr / ep_steps))
                collected_for_env[idx] = True
                episodes_completed += 1

            if len(done_indices) > 0:
                env_episode_step[done_indices] = 0
                env_corrupted_steps[done_indices] = 0
                pose_manager.reset(done_indices)

            policy_nn.reset(done)

    return {
        "success_rate": np.mean(successes) if successes else 0.0,
        "mean_episode_length": np.mean(episode_lengths) if episode_lengths else 0.0,
        "std_episode_length": np.std(episode_lengths) if episode_lengths else 0.0,
        "num_episodes": int(episodes_completed),
        # Keep original key for backward compatibility with your printing/saving pipeline,
        # but interpret it as "timestep-level corruption rate" for poe3 evaluation.
        "dropout_triggered_rate": np.mean(corrupted_step_rates) if corrupted_step_rates else 0.0,
        "corruption_step_rate": np.mean(corrupted_step_rates) if corrupted_step_rates else 0.0,
        "corruption_episode_rate": np.mean(triggered) if triggered else 0.0,
    }


def run_evaluation_episodes_pose_v2(
    env,
    policy,
    policy_nn,
    num_episodes: int,
    device: str,
    pose_manager,
    force_dim: int,
    seed: int,
) -> dict:
    """Run evaluation episodes for pose corruption Variant 2 (time-triggered)."""
    num_envs = env.unwrapped.num_envs

    successes = []
    episode_lengths = []
    triggered = []
    corrupted_step_rates = []

    target_env_count = int(min(num_episodes, num_envs))
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed) + 7331)
    perm = torch.randperm(num_envs, generator=gen)
    target_env_ids = perm[:target_env_count].to(device=device, dtype=torch.long)
    is_target_env = torch.zeros(num_envs, dtype=torch.bool, device=device)
    is_target_env[target_env_ids] = True
    collected_for_env = torch.zeros(num_envs, dtype=torch.bool, device=device)
    episodes_completed = 0
    all_env_ids = torch.arange(num_envs, device=device)
    pose_manager.reset(all_env_ids)

    # Reset env at the start of EACH condition so onset steps are measured from episode start.
    try:
        reset_out = env.reset()
        if isinstance(reset_out, tuple) and len(reset_out) >= 1:
            obs = reset_out[0]
        else:
            obs = reset_out
    except Exception:
        obs = env.get_observations()

    try:
        policy_nn.reset(torch.ones((num_envs,), device=device, dtype=torch.bool))
    except Exception:
        pass
    env_episode_step = torch.zeros(num_envs, dtype=torch.int32, device=device)
    env_corrupted_steps = torch.zeros(num_envs, dtype=torch.int32, device=device)

    while episodes_completed < num_episodes:
        with torch.inference_mode():
            pose_manager.step()

            # Count timestep-level corruption (before action; corresponds to corrupted observation this step).
            env_corrupted_steps += pose_manager.event_active.to(dtype=torch.int32)

            # Lazily infer and cache cube-pos indices (once per env) for correct corruption.
            if not hasattr(env.unwrapped, "__pose_eval_cube_pos_indices"):
                inferred = None
                if hasattr(obs, "keys") and "proprio" in obs and torch.is_tensor(obs["proprio"]) and obs["proprio"].ndim == 2:
                    inferred = _infer_cube_pos_indices(env.unwrapped, obs["proprio"])
                if inferred is not None:
                    setattr(env.unwrapped, "__pose_eval_cube_pos_indices", inferred)
                    print(f"  [INFO] pose-eval: inferred cube_position indices in obs['proprio']: {inferred}", flush=True)
                else:
                    setattr(env.unwrapped, "__pose_eval_cube_pos_indices", None)
                    print("  [WARN] pose-eval: could not infer cube_position indices; falling back to legacy slice.", flush=True)

            obs_corrupt = apply_pose_corruption_to_obs(obs, pose_manager, force_dim=force_dim, env_unwrapped=env.unwrapped)
            actions = policy(obs_corrupt)
            step_out = env.step(actions)

            if isinstance(step_out, (tuple, list)) and len(step_out) == 5:
                obs, rewards, terminated, truncated, info = step_out
                done = terminated | truncated
            elif isinstance(step_out, (tuple, list)) and len(step_out) == 4:
                obs, rewards, done, info = step_out
                if not torch.is_tensor(done):
                    done = torch.as_tensor(done, device=device)
                done = done.to(dtype=torch.bool)
                truncated = torch.zeros_like(done, dtype=torch.bool)
                terminated = done
            else:
                raise ValueError("Unexpected env.step() return")

            env_episode_step += 1
            done_indices = done.nonzero(as_tuple=False).squeeze(-1)

            for idx in done_indices.cpu().tolist():
                if episodes_completed >= target_env_count:
                    break
                if not bool(is_target_env[idx].item()):
                    continue
                if bool(collected_for_env[idx].item()):
                    continue

                success = False
                try:
                    if hasattr(env.unwrapped, "termination_manager"):
                        term = env.unwrapped.termination_manager.get_term("success_grasp")
                        success = bool(term[idx].item()) if torch.is_tensor(term) else bool(term)
                    else:
                        success = bool(terminated[idx].item())
                except Exception:
                    success = bool(terminated[idx].item())

                successes.append(float(success))
                episode_lengths.append(env_episode_step[idx].item())
                did_trigger = float(pose_manager.triggered_count[idx].item() > 0)
                triggered.append(did_trigger)
                ep_steps = max(int(env_episode_step[idx].item()), 1)
                ep_corr = int(env_corrupted_steps[idx].item())
                corrupted_step_rates.append(float(ep_corr / ep_steps))
                collected_for_env[idx] = True
                episodes_completed += 1

            if len(done_indices) > 0:
                env_episode_step[done_indices] = 0
                env_corrupted_steps[done_indices] = 0
                pose_manager.reset(done_indices)

            policy_nn.reset(done)

    return {
        "success_rate": np.mean(successes) if successes else 0.0,
        "mean_episode_length": np.mean(episode_lengths) if episode_lengths else 0.0,
        "std_episode_length": np.std(episode_lengths) if episode_lengths else 0.0,
        "num_episodes": int(episodes_completed),
        "dropout_triggered_rate": np.mean(corrupted_step_rates) if corrupted_step_rates else 0.0,
        "corruption_step_rate": np.mean(corrupted_step_rates) if corrupted_step_rates else 0.0,
        "corruption_episode_rate": np.mean(triggered) if triggered else 0.0,
    }


def run_variant2_eval_shared(
    env,
    policy,
    policy_nn,
    policy_name: str,
    onset_steps: list,
    durations: list,
    num_episodes: int,
    device: str,
    seed: int,
) -> tuple:
    """Run Variant 2 (time-based) evaluation using shared environment."""
    from Manipulation_policy.tasks.manager_based.manipulation.stack.mdp.eval_dropout_wrapper import (
        Variant2TimeDropoutManager,
    )
    from Manipulation_policy.tasks.manager_based.manipulation.stack.mdp.eval_dropout_cfg import (
        Variant2TimeDropoutCfg,
    )
    
    results = []
    success_matrix = []
    
    print(f"\n{'='*60}")
    print(f"VARIANT 2 (Time-Based) Evaluation: {policy_name}")
    print(f"Onset steps: {onset_steps}, Durations: {durations}")
    print(f"{'='*60}")
    
    print("  Starting evaluation grid...", flush=True)
    
    for onset in onset_steps:
        onset_results = []
        
        for duration in durations:
            print(f"  Onset={onset}, Duration={duration}...", end=" ", flush=True)
            
            # Create NEW dropout manager with this config
            cfg = Variant2TimeDropoutCfg(
                onset_step=onset,
                dropout_duration_steps=duration,
            )
            dropout_manager = Variant2TimeDropoutManager(
                cfg=cfg,
                num_envs=env.unwrapped.num_envs,
                device=device,
            )
            
            # Attach to environment
            env.unwrapped.dropout_manager = dropout_manager
            
            # Run evaluation
            metrics = run_evaluation_episodes_v2(
                env, policy, policy_nn, num_episodes, device, dropout_manager
            )
            
            result = EvalResult(
                policy_name=policy_name,
                variant=2,
                condition=str(onset),
                duration=duration,
                **metrics,
            )
            results.append(result)
            onset_results.append(metrics["success_rate"])
            
            print(f"Success: {metrics['success_rate']:.1%}, Dropout triggered: {metrics['dropout_triggered_rate']:.1%}")
        
        success_matrix.append(onset_results)
    
    summary = EvalSummary(
        policy_name=policy_name,
        variant=2,
        conditions=[str(o) for o in onset_steps],
        durations=durations,
        success_matrix=success_matrix,
        timestamp=datetime.now().isoformat(),
        seed=seed,
        num_episodes_per_condition=num_episodes,
    )
    
    return summary, results


def run_pose_variant1_eval_shared(
    env,
    policy,
    policy_nn,
    policy_name: str,
    phases: list,
    durations: list,
    modes: list[str],
    num_episodes: int,
    device: str,
    seed: int,
    force_dim: int,
    pose_delay_steps: int,
    pose_noise_std: float,
    pose_drift_noise_std: float,
) -> list[tuple[EvalSummary, list]]:
    """Run pose-corruption Variant 1 evaluation for each mode (phase-based)."""
    from Manipulation_policy.tasks.manager_based.manipulation.stack.mdp.phase_detector import PickPlacePhaseDetector
    from Manipulation_policy.tasks.manager_based.manipulation.stack.mdp.pose_corruption_cfg import PoseCorruptionCfg
    from Manipulation_policy.tasks.manager_based.manipulation.stack.mdp.eval_pose_corruption_manager import (
        PhaseTriggeredPoseCorruptionManager,
    )

    phase_detector = PickPlacePhaseDetector()
    outputs: list[tuple[EvalSummary, list]] = []

    print(f"\n{'='*60}")
    print(f"POSE CORRUPTION EVAL - VARIANT 1 (Phase-Based): {policy_name}")
    print(f"Phases: {phases}, Durations: {durations}, Modes: {modes}")
    print(f"{'='*60}")

    for mode in modes:
        results = []
        success_matrix = []
        print(f"\n--- Mode: {mode} ---", flush=True)

        base_cfg = PoseCorruptionCfg(
            enabled=True,
            mode=mode,
            event_probability=0.0,
            duration_range=(1, 1),
            delay_steps=int(pose_delay_steps),
            noise_std=float(pose_noise_std),
            drift_noise_std=float(pose_drift_noise_std),
        )

        for phase in phases:
            phase_results = []
            for duration in durations:
                print(f"  Phase={phase}, Duration={duration}...", end=" ", flush=True)
                pose_manager = PhaseTriggeredPoseCorruptionManager(
                    base_cfg,
                    target_phase=phase,
                    duration_steps=int(duration),
                    trigger_once_per_episode=True,
                    # poe3 intent: trigger immediately upon entering the phase (do NOT miss short/noisy phases)
                    require_stable_phase=False,
                    stable_phase_steps=0,
                    phase_entry_delay=0,
                    num_envs=env.unwrapped.num_envs,
                    device=device,
                )
                env.unwrapped.pose_corruption_manager = pose_manager

                metrics = run_evaluation_episodes_pose_v1(
                    env, policy, policy_nn, num_episodes, device,
                    pose_manager, phase_detector, force_dim=force_dim, seed=seed
                )

                result = EvalResult(
                    policy_name=policy_name,
                    variant=1,
                    condition=phase,
                    duration=int(duration),
                    **metrics,
                )
                results.append(result)
                phase_results.append(metrics["success_rate"])
                # Pose corrupted is timestep-level rate (matches duration sweep intuition); also show episode trigger rate.
                print(
                    f"Success: {metrics['success_rate']:.1%}, "
                    f"Pose corrupted (steps): {metrics['corruption_step_rate']:.1%}, "
                    f"Pose triggered (episodes): {metrics['corruption_episode_rate']:.1%}"
                )

            success_matrix.append(phase_results)

        summary = EvalSummary(
            policy_name=f"{policy_name}_pose_{mode}",
            variant=1,
            conditions=phases,
            durations=durations,
            success_matrix=success_matrix,
            timestamp=datetime.now().isoformat(),
            seed=seed,
            num_episodes_per_condition=num_episodes,
        )
        outputs.append((summary, results))

    return outputs


def run_pose_variant2_eval_shared(
    env,
    policy,
    policy_nn,
    policy_name: str,
    onset_steps: list,
    durations: list,
    modes: list[str],
    num_episodes: int,
    device: str,
    seed: int,
    force_dim: int,
    pose_delay_steps: int,
    pose_noise_std: float,
    pose_drift_noise_std: float,
) -> list[tuple[EvalSummary, list]]:
    """Run pose-corruption Variant 2 evaluation for each mode (time-based)."""
    from Manipulation_policy.tasks.manager_based.manipulation.stack.mdp.pose_corruption_cfg import PoseCorruptionCfg
    from Manipulation_policy.tasks.manager_based.manipulation.stack.mdp.eval_pose_corruption_manager import (
        TimeTriggeredPoseCorruptionManager,
    )

    outputs: list[tuple[EvalSummary, list]] = []

    print(f"\n{'='*60}")
    print(f"POSE CORRUPTION EVAL - VARIANT 2 (Time-Based): {policy_name}")
    print(f"Onset steps: {onset_steps}, Durations: {durations}, Modes: {modes}")
    print(f"{'='*60}")

    for mode in modes:
        results = []
        success_matrix = []
        print(f"\n--- Mode: {mode} ---", flush=True)

        base_cfg = PoseCorruptionCfg(
            enabled=True,
            mode=mode,
            event_probability=0.0,
            duration_range=(1, 1),
            delay_steps=int(pose_delay_steps),
            noise_std=float(pose_noise_std),
            drift_noise_std=float(pose_drift_noise_std),
        )

        for onset in onset_steps:
            onset_results = []
            for duration in durations:
                print(f"  Onset={onset}, Duration={duration}...", end=" ", flush=True)
                pose_manager = TimeTriggeredPoseCorruptionManager(
                    base_cfg,
                    onset_step=int(onset),
                    duration_steps=int(duration),
                    num_envs=env.unwrapped.num_envs,
                    device=device,
                )
                env.unwrapped.pose_corruption_manager = pose_manager

                metrics = run_evaluation_episodes_pose_v2(
                    env, policy, policy_nn, num_episodes, device,
                    pose_manager, force_dim=force_dim, seed=seed
                )

                result = EvalResult(
                    policy_name=policy_name,
                    variant=2,
                    condition=str(onset),
                    duration=int(duration),
                    **metrics,
                )
                results.append(result)
                onset_results.append(metrics["success_rate"])
                print(
                    f"Success: {metrics['success_rate']:.1%}, "
                    f"Pose corrupted (steps): {metrics['corruption_step_rate']:.1%}, "
                    f"Pose triggered (episodes): {metrics['corruption_episode_rate']:.1%}"
                )

            success_matrix.append(onset_results)

        summary = EvalSummary(
            policy_name=f"{policy_name}_pose_{mode}",
            variant=2,
            conditions=[str(o) for o in onset_steps],
            durations=durations,
            success_matrix=success_matrix,
            timestamp=datetime.now().isoformat(),
            seed=seed,
            num_episodes_per_condition=num_episodes,
        )
        outputs.append((summary, results))

    return outputs


def run_evaluation_episodes_v2(
    env,
    policy,
    policy_nn,
    num_episodes: int,
    device: str,
    dropout_manager,
) -> dict:
    """Run evaluation episodes for Variant 2 (time-based dropout)."""
    num_envs = env.unwrapped.num_envs
    
    successes = []
    episode_lengths = []
    dropout_triggered = []
    
    episodes_completed = 0
    
    # Reset dropout manager for this condition
    all_env_ids = torch.arange(num_envs, device=device)
    dropout_manager.reset(all_env_ids)
    
    # Get current observations (don't reset env - causes inference mode issues)
    obs = env.get_observations()
    
    env_episode_step = torch.zeros(num_envs, dtype=torch.int32, device=device)
    
    while episodes_completed < num_episodes:
        with torch.inference_mode():
            # Update dropout state BEFORE getting actions (time-based)
            dropout_manager.step()
            
            # Apply dropout to observations
            obs_with_dropout = apply_dropout_to_obs(obs, dropout_manager, device)
            
            actions = policy(obs_with_dropout)
            step_out = env.step(actions)
            
            if isinstance(step_out, (tuple, list)) and len(step_out) == 5:
                obs, rewards, terminated, truncated, info = step_out
                done = terminated | truncated
            elif isinstance(step_out, (tuple, list)) and len(step_out) == 4:
                obs, rewards, done, info = step_out
                if not torch.is_tensor(done):
                    done = torch.as_tensor(done, device=device)
                done = done.to(dtype=torch.bool)
                truncated = torch.zeros_like(done, dtype=torch.bool)
                terminated = done
            else:
                raise ValueError(f"Unexpected env.step() return")
            
            env_episode_step += 1
            
            done_indices = done.nonzero(as_tuple=False).squeeze(-1)
            
            for idx in done_indices.cpu().tolist():
                if episodes_completed >= num_episodes:
                    break
                
                # See `run_evaluation_episodes` for rationale: only count true task success.
                success = False
                try:
                    if hasattr(env.unwrapped, "termination_manager"):
                        term = env.unwrapped.termination_manager.get_term("success_grasp")
                        if torch.is_tensor(term):
                            success = bool(term[idx].item())
                        else:
                            success = bool(term)
                    else:
                        success = bool(terminated[idx].item())
                except Exception:
                    success = bool(terminated[idx].item())
                successes.append(float(success))
                episode_lengths.append(env_episode_step[idx].item())
                
                triggered = dropout_manager.dropout_triggered_count[idx].item() > 0
                dropout_triggered.append(float(triggered))
                
                episodes_completed += 1
            
            if len(done_indices) > 0:
                env_episode_step[done_indices] = 0
                dropout_manager.reset(done_indices)
            
            policy_nn.reset(done)
    
    return {
        "success_rate": np.mean(successes) if successes else 0.0,
        "mean_episode_length": np.mean(episode_lengths) if episode_lengths else 0.0,
        "std_episode_length": np.std(episode_lengths) if episode_lengths else 0.0,
        "num_episodes": len(successes),
        "dropout_triggered_rate": np.mean(dropout_triggered) if dropout_triggered else 0.0,
    }


def save_eval_results(output_dir: str, summary: EvalSummary, results: list):
    """Save evaluation results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    variant_str = f"variant{summary.variant}"
    filename_base = f"{summary.policy_name}_{variant_str}"
    
    # Save summary JSON
    summary_path = os.path.join(output_dir, f"{filename_base}_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(asdict(summary), f, indent=2)
    print(f"  Saved: {summary_path}")
    
    # Save results JSON
    results_path = os.path.join(output_dir, f"{filename_base}_results.json")
    with open(results_path, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    
    # Save CSV matrix
    import csv
    csv_path = os.path.join(output_dir, f"{filename_base}_matrix.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ["condition"] + [str(d) for d in summary.durations]
        writer.writerow(header)
        for i, cond in enumerate(summary.conditions):
            row = [cond] + [f"{v:.4f}" for v in summary.success_matrix[i]]
            writer.writerow(row)
    print(f"  Saved: {csv_path}")


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

    # Filter out non-files just in case and de-duplicate (patterns overlap).
    candidates = sorted({c for c in candidates if os.path.isfile(c)})

    # NOTE: `retrieve_file_path()` may copy files into `/tmp/...` and change mtimes unpredictably.
    # Prefer the numerically-largest training step encoded in filenames like `model_1000.pt`.
    step_candidates: list[tuple[int, str]] = []
    for c in candidates:
        name = os.path.basename(c)
        m = re.match(r"^model[_-]?(\d+)\.pt$", name)
        if m:
            step_candidates.append((int(m.group(1)), c))

    if step_candidates:
        # Max step wins; tie-break by mtime.
        step_candidates.sort(key=lambda t: (t[0], os.path.getmtime(t[1])), reverse=True)
        chosen = step_candidates[0][1]
        print(f"[INFO] Resolved checkpoint directory '{p}' -> '{chosen}'")
        return chosen

    # Fallback: newest by mtime for non-standard filenames.
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

    # =========================================================================
    # EVALUATION MODE: Run systematic dropout evaluation and exit
    # =========================================================================
    if args_cli.eval_pose_corruption:
        print("\n" + "=" * 70)
        print("POSE CORRUPTION ROBUSTNESS EVALUATION MODE (poe3.pdf)")
        print("=" * 70)

        phases = [p.strip() for p in args_cli.pose_eval_phases.split(",")]
        onset_steps = [int(s.strip()) for s in args_cli.pose_eval_onset_steps.split(",")]
        durations = [int(d.strip()) for d in args_cli.pose_eval_durations.split(",")]
        modes = [m.strip() for m in args_cli.pose_eval_modes.split(",") if m.strip()]

        policy_name = args_cli.policy_name
        if policy_name is None:
            policy_name = os.path.basename(os.path.dirname(resume_path))

        print(f"Policy: {policy_name}")
        print(f"Checkpoint: {resume_path}")
        print(f"Variant: {args_cli.pose_eval_variant}")
        print(f"Modes: {modes}")
        print(f"Episodes per condition: {args_cli.pose_eval_episodes}")
        print(f"Output: {args_cli.pose_eval_output_dir}")
        print("=" * 70)

        device = agent_cfg.device
        all_summaries = []

        # Create environment ONCE
        #
        # NOTE: We allow large `num_envs` for speed. To avoid evaluation bias when `num_envs` is much larger
        # than `pose_eval_episodes`, the episode collector samples a fixed random subset of env_ids and collects
        # one episode per sampled env (unbiased wrt completion time).
        requested_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        env_cfg.scene.num_envs = int(requested_envs)
        if int(requested_envs) > int(args_cli.pose_eval_episodes):
            print(
                f"[INFO] pose-eval: num_envs={requested_envs} > pose_eval_episodes={args_cli.pose_eval_episodes}; "
                f"using random env-subset sampling (one episode per sampled env) for unbiased fast eval."
            )

        print("\nCreating environment (one-time for pose eval)...", flush=True)
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)

        # Add force sensor wrapper if needed
        if args_cli.force_sensing:
            try:
                from Manipulation_policy.tasks.manager_based.manipulation.stack.mdp.force_sensor_env_wrapper import (
                    VecEnvForceSensorWrapper,
                    ForceSensorConfig,
                )
                force_cfg = ForceSensorConfig(
                    enabled=True,
                    force_obs_mode=args_cli.force_mode,
                    normalize=True,
                    effort_limit=70.0,
                    robot_name="robot",
                    object_name="cube_2",
                    ee_frame_name="ee_frame",
                )
                env = VecEnvForceSensorWrapper(env, force_cfg)
                print(f"  Force sensing enabled: mode={args_cli.force_mode}")
            except Exception as e:
                print(f"[ERROR] Failed to add force sensor wrapper: {e}")
                raise

        force_dim = _infer_force_dim(args_cli.force_mode) if args_cli.force_sensing else 0

        # Wrap for RSL-RL
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        # Load policy ONCE
        print("Loading policy (one-time for pose eval)...", flush=True)
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        runner.load(resume_path)

        policy = runner.get_inference_policy(device=device)
        try:
            policy_nn = runner.alg.policy
        except AttributeError:
            policy_nn = runner.alg.actor_critic

        # Run Variant 1
        if args_cli.pose_eval_variant in ["1", "both"]:
            outputs = run_pose_variant1_eval_shared(
                env=env,
                policy=policy,
                policy_nn=policy_nn,
                policy_name=policy_name,
                phases=phases,
                durations=durations,
                modes=modes,
                num_episodes=args_cli.pose_eval_episodes,
                device=device,
                seed=agent_cfg.seed,
                force_dim=force_dim,
                pose_delay_steps=args_cli.pose_eval_delay_steps,
                pose_noise_std=args_cli.pose_eval_noise_std,
                pose_drift_noise_std=args_cli.pose_eval_drift_noise_std,
            )
            for summary, results in outputs:
                save_eval_results(args_cli.pose_eval_output_dir, summary, results)
                all_summaries.append(summary)

        # Run Variant 2
        if args_cli.pose_eval_variant in ["2", "both"]:
            outputs = run_pose_variant2_eval_shared(
                env=env,
                policy=policy,
                policy_nn=policy_nn,
                policy_name=policy_name,
                onset_steps=onset_steps,
                durations=durations,
                modes=modes,
                num_episodes=args_cli.pose_eval_episodes,
                device=device,
                seed=agent_cfg.seed,
                force_dim=force_dim,
                pose_delay_steps=args_cli.pose_eval_delay_steps,
                pose_noise_std=args_cli.pose_eval_noise_std,
                pose_drift_noise_std=args_cli.pose_eval_drift_noise_std,
            )
            for summary, results in outputs:
                save_eval_results(args_cli.pose_eval_output_dir, summary, results)
                all_summaries.append(summary)

        env.close()

        combined_path = os.path.join(args_cli.pose_eval_output_dir, f"{policy_name}_pose_all_summaries.json")
        os.makedirs(args_cli.pose_eval_output_dir, exist_ok=True)
        with open(combined_path, "w") as f:
            json.dump([asdict(s) for s in all_summaries], f, indent=2)

        print("\n" + "=" * 70)
        print("POSE EVALUATION COMPLETE")
        print(f"Results saved to: {args_cli.pose_eval_output_dir}")
        print("=" * 70)
        return

    if args_cli.eval_dropout:
        print("\n" + "=" * 70)
        print("DROPOUT ROBUSTNESS EVALUATION MODE")
        print("=" * 70)
        
        # Parse evaluation parameters
        phases = [p.strip() for p in args_cli.eval_phases.split(",")]
        onset_steps = [int(s.strip()) for s in args_cli.eval_onset_steps.split(",")]
        durations = [int(d.strip()) for d in args_cli.eval_durations.split(",")]
        
        # Auto-detect policy name from checkpoint path
        policy_name = args_cli.policy_name
        if policy_name is None:
            policy_name = os.path.basename(os.path.dirname(resume_path))
        
        print(f"Policy: {policy_name}")
        print(f"Checkpoint: {resume_path}")
        print(f"Variant: {args_cli.eval_variant}")
        print(f"Episodes per condition: {args_cli.eval_episodes}")
        print(f"Output: {args_cli.eval_output_dir}")
        print("=" * 70)
        
        device = agent_cfg.device
        all_summaries = []
        
        # Create environment ONCE for all variants
        print("\nCreating environment (one-time for all variants)...", flush=True)
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
        
        # Add force sensor wrapper if needed (for M2/M4 policies)
        if args_cli.force_sensing:
            try:
                from Manipulation_policy.tasks.manager_based.manipulation.stack.mdp.force_sensor_env_wrapper import (
                    VecEnvForceSensorWrapper,
                    ForceSensorConfig,
                )
                force_cfg = ForceSensorConfig(
                    enabled=True,
                    force_obs_mode=args_cli.force_mode,
                    normalize=True,
                    effort_limit=70.0,
                    robot_name="robot",
                    object_name="cube_2",
                    ee_frame_name="ee_frame",
                )
                env = VecEnvForceSensorWrapper(env, force_cfg)
                print(f"  Force sensing enabled: mode={args_cli.force_mode}")
            except Exception as e:
                print(f"[ERROR] Failed to add force sensor wrapper: {e}")
                raise
        
        # Wrap for RSL-RL
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        
        # Load policy ONCE
        print("Loading policy (one-time for all variants)...", flush=True)
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        runner.load(resume_path)
        
        policy = runner.get_inference_policy(device=device)
        try:
            policy_nn = runner.alg.policy
        except AttributeError:
            policy_nn = runner.alg.actor_critic
        
        # Run Variant 1 (Phase-based)
        if args_cli.eval_variant in ["1", "both"]:
            summary, results = run_variant1_eval_shared(
                env=env,
                policy=policy,
                policy_nn=policy_nn,
                policy_name=policy_name,
                phases=phases,
                durations=durations,
                num_episodes=args_cli.eval_episodes,
                device=device,
                seed=agent_cfg.seed,
            )
            save_eval_results(args_cli.eval_output_dir, summary, results)
            all_summaries.append(summary)
        
        # Run Variant 2 (Time-based)
        if args_cli.eval_variant in ["2", "both"]:
            summary, results = run_variant2_eval_shared(
                env=env,
                policy=policy,
                policy_nn=policy_nn,
                policy_name=policy_name,
                onset_steps=onset_steps,
                durations=durations,
                num_episodes=args_cli.eval_episodes,
                device=device,
                seed=agent_cfg.seed,
            )
            save_eval_results(args_cli.eval_output_dir, summary, results)
            all_summaries.append(summary)
        
        # Close environment
        env.close()
        
        # Save combined summary
        combined_path = os.path.join(args_cli.eval_output_dir, f"{policy_name}_all_summaries.json")
        with open(combined_path, 'w') as f:
            json.dump([asdict(s) for s in all_summaries], f, indent=2)
        
        print("\n" + "=" * 70)
        print("EVALUATION COMPLETE")
        print(f"Results saved to: {args_cli.eval_output_dir}")
        print("=" * 70)
        
        return  # Exit after evaluation
    
    # =========================================================================
    # NORMAL PLAY MODE
    # =========================================================================
    
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

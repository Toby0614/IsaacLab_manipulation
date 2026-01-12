# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Systematic dropout evaluation script for M1-M4 policies.

This script evaluates trained policies under:
- Variant 1: Phase-based dropout (dropout at specific manipulation phases)
- Variant 2: Time-based dropout (dropout at specific onset times)

Results are saved as JSON/CSV for 2D sensitivity map visualization.

Usage:
    # Evaluate single policy with Variant 1 (phase-based)
    python scripts/rsl_rl/evaluate_dropout.py \
        --task Isaac-Franka-PickPlace-v0 \
        --checkpoint logs/rsl_rl/franka_pickplace/M1_policy/model_1000.pt \
        --variant 1 \
        --output_dir results/dropout_eval

    # Evaluate all M1-M4 policies
    python scripts/rsl_rl/evaluate_dropout.py \
        --task Isaac-Franka-PickPlace-v0 \
        --eval_all_policies \
        --variant both \
        --output_dir results/dropout_eval
"""

import argparse
import sys
import os
import re

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate policy robustness under modality dropout.")

# Environment and policy
parser.add_argument("--task", type=str, default="Isaac-Franka-PickPlace-v0", help="Task name")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to policy checkpoint")
parser.add_argument("--eval_all_policies", action="store_true", help="Evaluate all M1-M4 policies")
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments")
parser.add_argument("--seed", type=int, default=42, help="Random seed")

# Dropout evaluation parameters
parser.add_argument("--variant", type=str, default="both", choices=["1", "2", "both"],
                    help="Evaluation variant: 1 (phase), 2 (time), or both")
parser.add_argument("--num_episodes", type=int, default=100, help="Episodes per condition")
parser.add_argument("--episode_length", type=int, default=250, help="Max steps per episode")

# Grid parameters (can be overridden)
parser.add_argument("--phases", type=str, default="reach,grasp,lift,transport,place",
                    help="Comma-separated phases for Variant 1")
parser.add_argument("--onset_steps", type=str, default="10,25,50,75,100,125,150",
                    help="Comma-separated onset steps for Variant 2")
parser.add_argument("--durations", type=str, default="5,10,20,40,60,80,100",
                    help="Comma-separated dropout durations (steps)")

# Output
parser.add_argument("--output_dir", type=str, default="results/dropout_eval", help="Output directory")
parser.add_argument("--policy_name", type=str, default=None, help="Policy name for output files")

# Force sensor (for M2/M4 policies)
parser.add_argument("--use_force_sensor", action="store_true", help="Enable force sensor wrapper")

# Misc
parser.add_argument("--disable_fabric", action="store_true", help="Disable fabric")

# Add IsaacLab app launcher args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Script defaults (leave `--headless` / `--enable_cameras` ownership to AppLauncher)
args_cli.enable_cameras = True

# Launch app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest of imports after app launch."""

import json
import glob
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import torch
import numpy as np
import gymnasium as gym

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import Manipulation_policy.tasks  # noqa: F401


@dataclass
class EvalResult:
    """Single evaluation result."""
    policy_name: str
    variant: int  # 1 or 2
    condition: str  # phase name or onset step
    duration: int
    success_rate: float
    grasp_rate: float
    place_rate: float
    mean_episode_length: float
    std_episode_length: float
    num_episodes: int
    dropout_triggered_rate: float


@dataclass
class EvalSummary:
    """Summary of evaluation run."""
    policy_name: str
    variant: int
    conditions: list
    durations: list
    success_matrix: list  # 2D matrix [condition x duration]
    timestamp: str
    seed: int
    num_episodes_per_condition: int


def resolve_checkpoint(checkpoint_arg: str) -> str:
    """Resolve checkpoint path to a .pt file."""
    if checkpoint_arg is None:
        raise ValueError("Checkpoint path required")
    
    p = retrieve_file_path(checkpoint_arg)
    if os.path.isfile(p):
        return p
    
    if not os.path.isdir(p):
        raise FileNotFoundError(f"Checkpoint path does not exist: {p}")
    
    # Find .pt files in directory
    candidates = glob.glob(os.path.join(p, "*.pt"))
    candidates = sorted({c for c in candidates if os.path.isfile(c)})

    # Prefer numerically-largest training step in `model_*.pt` names (most reliable across `/tmp` copies).
    step_candidates: list[tuple[int, str]] = []
    for c in candidates:
        name = os.path.basename(c)
        m = re.match(r"^model[_-]?(\d+)\.pt$", name)
        if m:
            step_candidates.append((int(m.group(1)), c))

    if step_candidates:
        step_candidates.sort(key=lambda t: (t[0], os.path.getmtime(t[1])), reverse=True)
        return step_candidates[0][1]

    # Fallback: newest by mtime for non-standard filenames.
    candidates.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    if not candidates:
        raise FileNotFoundError(f"No .pt files found in {p}")
    
    return candidates[0]


def get_policy_paths() -> dict:
    """Get paths to M1-M4 policies."""
    base_dir = "logs/rsl_rl/franka_pickplace"
    
    policies = {}
    for name in ["M1_policy", "M2_policy", "M3_mix", "M4_mix"]:
        policy_dir = os.path.join(base_dir, name)
        if os.path.isdir(policy_dir):
            try:
                checkpoint = resolve_checkpoint(policy_dir)
                policies[name] = checkpoint
            except FileNotFoundError:
                print(f"[WARN] No checkpoint found for {name}")
    
    return policies


def create_env(task: str, num_envs: int, seed: int):
    """Create environment."""
    env = gym.make(task, num_envs=num_envs, seed=seed)
    return env


def load_policy(env, checkpoint_path: str, device: str = "cuda"):
    """Load trained policy from checkpoint."""
    from isaaclab_tasks.utils.hydra import hydra_task_config
    from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg
    
    # Create minimal runner config
    class MinimalRunnerCfg:
        seed = 42
        # Avoid capturing the function arg at class-definition time (static analyzers complain).
        device = "cuda"
        num_steps_per_env = 24
        max_iterations = 1
        empirical_normalization = True
        save_interval = 100
        experiment_name = "eval"
        run_name = ""
        logger = "tensorboard"
        resume = False
        load_run = ".*"
        load_checkpoint = "model_.*.pt"
        clip_actions = 1.0
        
        class policy:
            class_name = "ActorCritic"
            init_noise_std = 1.0
            actor_hidden_dims = [256, 256, 256]
            critic_hidden_dims = [256, 256, 256]
            activation = "elu"
        
        class algorithm:
            class_name = "PPO"
            value_loss_coef = 1.0
            use_clipped_value_loss = True
            clip_param = 0.2
            entropy_coef = 0.01
            num_learning_epochs = 5
            num_mini_batches = 4
            learning_rate = 1e-3
            schedule = "adaptive"
            gamma = 0.99
            lam = 0.95
            desired_kl = 0.01
            max_grad_norm = 1.0
        
        def to_dict(self):
            return {
                "seed": self.seed,
                "device": self.device,
                "num_steps_per_env": self.num_steps_per_env,
                "max_iterations": self.max_iterations,
                "empirical_normalization": self.empirical_normalization,
                "policy": {
                    "class_name": self.policy.class_name,
                    "init_noise_std": self.policy.init_noise_std,
                    "actor_hidden_dims": self.policy.actor_hidden_dims,
                    "critic_hidden_dims": self.policy.critic_hidden_dims,
                    "activation": self.policy.activation,
                },
                "algorithm": {
                    "class_name": self.algorithm.class_name,
                    "value_loss_coef": self.algorithm.value_loss_coef,
                    "use_clipped_value_loss": self.algorithm.use_clipped_value_loss,
                    "clip_param": self.algorithm.clip_param,
                    "entropy_coef": self.algorithm.entropy_coef,
                    "num_learning_epochs": self.algorithm.num_learning_epochs,
                    "num_mini_batches": self.algorithm.num_mini_batches,
                    "learning_rate": self.algorithm.learning_rate,
                    "schedule": self.algorithm.schedule,
                    "gamma": self.algorithm.gamma,
                    "lam": self.algorithm.lam,
                    "desired_kl": self.algorithm.desired_kl,
                    "max_grad_norm": self.algorithm.max_grad_norm,
                }
            }
    
    cfg = MinimalRunnerCfg()
    
    # Load from checkpoint's params if available
    params_dir = os.path.join(os.path.dirname(checkpoint_path), "params")
    agent_yaml = os.path.join(params_dir, "agent.yaml")
    
    if os.path.exists(agent_yaml):
        import yaml
        with open(agent_yaml, 'r') as f:
            agent_params = yaml.safe_load(f)
        # Could update cfg from agent_params if needed
    
    runner = OnPolicyRunner(env, cfg.to_dict(), log_dir=None, device=device)
    runner.load(checkpoint_path)
    
    policy = runner.get_inference_policy(device=device)
    policy_nn = runner.alg.policy
    
    return policy, policy_nn


def evaluate_condition(
    env,
    policy,
    policy_nn,
    num_episodes: int,
    episode_length: int,
    device: str = "cuda",
) -> dict:
    """Evaluate policy under current dropout wrapper configuration.
    
    Returns dict with success metrics.
    """
    num_envs = env.unwrapped.num_envs
    
    # Metrics
    successes = []
    grasps = []
    places = []
    episode_lengths = []
    dropout_triggered = []
    
    episodes_completed = 0
    obs = env.get_observations()
    
    # Per-env tracking
    env_episode_step = torch.zeros(num_envs, dtype=torch.int32, device=device)
    env_grasped = torch.zeros(num_envs, dtype=torch.bool, device=device)
    env_placed = torch.zeros(num_envs, dtype=torch.bool, device=device)
    
    while episodes_completed < num_episodes:
        with torch.inference_mode():
            actions = policy(obs)
            obs, rewards, terminated, truncated, info = env.step(actions)
            
            env_episode_step += 1
            
            # Track intermediate success (grasp, place) from info if available
            if "is_grasped" in info:
                env_grasped |= info["is_grasped"]
            if "is_placed" in info:
                env_placed |= info["is_placed"]
            
            # Handle episode terminations
            done = terminated | truncated
            done_indices = done.nonzero(as_tuple=False).squeeze(-1)
            
            for idx in done_indices.cpu().tolist():
                if episodes_completed >= num_episodes:
                    break
                
                # Determine success
                success = False
                # Prefer explicit success signals if the env provides them.
                if isinstance(info, dict) and "success" in info:
                    if isinstance(info["success"], torch.Tensor):
                        success = bool(info["success"][idx].item())
                    else:
                        success = bool(info["success"])
                else:
                    # IsaacLab: terminated can be success OR failure. For this task, success is
                    # the `success_grasp` termination term in the termination manager.
                    try:
                        if hasattr(env.unwrapped, "termination_manager"):
                            term = env.unwrapped.termination_manager.get_term("success_grasp")
                            if torch.is_tensor(term):
                                success = bool(term[idx].item())
                            else:
                                success = bool(term)
                        else:
                            success = False
                    except Exception:
                        success = False
                
                successes.append(float(success))
                grasps.append(float(env_grasped[idx].item()) if env_grasped[idx].item() else float(success))
                places.append(float(success))
                episode_lengths.append(env_episode_step[idx].item())
                
                # Check if dropout was triggered
                if hasattr(env, 'dropout_manager'):
                    triggered = env.dropout_manager.dropout_triggered_count[idx].item() > 0
                else:
                    triggered = False
                dropout_triggered.append(float(triggered))
                
                episodes_completed += 1
            
            # Reset tracking for done envs
            if len(done_indices) > 0:
                env_episode_step[done_indices] = 0
                env_grasped[done_indices] = False
                env_placed[done_indices] = False
            
            # Reset policy RNN states
            policy_nn.reset(done)
    
    return {
        "success_rate": np.mean(successes),
        "grasp_rate": np.mean(grasps),
        "place_rate": np.mean(places),
        "mean_episode_length": np.mean(episode_lengths),
        "std_episode_length": np.std(episode_lengths),
        "num_episodes": len(successes),
        "dropout_triggered_rate": np.mean(dropout_triggered),
    }


def run_variant1_evaluation(
    task: str,
    checkpoint_path: str,
    policy_name: str,
    phases: list,
    durations: list,
    num_envs: int,
    num_episodes: int,
    episode_length: int,
    seed: int,
    use_force_sensor: bool = False,
    device: str = "cuda",
) -> EvalSummary:
    """Run Variant 1 (phase-based) evaluation grid."""
    from Manipulation_policy.tasks.manager_based.manipulation.stack.mdp.eval_dropout_wrapper import (
        VecEnvVariant1EvalWrapper,
    )
    from Manipulation_policy.tasks.manager_based.manipulation.stack.mdp.eval_dropout_cfg import (
        Variant1PhaseDropoutCfg,
    )
    
    results = []
    success_matrix = []
    
    print(f"\n{'='*60}")
    print(f"Variant 1 (Phase-Based) Evaluation: {policy_name}")
    print(f"Phases: {phases}")
    print(f"Durations: {durations}")
    print(f"{'='*60}\n")
    
    for phase in phases:
        phase_results = []
        
        for duration in durations:
            print(f"  Evaluating phase={phase}, duration={duration}...")
            
            # Create fresh environment
            env = gym.make(task, num_envs=num_envs, seed=seed)
            
            # Apply force sensor if needed
            if use_force_sensor:
                from Manipulation_policy.tasks.manager_based.manipulation.stack.mdp.force_sensor_env_wrapper import (
                    VecEnvForceSensorWrapper,
                )
                env = VecEnvForceSensorWrapper(env)
            
            # Apply dropout wrapper
            cfg = Variant1PhaseDropoutCfg(
                target_phase=phase,
                dropout_duration_steps=duration,
            )
            env = VecEnvVariant1EvalWrapper(env, cfg)
            
            # Wrap for RSL-RL
            env = RslRlVecEnvWrapper(env)
            
            # Load policy
            policy, policy_nn = load_policy(env, checkpoint_path, device)
            
            # Evaluate
            metrics = evaluate_condition(
                env, policy, policy_nn,
                num_episodes=num_episodes,
                episode_length=episode_length,
                device=device,
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
            
            print(f"    -> Success: {metrics['success_rate']:.2%}, "
                  f"Dropout triggered: {metrics['dropout_triggered_rate']:.2%}")
            
            env.close()
        
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


def run_variant2_evaluation(
    task: str,
    checkpoint_path: str,
    policy_name: str,
    onset_steps: list,
    durations: list,
    num_envs: int,
    num_episodes: int,
    episode_length: int,
    seed: int,
    use_force_sensor: bool = False,
    device: str = "cuda",
) -> EvalSummary:
    """Run Variant 2 (time-based) evaluation grid."""
    from Manipulation_policy.tasks.manager_based.manipulation.stack.mdp.eval_dropout_wrapper import (
        VecEnvVariant2EvalWrapper,
    )
    from Manipulation_policy.tasks.manager_based.manipulation.stack.mdp.eval_dropout_cfg import (
        Variant2TimeDropoutCfg,
    )
    
    results = []
    success_matrix = []
    
    print(f"\n{'='*60}")
    print(f"Variant 2 (Time-Based) Evaluation: {policy_name}")
    print(f"Onset steps: {onset_steps}")
    print(f"Durations: {durations}")
    print(f"{'='*60}\n")
    
    for onset in onset_steps:
        onset_results = []
        
        for duration in durations:
            print(f"  Evaluating onset={onset}, duration={duration}...")
            
            # Create fresh environment
            env = gym.make(task, num_envs=num_envs, seed=seed)
            
            # Apply force sensor if needed
            if use_force_sensor:
                from Manipulation_policy.tasks.manager_based.manipulation.stack.mdp.force_sensor_env_wrapper import (
                    VecEnvForceSensorWrapper,
                )
                env = VecEnvForceSensorWrapper(env)
            
            # Apply dropout wrapper
            cfg = Variant2TimeDropoutCfg(
                onset_step=onset,
                dropout_duration_steps=duration,
            )
            env = VecEnvVariant2EvalWrapper(env, cfg)
            
            # Wrap for RSL-RL
            env = RslRlVecEnvWrapper(env)
            
            # Load policy
            policy, policy_nn = load_policy(env, checkpoint_path, device)
            
            # Evaluate
            metrics = evaluate_condition(
                env, policy, policy_nn,
                num_episodes=num_episodes,
                episode_length=episode_length,
                device=device,
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
            
            print(f"    -> Success: {metrics['success_rate']:.2%}, "
                  f"Dropout triggered: {metrics['dropout_triggered_rate']:.2%}")
            
            env.close()
        
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


def save_results(
    output_dir: str,
    summary: EvalSummary,
    results: list,
    prefix: str = "",
):
    """Save evaluation results to JSON and CSV."""
    os.makedirs(output_dir, exist_ok=True)
    
    variant_str = f"variant{summary.variant}"
    filename_base = f"{prefix}{summary.policy_name}_{variant_str}"
    
    # Save summary as JSON
    summary_path = os.path.join(output_dir, f"{filename_base}_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(asdict(summary), f, indent=2)
    print(f"Saved summary: {summary_path}")
    
    # Save detailed results as JSON
    results_path = os.path.join(output_dir, f"{filename_base}_results.json")
    with open(results_path, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"Saved results: {results_path}")
    
    # Save success matrix as CSV for easy plotting
    import csv
    csv_path = os.path.join(output_dir, f"{filename_base}_matrix.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header: condition/duration, d1, d2, ...
        header = ["condition"] + [str(d) for d in summary.durations]
        writer.writerow(header)
        for i, cond in enumerate(summary.conditions):
            row = [cond] + [f"{v:.4f}" for v in summary.success_matrix[i]]
            writer.writerow(row)
    print(f"Saved matrix: {csv_path}")


def main():
    """Main evaluation function."""
    # Parse grid parameters
    phases = [p.strip() for p in args_cli.phases.split(",")]
    onset_steps = [int(s.strip()) for s in args_cli.onset_steps.split(",")]
    durations = [int(d.strip()) for d in args_cli.durations.split(",")]
    
    # Get policies to evaluate
    if args_cli.eval_all_policies:
        policies = get_policy_paths()
        if not policies:
            print("[ERROR] No M1-M4 policies found. Train policies first.")
            return
    elif args_cli.checkpoint:
        checkpoint = resolve_checkpoint(args_cli.checkpoint)
        policy_name = args_cli.policy_name or os.path.basename(os.path.dirname(checkpoint))
        policies = {policy_name: checkpoint}
    else:
        print("[ERROR] Provide --checkpoint or --eval_all_policies")
        return
    
    print(f"\n{'='*60}")
    print("DROPOUT ROBUSTNESS EVALUATION")
    print(f"{'='*60}")
    print(f"Task: {args_cli.task}")
    print(f"Policies: {list(policies.keys())}")
    print(f"Variant: {args_cli.variant}")
    print(f"Phases: {phases}")
    print(f"Onset steps: {onset_steps}")
    print(f"Durations: {durations}")
    print(f"Episodes per condition: {args_cli.num_episodes}")
    print(f"Output: {args_cli.output_dir}")
    print(f"{'='*60}\n")
    
    # Create output directory
    os.makedirs(args_cli.output_dir, exist_ok=True)
    
    all_summaries = []
    
    for policy_name, checkpoint_path in policies.items():
        print(f"\n{'#'*60}")
        print(f"Evaluating: {policy_name}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"{'#'*60}")
        
        # Determine if force sensor needed
        use_force = args_cli.use_force_sensor or "M2" in policy_name or "M4" in policy_name
        
        if args_cli.variant in ["1", "both"]:
            summary, results = run_variant1_evaluation(
                task=args_cli.task,
                checkpoint_path=checkpoint_path,
                policy_name=policy_name,
                phases=phases,
                durations=durations,
                num_envs=args_cli.num_envs,
                num_episodes=args_cli.num_episodes,
                episode_length=args_cli.episode_length,
                seed=args_cli.seed,
                use_force_sensor=use_force,
            )
            save_results(args_cli.output_dir, summary, results)
            all_summaries.append(summary)
        
        if args_cli.variant in ["2", "both"]:
            summary, results = run_variant2_evaluation(
                task=args_cli.task,
                checkpoint_path=checkpoint_path,
                policy_name=policy_name,
                onset_steps=onset_steps,
                durations=durations,
                num_envs=args_cli.num_envs,
                num_episodes=args_cli.num_episodes,
                episode_length=args_cli.episode_length,
                seed=args_cli.seed,
                use_force_sensor=use_force,
            )
            save_results(args_cli.output_dir, summary, results)
            all_summaries.append(summary)
    
    # Save combined summary
    combined_path = os.path.join(args_cli.output_dir, "all_summaries.json")
    with open(combined_path, 'w') as f:
        json.dump([asdict(s) for s in all_summaries], f, indent=2)
    print(f"\nSaved combined summaries: {combined_path}")
    
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"Results saved to: {args_cli.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
    simulation_app.close()


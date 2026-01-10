#!/usr/bin/env python
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simple wrapper to add modality dropout to existing train.py.

This script modifies the gym.make call to wrap the environment with dropout.
Much simpler than rewriting the entire training script!

Usage:
    python scripts/rsl_rl/train_with_dropout_simple.py --task=Isaac-Franka-PickPlace-Direct-v0 --dropout_mode=mixed
"""

import sys
import os

# Parse dropout arguments BEFORE importing anything else
import argparse
pre_parser = argparse.ArgumentParser(add_help=False)
pre_parser.add_argument("--dropout_mode", type=str, default="hard", choices=["none", "hard", "soft", "mixed"])
pre_parser.add_argument("--dropout_prob", type=float, default=None)
pre_parser.add_argument("--dropout_duration_min", type=int, default=None)
pre_parser.add_argument("--dropout_duration_max", type=int, default=None)
dropout_args, remaining = pre_parser.parse_known_args()

# Store for later use
DROPOUT_MODE = dropout_args.dropout_mode
DROPOUT_PROB = dropout_args.dropout_prob
DROPOUT_DUR_MIN = dropout_args.dropout_duration_min
DROPOUT_DUR_MAX = dropout_args.dropout_duration_max

# Remove dropout args from sys.argv so train.py doesn't see them
sys.argv = [sys.argv[0]] + [arg for arg in remaining if not arg.startswith('--dropout')]

# Now run the normal train.py but intercept gym.make
if __name__ == "__main__":
    print(f"[WRAPPER] starting {__file__}", flush=True)
    print(f"[WRAPPER] sys.argv (post-dropout-strip): {sys.argv}", flush=True)
    print(
        f"[WRAPPER] dropout_mode={DROPOUT_MODE} dropout_prob={DROPOUT_PROB} "
        f"dur_min={DROPOUT_DUR_MIN} dur_max={DROPOUT_DUR_MAX}",
        flush=True,
    )
    # Import train.py as a module
    train_dir = os.path.dirname(__file__)
    sys.path.insert(0, train_dir)
    
    # Patch gym.make BEFORE train.py imports it
    import gymnasium as gym_orig
    original_make = gym_orig.make
    
    def make_with_dropout(id, **kwargs):
        """Wrapped gym.make that adds dropout."""
        # Create base environment
        env = original_make(id, **kwargs)
        
        # Add dropout wrapper if not disabled
        if DROPOUT_MODE != "none":
            try:
                # Add project root to path
                project_root = os.path.abspath(os.path.join(train_dir, "../.."))
                if project_root not in sys.path:
                    sys.path.insert(0, project_root)
                
                from source.Manipulation_policy.Manipulation_policy.tasks.manager_based.manipulation.stack.mdp.dropout_env_wrapper import VecEnvDropoutWrapper
                from source.Manipulation_policy.Manipulation_policy.tasks.manager_based.manipulation.stack.mdp.modality_dropout_cfg import (
                    ModalityDropoutCfg,
                    HardDropoutTrainingCfg,
                    SoftDropoutTrainingCfg,
                    MixedDropoutTrainingCfg,
                )
                
                # Create config
                if DROPOUT_MODE == "hard":
                    cfg = HardDropoutTrainingCfg()
                elif DROPOUT_MODE == "soft":
                    cfg = SoftDropoutTrainingCfg()
                elif DROPOUT_MODE == "mixed":
                    cfg = MixedDropoutTrainingCfg()
                else:
                    cfg = ModalityDropoutCfg(enabled=False)
                
                # Apply overrides
                if DROPOUT_PROB is not None:
                    cfg.dropout_probability = DROPOUT_PROB
                if DROPOUT_DUR_MIN is not None and DROPOUT_DUR_MAX is not None:
                    cfg.dropout_duration_range = (DROPOUT_DUR_MIN, DROPOUT_DUR_MAX)
                
                # Wrap environment
                env = VecEnvDropoutWrapper(env, cfg)
                
                print("="*80)
                print(f"[DROPOUT] Modality dropout ENABLED:")
                print(f"  Mode: {cfg.dropout_mode}")
                print(f"  Probability: {cfg.dropout_probability:.3f}")
                print(f"  Duration range: {cfg.dropout_duration_range} steps")
                print(f"  RGB dropout: {cfg.dropout_rgb}")
                print(f"  Depth dropout: {cfg.dropout_depth}")
                print("="*80)
            except Exception as e:
                print(f"[WARNING] Failed to add dropout wrapper: {e}")
                print(f"[WARNING] Continuing without dropout")
        
        return env
    
    # Replace gym.make
    gym_orig.make = make_with_dropout
    
    # Now import and run train.py
    try:
        import train
    except BaseException as e:
        import traceback

        print("[WRAPPER] failed importing scripts/rsl_rl/train.py", flush=True)
        traceback.print_exc()
        raise

    print("[WRAPPER] imported train.py; deciding entrypoint...", flush=True)
    
    # train.py will use our patched gym.make automatically!
    # Note: importing `train` already launches the Isaac Sim app (it constructs AppLauncher at module import time).
    # We must explicitly run training here; otherwise the wrapper script exits immediately and Kit shuts down.
    #
    # Important: `train.main` is decorated with IsaacLab's `@hydra_task_config(...)`, which invokes Hydra's CLI runner.
    # When called programmatically from another script, Hydra may exit early (sometimes with exit code 0) before
    # ever calling the wrapped function body. To make this wrapper reliable, we bypass Hydra and call the
    # underlying function (`train.main.__wrapped__`) with configs loaded directly from the task registry.
    try:
        has_wrapped = hasattr(train.main, "__wrapped__")
        print(f"[WRAPPER] train.main has __wrapped__: {has_wrapped}", flush=True)

        if has_wrapped:
            from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

            task_name = str(getattr(train.args_cli, "task", "")).split(":")[-1]
            agent_entry = str(getattr(train.args_cli, "agent", "") or "")
            print(f"[WRAPPER] resolved task_name={task_name!r} agent_entry={agent_entry!r}", flush=True)

            if not task_name:
                raise RuntimeError("Missing --task. Please pass --task <task_name>.")

            print("[WRAPPER] loading env_cfg from registry...", flush=True)
            try:
                env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
            except BaseException as e:
                import traceback

                print(f"[WRAPPER] env_cfg load failed: {type(e).__name__}: {e!r}", flush=True)
                traceback.print_exc()
                raise
            print(f"[WRAPPER] env_cfg loaded: {type(env_cfg)}", flush=True)

            print("[WRAPPER] loading agent_cfg from registry...", flush=True)
            try:
                agent_cfg = load_cfg_from_registry(task_name, agent_entry) if agent_entry else None
            except BaseException as e:
                import traceback

                print(f"[WRAPPER] agent_cfg load failed: {type(e).__name__}: {e!r}", flush=True)
                traceback.print_exc()
                raise
            print(f"[WRAPPER] agent_cfg loaded: {type(agent_cfg)}", flush=True)

            # Call the original (undecorated) train.main
            print("[WRAPPER] calling train.main.__wrapped__(env_cfg, agent_cfg)...", flush=True)
            train.main.__wrapped__(env_cfg, agent_cfg)
            print("[WRAPPER] train.main.__wrapped__ returned", flush=True)
        else:
            # Fallback (shouldn't happen): run the decorated entrypoint.
            print("[WRAPPER] calling train.main() (decorated)", flush=True)
            train.main()
    finally:
        # Ensure the simulator is closed even if training errors out.
        if hasattr(train, "simulation_app"):
            print("[WRAPPER] closing simulation_app", flush=True)
            train.simulation_app.close()


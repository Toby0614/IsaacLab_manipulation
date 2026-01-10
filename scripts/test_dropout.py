#!/usr/bin/env python
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test script to verify modality dropout system is working correctly.

This script:
1. Creates a small environment
2. Wraps it with dropout
3. Runs a few episodes
4. Verifies dropout is being applied
5. Visualizes dropout timeline (optional)

Usage:
    python scripts/test_dropout.py
    python scripts/test_dropout.py --mode=soft
    python scripts/test_dropout.py --mode=phase --enable_viz
"""

import argparse
import sys
import os
import torch
import gymnasium as gym

# Add project to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import dropout components
from source.Manipulation_policy.Manipulation_policy.tasks.manager_based.manipulation.stack.mdp.dropout_env_wrapper import VecEnvDropoutWrapper
from source.Manipulation_policy.Manipulation_policy.tasks.manager_based.manipulation.stack.mdp.modality_dropout_cfg import (
    HardDropoutTrainingCfg,
    SoftDropoutTrainingCfg,
    PhaseBasedDropoutCfg,
)


def test_dropout(args):
    """Test dropout functionality."""
    
    print("="*80)
    print("Modality Dropout System - Verification Test")
    print("="*80)
    
    # Create environment
    print(f"\n[1/5] Creating environment with {args.num_envs} parallel envs...")
    try:
        env = gym.make(
            args.task,
            num_envs=args.num_envs,
            render_mode="rgb_array" if args.headless else None,
        )
        print(f"✓ Environment created: {args.task}")
    except Exception as e:
        print(f"✗ Failed to create environment: {e}")
        print("\nMake sure your environment is registered. Try:")
        print("  python scripts/test_dropout.py --task=Isaac-Franka-PickPlace-Direct-v0")
        return False
    
    # Create dropout config
    print(f"\n[2/5] Configuring dropout mode: {args.mode}...")
    if args.mode == "hard":
        dropout_cfg = HardDropoutTrainingCfg()
    elif args.mode == "soft":
        dropout_cfg = SoftDropoutTrainingCfg()
    elif args.mode == "phase":
        dropout_cfg = PhaseBasedDropoutCfg()
    else:
        print(f"✗ Unknown mode: {args.mode}")
        return False
    
    # Increase probability for testing (so we see dropout events quickly)
    dropout_cfg.dropout_probability = 0.1  # 10% per step
    dropout_cfg.dropout_duration_range = (5, 20)  # 0.25-1.0s
    
    print(f"✓ Config: mode={dropout_cfg.dropout_mode}, prob={dropout_cfg.dropout_probability}, "
          f"duration={dropout_cfg.dropout_duration_range}")
    
    # Wrap with dropout
    print(f"\n[3/5] Wrapping environment with dropout...")
    try:
        env = VecEnvDropoutWrapper(
            env,
            dropout_cfg=dropout_cfg,
            enable_phase_detection=(args.mode == "phase"),
        )
        print("✓ Dropout wrapper applied")
    except Exception as e:
        print(f"✗ Failed to wrap environment: {e}")
        return False
    
    # Run test episodes
    print(f"\n[4/5] Running {args.num_steps} steps to observe dropout events...")
    
    obs = env.reset()
    dropout_history = []
    dropout_events = []
    step_count = 0
    
    try:
        for step in range(args.num_steps):
            # Random action
            action = torch.zeros(env.env.unwrapped.num_envs, env.env.unwrapped.action_space.shape[0], device=env.env.unwrapped.device)
            
            # Step
            obs, reward, done, info = env.step(action)
            
            # Track dropout state
            stats = env.get_dropout_stats()
            dropout_active_count = stats['dropout_active_count']
            dropout_history.append(dropout_active_count)
            
            # Record events
            if dropout_active_count > 0 and (len(dropout_events) == 0 or dropout_events[-1]['end'] < step):
                dropout_events.append({
                    'start': step,
                    'end': step,
                    'envs': dropout_active_count
                })
            elif dropout_active_count > 0 and len(dropout_events) > 0:
                dropout_events[-1]['end'] = step
                dropout_events[-1]['envs'] = max(dropout_events[-1]['envs'], dropout_active_count)
            
            # Progress
            if (step + 1) % 20 == 0:
                print(f"  Step {step+1}/{args.num_steps}: {dropout_active_count}/{args.num_envs} envs with dropout active")
            
            step_count = step + 1
    except KeyboardInterrupt:
        print("\n✗ Interrupted by user")
        step_count = step
    except Exception as e:
        print(f"\n✗ Error during stepping: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Analyze results
    print(f"\n[5/5] Analyzing results...")
    
    total_dropout_steps = sum(dropout_history)
    total_possible_steps = step_count * args.num_envs
    dropout_fraction = total_dropout_steps / total_possible_steps if total_possible_steps > 0 else 0
    
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    print(f"Total steps simulated: {step_count}")
    print(f"Total environment-steps: {total_possible_steps}")
    print(f"Dropout events observed: {len(dropout_events)}")
    print(f"Total env-steps with dropout: {total_dropout_steps}")
    print(f"Dropout fraction: {dropout_fraction:.2%}")
    print(f"Expected dropout fraction: ~{dropout_cfg.dropout_probability * sum(dropout_cfg.dropout_duration_range)/2:.2%}")
    
    if len(dropout_events) > 0:
        print(f"\nFirst 5 dropout events:")
        for i, event in enumerate(dropout_events[:5]):
            duration = event['end'] - event['start'] + 1
            print(f"  Event {i+1}: steps {event['start']}-{event['end']} (duration={duration}, envs={event['envs']})")
    
    # Verdict
    print(f"\n{'='*80}")
    if len(dropout_events) > 0:
        print("✓ SUCCESS: Dropout system is working correctly!")
        print(f"  Observed {len(dropout_events)} dropout events")
    else:
        print("⚠ WARNING: No dropout events observed")
        print("  This might be due to low probability + short test duration")
        print("  Try: python scripts/test_dropout.py --num_steps=200")
    print(f"{'='*80}\n")
    
    # Visualization
    if args.enable_viz and len(dropout_history) > 0:
        print("\n[Optional] Generating visualization...")
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 4))
            plt.plot(dropout_history, linewidth=1.5)
            plt.xlabel("Simulation Step")
            plt.ylabel("# Envs with Dropout Active")
            plt.title(f"Dropout Timeline (mode={args.mode})")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            output_path = "/tmp/dropout_timeline.png"
            plt.savefig(output_path, dpi=150)
            print(f"✓ Visualization saved to: {output_path}")
        except ImportError:
            print("⚠ matplotlib not available, skipping visualization")
        except Exception as e:
            print(f"⚠ Visualization failed: {e}")
    
    return len(dropout_events) > 0


def main():
    parser = argparse.ArgumentParser(description="Test modality dropout system")
    
    parser.add_argument(
        "--task",
        type=str,
        default="Isaac-Franka-PickPlace-Direct-v0",
        help="Environment to test"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default="hard",
        choices=["hard", "soft", "phase"],
        help="Dropout mode to test"
    )
    
    parser.add_argument(
        "--num_envs",
        type=int,
        default=16,
        help="Number of parallel environments"
    )
    
    parser.add_argument(
        "--num_steps",
        type=int,
        default=100,
        help="Number of steps to simulate"
    )
    
    parser.add_argument(
        "--enable_viz",
        action="store_true",
        help="Generate visualization (requires matplotlib)"
    )
    
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode"
    )
    
    args = parser.parse_args()
    
    # Run test
    success = test_dropout(args)
    
    # Exit code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()


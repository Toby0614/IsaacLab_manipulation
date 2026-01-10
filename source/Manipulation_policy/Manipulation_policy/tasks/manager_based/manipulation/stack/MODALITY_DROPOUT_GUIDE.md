# Modality Dropout System - User Guide

This guide explains how to use the modality dropout system for vision-based RL robustness studies, as described in `poe2.pdf`.

## Overview

The modality dropout system simulates intermittent vision sensor failures during training and evaluation. It supports:

- **Hard dropout**: Complete vision blackout (all pixels set to 0)
- **Soft dropout**: Noise/corruption (Gaussian noise, cutout, speckle)
- **Duration control**: Continuous failures over multiple steps (e.g., 0.5-2.5 seconds)
- **Phase-aware dropout**: Different failure rates during reach/grasp/lift/transport/place phases
- **Announced vs unannounced**: Policy can optionally observe dropout state

## Quick Start

### Method 1: Using Environment Wrapper (Recommended)

This is the **easiest approach** - wrap your existing environment without modifying any configs:

```python
# In your training script (e.g., scripts/rsl_rl/train.py)
import gymnasium as gym
from Manipulation_policy.tasks.manager_based.manipulation.stack.mdp.dropout_env_wrapper import DropoutEnvWrapper
from Manipulation_policy.tasks.manager_based.manipulation.stack.mdp.modality_dropout_cfg import HardDropoutTrainingCfg

# Create your environment normally
env = gym.make("Isaac-Franka-PickPlace-Direct-v0", num_envs=4096, ...)

# Wrap with dropout - that's it!
dropout_cfg = HardDropoutTrainingCfg()
env = DropoutEnvWrapper(env, dropout_cfg)

# Train normally - dropout is applied automatically
agent.learn(env)
```

**Advantages:**
- No changes to environment configs
- No changes to observation functions  
- Works with your existing system
- Easy to toggle on/off

### Method 2: Modifying Environment Config

If you want dropout baked into the environment config:

**Step 1:** Use dropout-aware observation functions in your config:

```python
# In your environment config file
from isaaclab.managers import ObservationTermCfg as ObsTerm
from ... import mdp

@configclass
class ObservationsCfg:
    @configclass
    class MultiCamCfg(ObsGroup):
        # Change this:
        # multi_cam = ObsTerm(func=mdp.multi_cam_tensor_chw, ...)
        
        # To this:
        multi_cam = ObsTerm(func=mdp.multi_cam_tensor_chw_with_dropout, ...)
```

**Step 2:** Initialize dropout manager in your environment's `__post_init__`:

```python
from ...mdp.modality_dropout_cfg import HardDropoutTrainingCfg
from ...mdp.modality_dropout_manager import ModalityDropoutManager

@configclass
class MyEnvCfg(BaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        # Configure dropout
        self.dropout_cfg = HardDropoutTrainingCfg()
```

**Step 3:** In your environment class, create the manager:

```python
class MyEnv(ManagerBasedRLEnv):
    def __init__(self, cfg, ...):
        super().__init__(cfg, ...)
        
        # Initialize dropout manager
        if hasattr(self.cfg, 'dropout_cfg'):
            self.dropout_manager = ModalityDropoutManager(
                cfg=self.cfg.dropout_cfg,
                num_envs=self.num_envs,
                device=self.device
            )
    
    def step(self, action):
        # Update dropout before observation
        if hasattr(self, 'dropout_manager'):
            self.dropout_manager.step()
        
        # Normal step logic...
        return super().step(action)
    
    def reset(self, ...):
        # Reset dropout state
        if hasattr(self, 'dropout_manager'):
            self.dropout_manager.reset(env_ids)
        
        return super().reset(...)
```

## Configuration Options

### Preset Configurations

We provide several ready-to-use configurations:

#### 1. HardDropoutTrainingCfg
Train with complete vision blackouts (recommended for M3 baseline in paper):
```python
from ...mdp.modality_dropout_cfg import HardDropoutTrainingCfg

cfg = HardDropoutTrainingCfg()
# - 2.5% chance per step to start dropout
# - Dropout lasts 0.5-3.0 seconds
# - Complete blackout (pixels = 0)
```

#### 2. SoftDropoutTrainingCfg  
Train with noisy/corrupted vision instead of blackout:
```python
from ...mdp.modality_dropout_cfg import SoftDropoutTrainingCfg

cfg = SoftDropoutTrainingCfg()
# - 5% chance per step (more frequent but less severe)
# - Adds Gaussian noise, cutouts, speckle
# - Vision degraded but not completely lost
```

#### 3. PhaseBasedDropoutCfg
Different dropout rates per manipulation phase:
```python
from ...mdp.modality_dropout_cfg import PhaseBasedDropoutCfg

cfg = PhaseBasedDropoutCfg()
# - Reach: 1.5% dropout chance
# - Grasp: 5% (most critical!)
# - Lift: 3%
# - Transport: 6% (highest risk)
# - Place: 4%
```

#### 4. EvalDropoutCfg
Deterministic dropout for reproducible evaluation:
```python
from ...mdp.modality_dropout_cfg import EvalDropoutCfg

cfg = EvalDropoutCfg()
# - Dropout starts at step 25 (1.25 seconds)
# - Lasts for 20 steps (1.0 second)
# - Same for all episodes/seeds
```

### Custom Configuration

Create your own configuration:

```python
from ...mdp.modality_dropout_cfg import ModalityDropoutCfg

cfg = ModalityDropoutCfg(
    enabled=True,
    
    # Type of dropout
    dropout_mode="mixed",  # "hard", "soft", or "mixed"
    
    # Duration control
    dropout_probability=0.03,  # 3% chance per step
    dropout_duration_range=(15, 60),  # 0.75-3.0 seconds @ 20Hz
    
    # Phase-aware (optional)
    phase_aware=True,
    phase_dropout_config={
        "reach": 0.01,
        "grasp": 0.05,
        "lift": 0.03,
        "transport": 0.06,
        "place": 0.04,
    },
    
    # Noise parameters (for soft dropout)
    gaussian_noise_std_rgb=25.0,  # On 0-255 scale
    gaussian_noise_std_depth=0.06,  # Normalized
    cutout_prob=0.4,
    cutout_size=(24, 24),
    depth_speckle_prob=0.20,
    
    # Modality selection
    dropout_rgb=True,
    dropout_depth=True,
    
    # Announced vs unannounced
    provide_dropout_indicator=False,  # Policy doesn't know when dropout occurs
)
```

## Phase-Aware Dropout

To use phase-aware dropout, you need to detect which phase each environment is in:

### Automatic Phase Detection

```python
from ...mdp.phase_detector import PickPlacePhaseDetector
from ...mdp.modality_dropout_cfg import PhaseBasedDropoutCfg

# Configure phase detector
phase_detector = PickPlacePhaseDetector(
    goal_pos=(0.70, 0.20, 0.0203),
    table_z=0.0203,
    lift_threshold=0.05,
)

# Enable phase-aware dropout
dropout_cfg = PhaseBasedDropoutCfg()

# Use with wrapper
env = DropoutEnvWrapper(
    env, 
    dropout_cfg,
    enable_phase_detection=True,
    phase_detector_kwargs={
        "goal_pos": (0.70, 0.20, 0.0203),
        "table_z": 0.0203,
    }
)
```

### Manual Phase Updates

If you have custom phase detection logic:

```python
# In your environment step:
phases = my_custom_phase_detector(env)  # Returns list of phase names
env.dropout_manager.update_phases(phases)
```

## Training Multiple Models (Paper Experiment Design)

Based on `poe2.pdf`, you should train 4 models for a complete study:

### M1: Baseline (No Dropout)
```python
# Don't use dropout wrapper, or use disabled config
dropout_cfg = ModalityDropoutCfg(enabled=False)
env = DropoutEnvWrapper(env, dropout_cfg)
```

### M2: Force Sensing (No Dropout Training)
```python
# Add force/contact observations to policy
# Train without dropout (same as M1)
dropout_cfg = ModalityDropoutCfg(enabled=False)
```

### M3: Vision-Only with Dropout Training
```python
# Train with hard dropout
dropout_cfg = HardDropoutTrainingCfg()
env = DropoutEnvWrapper(env, dropout_cfg)
```

### M4: Force Sensing with Dropout Training (Best)
```python
# Add force observations + train with dropout
dropout_cfg = HardDropoutTrainingCfg()
env = DropoutEnvWrapper(env, dropout_cfg)
```

## Evaluation

### Deterministic Dropout Schedule

For reproducible evaluation across all models:

```python
from ...mdp.modality_dropout_cfg import EvalDropoutCfg

# Test dropout at specific time + duration
eval_cfg = EvalDropoutCfg()
eval_cfg.eval_dropout_start_step = 30  # Start at 1.5s
eval_cfg.eval_dropout_duration = 25    # Last for 1.25s

env = DropoutEnvWrapper(env, eval_cfg)

# Run evaluation
results = evaluate_policy(policy, env, n_episodes=500)
```

### Sweep Dropout Duration

Test sensitivity to dropout duration:

```python
durations = [5, 10, 20, 40, 60]  # 0.25s to 3.0s
results = {}

for duration in durations:
    cfg = EvalDropoutCfg()
    cfg.eval_dropout_duration = duration
    
    env = DropoutEnvWrapper(base_env, cfg)
    success_rate = evaluate_policy(policy, env)
    results[duration] = success_rate
    
# Plot: success rate vs dropout duration
```

### Phase-Based Evaluation

Test dropout at different task phases:

```python
phases = ["reach", "grasp", "lift", "transport", "place"]
phase_start_steps = [5, 15, 25, 35, 45]  # Approximate timings

results = {}
for phase, start_step in zip(phases, phase_start_steps):
    cfg = EvalDropoutCfg()
    cfg.eval_dropout_start_step = start_step
    cfg.eval_dropout_duration = 20  # 1 second
    
    env = DropoutEnvWrapper(base_env, cfg)
    success_rate = evaluate_policy(policy, env)
    results[phase] = success_rate
```

## Logging and Debugging

### Get Dropout Statistics

```python
# During training/evaluation
stats = env.get_dropout_stats()
print(stats)
# {
#   'dropout_active_count': 123,
#   'dropout_active_fraction': 0.03,
#   'avg_remaining_steps': 15.2
# }

# Log to TensorBoard/Weights & Biases
logger.log_scalar("dropout/active_fraction", stats['dropout_active_fraction'], step)
```

### Visualize Dropout Events

```python
# Track dropout state over episode
dropout_history = []

obs = env.reset()
for step in range(max_steps):
    action = policy(obs)
    obs, reward, done, info = env.step(action)
    
    # Record if dropout was active
    dropout_active = env.dropout_manager.dropout_active[0].item()  # Env 0
    dropout_history.append(dropout_active)

# Plot dropout timeline
import matplotlib.pyplot as plt
plt.plot(dropout_history)
plt.xlabel("Step")
plt.ylabel("Dropout Active")
plt.title("Dropout Events Over Episode")
plt.show()
```

## Troubleshooting

### Dropout not being applied

**Check 1:** Is dropout enabled?
```python
print(env.dropout_manager.cfg.enabled)  # Should be True
```

**Check 2:** Are you using dropout-aware observation functions?
```python
# Your config should use:
# mdp.multi_cam_tensor_chw_with_dropout (with dropout)
# NOT
# mdp.multi_cam_tensor_chw (original, no dropout)
```

**Check 3:** Is dropout manager attached to env?
```python
print(hasattr(env.unwrapped, 'dropout_manager'))  # Should be True
```

### Dropout too frequent/infrequent

Adjust probability and duration:
```python
cfg.dropout_probability = 0.01  # 1% per step (decrease for less frequent)
cfg.dropout_duration_range = (20, 80)  # 1-4 seconds (increase for longer events)
```

### Training unstable with dropout

Start with curriculum:
```python
# Early training: low dropout probability
cfg_early = ModalityDropoutCfg(enabled=True, dropout_probability=0.01)

# After 50% of training: increase
cfg_late = ModalityDropoutCfg(enabled=True, dropout_probability=0.03)
```

Or use soft dropout first:
```python
# Train with noise first
cfg = SoftDropoutTrainingCfg()

# Then fine-tune with hard dropout
cfg = HardDropoutTrainingCfg()
```

## File Reference

```
mdp/
├── modality_dropout_cfg.py              # Configuration classes
├── modality_dropout_manager.py          # Core dropout logic
├── modality_dropout_observations.py     # Dropout-aware observation functions
├── dropout_env_wrapper.py               # Environment wrapper (recommended!)
├── phase_detector.py                    # Automatic phase detection
└── MODALITY_DROPOUT_GUIDE.md           # This file
```

## Examples

### Example 1: Quick Test

```python
import gymnasium as gym
from Manipulation_policy.tasks.manager_based.manipulation.stack.mdp.dropout_env_wrapper import DropoutEnvWrapper
from Manipulation_policy.tasks.manager_based.manipulation.stack.mdp.modality_dropout_cfg import HardDropoutTrainingCfg

env = gym.make("Isaac-Franka-PickPlace-Direct-v0", num_envs=16)
env = DropoutEnvWrapper(env, HardDropoutTrainingCfg())

obs = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    
    if done.any():
        print(f"Stats: {env.get_dropout_stats()}")
```

### Example 2: Training Script Integration

```python
# scripts/rsl_rl/train.py (add these lines)

# After environment creation:
if args.enable_dropout:
    from Manipulation_policy.tasks.manager_based.manipulation.stack.mdp.dropout_env_wrapper import DropoutEnvWrapper
    from Manipulation_policy.tasks.manager_based.manipulation.stack.mdp.modality_dropout_cfg import HardDropoutTrainingCfg
    
    dropout_cfg = HardDropoutTrainingCfg()
    env = DropoutEnvWrapper(env, dropout_cfg)
    print(f"[INFO] Modality dropout enabled: {dropout_cfg.dropout_mode}")

# Then train normally
```

### Example 3: Evaluation Script

```python
# scripts/rsl_rl/play.py (add evaluation loop)

from Manipulation_policy.tasks.manager_based.manipulation.stack.mdp.dropout_env_wrapper import DropoutEnvWrapper
from Manipulation_policy.tasks.manager_based.manipulation.stack.mdp.modality_dropout_cfg import EvalDropoutCfg

# Test different dropout durations
durations = [0, 10, 20, 30, 40, 50]
for duration in durations:
    cfg = EvalDropoutCfg()
    cfg.eval_dropout_duration = duration
    cfg.enabled = (duration > 0)
    
    env_eval = DropoutEnvWrapper(env, cfg)
    
    successes = 0
    for episode in range(100):
        obs = env_eval.reset()
        done = False
        while not done:
            action = policy(obs)
            obs, reward, done, info = env_eval.step(action)
            if info.get('success', False):
                successes += 1
    
    success_rate = successes / 100
    print(f"Dropout duration {duration*0.05:.2f}s: {success_rate:.1%} success")
```

## Citation

If you use this modality dropout system in your research, please cite the original paper and Isaac Lab:

```bibtex
@article{your_paper_2026,
  title={Robust Vision-Based RL Manipulation Under Intermittent Sensor Failures via Gripper Force Sensing},
  author={Your Name},
  journal={Your Conference},
  year={2026}
}

@misc{isaaclab,
  author = {Isaac Lab Project Contributors},
  title = {Isaac Lab},
  year = {2025},
  url = {https://github.com/isaac-sim/IsaacLab}
}
```


# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment wrapper that adds modality dropout functionality.

This wrapper automatically manages the ModalityDropoutManager lifecycle,
so you don't need to manually initialize or update it.

Usage:
    # In your training script (e.g., scripts/rsl_rl/train.py):
    from Manipulation_policy.tasks.manager_based.manipulation.stack.mdp.dropout_env_wrapper import DropoutEnvWrapper
    from Manipulation_policy.tasks.manager_based.manipulation.stack.mdp.modality_dropout_cfg import HardDropoutTrainingCfg
    
    # Create base environment
    env = gym.make("Isaac-Franka-PickPlace-Direct-v0", ...)
    
    # Wrap with dropout
    dropout_cfg = HardDropoutTrainingCfg()
    env = DropoutEnvWrapper(env, dropout_cfg)
    
    # Train normally - dropout is applied automatically!
"""

from __future__ import annotations

import torch
import gymnasium as gym
from typing import Any

from .modality_dropout_cfg import ModalityDropoutCfg
from .modality_dropout_manager import ModalityDropoutManager
from .phase_detector import PickPlacePhaseDetector


class DropoutEnvWrapper(gym.Wrapper):
    """Gym wrapper that adds modality dropout to an IsaacLab environment.
    
    This wrapper:
    1. Creates and manages a ModalityDropoutManager
    2. Automatically updates dropout state each step
    3. Resets dropout state on episode reset
    4. Optionally detects and updates task phases for phase-aware dropout
    
    The wrapped environment will have env.dropout_manager available,
    which the observation functions (e.g., multi_cam_tensor_chw_with_dropout)
    will automatically use.
    """
    
    def __init__(
        self,
        env: gym.Env,
        dropout_cfg: ModalityDropoutCfg,
        enable_phase_detection: bool = False,
        phase_detector_kwargs: dict | None = None,
    ):
        """Initialize the dropout wrapper.
        
        Args:
            env: Base IsaacLab environment to wrap
            dropout_cfg: Modality dropout configuration
            enable_phase_detection: If True, automatically detect and update task phases
            phase_detector_kwargs: Kwargs for PickPlacePhaseDetector (if phase detection enabled)
        """
        super().__init__(env)
        
        self.dropout_cfg = dropout_cfg
        self.enable_phase_detection = enable_phase_detection or dropout_cfg.phase_aware
        
        # Create dropout manager
        self.dropout_manager = ModalityDropoutManager(
            cfg=dropout_cfg,
            num_envs=self.env.unwrapped.num_envs,
            device=str(self.env.unwrapped.device),
        )
        
        # Attach to unwrapped env so observation functions can access it
        self.env.unwrapped.dropout_manager = self.dropout_manager
        
        # Create phase detector if needed
        if self.enable_phase_detection:
            phase_kwargs = phase_detector_kwargs or {}
            self.phase_detector = PickPlacePhaseDetector(**phase_kwargs)
        else:
            self.phase_detector = None
        
        print(f"[DropoutEnvWrapper] Initialized with mode={dropout_cfg.dropout_mode}, "
              f"enabled={dropout_cfg.enabled}, phase_aware={dropout_cfg.phase_aware}")
    
    def reset(self, **kwargs):
        """Reset environment and dropout state."""
        result = self.env.reset(**kwargs)
        
        # Reset dropout for all environments
        all_env_ids = torch.arange(self.env.unwrapped.num_envs, device=self.env.unwrapped.device)
        self.dropout_manager.reset(all_env_ids)
        
        return result
    
    def step(self, action):
        """Step environment and update dropout state."""
        # Step base environment (IsaacLab returns 5 values)
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Update phase detection (if enabled)
        if self.enable_phase_detection and self.phase_detector is not None:
            try:
                phases = self.phase_detector.detect_phases(self.env.unwrapped)
                self.dropout_manager.update_phases(phases)
            except Exception as e:
                # Fallback: if phase detection fails, just use default
                print(f"[DropoutEnvWrapper] Phase detection failed: {e}")
        
        # Update dropout state (must happen BEFORE next observation is generated)
        self.dropout_manager.step()
        
        # Handle resets for terminated/truncated environments
        done = terminated | truncated
        reset_env_ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.dropout_manager.reset(reset_env_ids)
        
        return obs, reward, terminated, truncated, info
    
    def get_dropout_stats(self) -> dict:
        """Get current dropout statistics (useful for logging)."""
        return self.dropout_manager.get_stats()


class VecEnvDropoutWrapper:
    """Alternative wrapper for IsaacLab VecEnv interface.
    
    Use this if the standard gym.Wrapper doesn't work with your training loop.
    
    Usage:
        env = gym.make(...)
        env = VecEnvDropoutWrapper(env, dropout_cfg)
    """
    
    def __init__(
        self,
        env,
        dropout_cfg: ModalityDropoutCfg,
        enable_phase_detection: bool = False,
        phase_detector_kwargs: dict | None = None,
    ):
        """Initialize wrapper."""
        self.env = env
        self.dropout_cfg = dropout_cfg
        self.enable_phase_detection = enable_phase_detection or dropout_cfg.phase_aware
        
        # Get unwrapped env
        if hasattr(env, 'unwrapped'):
            base_env = env.unwrapped
        else:
            base_env = env
        
        # Create dropout manager
        self.dropout_manager = ModalityDropoutManager(
            cfg=dropout_cfg,
            num_envs=base_env.num_envs,
            device=str(base_env.device),
        )
        
        # Attach to env
        base_env.dropout_manager = self.dropout_manager
        
        # Phase detector
        if self.enable_phase_detection:
            phase_kwargs = phase_detector_kwargs or {}
            self.phase_detector = PickPlacePhaseDetector(**phase_kwargs)
        else:
            self.phase_detector = None
        
        print(f"[VecEnvDropoutWrapper] Initialized with mode={dropout_cfg.dropout_mode}, "
              f"enabled={dropout_cfg.enabled}")
    
    def reset(self, **kwargs):
        """Reset environment."""
        result = self.env.reset(**kwargs)
        
        base_env = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
        all_env_ids = torch.arange(base_env.num_envs, device=base_env.device)
        self.dropout_manager.reset(all_env_ids)
        
        return result
    
    def step(self, action):
        """Step environment."""
        # IsaacLab uses new Gymnasium API: (obs, reward, terminated, truncated, info)
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Update phases
        if self.enable_phase_detection and self.phase_detector is not None:
            try:
                base_env = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
                phases = self.phase_detector.detect_phases(base_env)
                self.dropout_manager.update_phases(phases)
            except Exception:
                pass
        
        # Update dropout
        self.dropout_manager.step()
        
        # Reset terminated/truncated envs
        base_env = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
        done = terminated | truncated
        reset_env_ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.dropout_manager.reset(reset_env_ids)
        
        return obs, reward, terminated, truncated, info
    
    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to wrapped env."""
        return getattr(self.env, name)
    
    def get_dropout_stats(self) -> dict:
        """Get dropout statistics."""
        return self.dropout_manager.get_stats()


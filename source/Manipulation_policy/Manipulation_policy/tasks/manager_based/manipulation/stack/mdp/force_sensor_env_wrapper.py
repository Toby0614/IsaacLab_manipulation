# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment wrapper that adds gripper force sensing (tensile sensor) to observations.

This wrapper adds gripper force data to the proprioceptive observations WITHOUT
modifying the underlying environment configuration or reward system. It's designed
to work with the Isaac-Franka-PickPlace-v0 task (or any similar task).

Usage:
    # In your training script (e.g., scripts/rsl_rl/train.py):
    from Manipulation_policy.tasks.manager_based.manipulation.stack.mdp.force_sensor_env_wrapper import ForceSensorEnvWrapper
    
    # Create base environment
    env = gym.make("Isaac-Franka-PickPlace-v0", ...)
    
    # Wrap with force sensing
    env = ForceSensorEnvWrapper(env)
    
    # Train normally - force data is automatically added to proprio observations!
    
The wrapper:
1. Intercepts observations from the base environment
2. Computes gripper force from joint torques
3. Concatenates force data to the 'proprio' observation group
4. Updates the observation space accordingly
"""

from __future__ import annotations

import torch
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Any
from dataclasses import dataclass

from isaaclab.utils import configclass


@configclass
class ForceSensorConfig:
    """Configuration for the force sensor wrapper."""
    
    enabled: bool = True
    """Whether force sensing is enabled."""
    
    force_obs_mode: str = "scalar"
    """Force observation mode:
    - "scalar": Single average force value (1 dim)
    - "per_finger": Force for each finger (2 dims)
    - "with_closure": Force + closure + product (3 dims)
    - "contact_estimate": Contact force + indicator (2 dims)
    - "grasp_indicator": Force + quality + is_grasping (3 dims)
    """
    
    normalize: bool = True
    """Whether to normalize force values to [0, ~1] range."""
    
    effort_limit: float = 70.0
    """Maximum expected joint effort for normalization (N for Franka fingers)."""
    
    proximity_threshold: float = 0.06
    """Distance threshold for contact detection (meters)."""
    
    robot_name: str = "robot"
    """Name of the robot asset in the scene."""
    
    object_name: str = "cube_2"
    """Name of the target object for contact estimation."""
    
    ee_frame_name: str = "ee_frame"
    """Name of the end-effector frame sensor."""


# Presets for common configurations
@configclass
class SimpleForceConfig(ForceSensorConfig):
    """Simple single-value force sensing (minimal overhead)."""
    force_obs_mode: str = "scalar"


@configclass
class DetailedForceConfig(ForceSensorConfig):
    """Detailed force sensing with grasp quality indicators."""
    force_obs_mode: str = "grasp_indicator"


@configclass
class ContactForceConfig(ForceSensorConfig):
    """Contact-aware force sensing."""
    force_obs_mode: str = "contact_estimate"


class ForceSensorEnvWrapper(gym.Wrapper):
    """Gym wrapper that adds gripper force sensing to an IsaacLab environment.
    
    This wrapper:
    1. Computes gripper force from joint applied torques
    2. Adds force observations to the 'proprio' observation group
    3. Updates the observation space to reflect the new dimensions
    4. Does NOT modify rewards, terminations, or any other environment behavior
    
    The force sensing simulates a tensile/force sensor at the gripper without
    requiring physical sensor simulation, making it computationally efficient.
    """
    
    def __init__(
        self,
        env: gym.Env,
        cfg: ForceSensorConfig | None = None,
    ):
        """Initialize the force sensor wrapper.
        
        Args:
            env: Base IsaacLab environment to wrap
            cfg: Force sensor configuration (uses SimpleForceConfig if None)
        """
        super().__init__(env)
        
        self.cfg = cfg if cfg is not None else SimpleForceConfig()
        # If we patch the unwrapped observation_manager, we must avoid double-augmenting in reset/step.
        self._obs_manager_patched: bool = False
        self._orig_obs_manager_compute = None
        
        # Store the unwrapped env for direct access
        self._unwrapped = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
        
        # Calculate force observation dimensions based on mode
        self._force_dim = self._get_force_dim()
        
        # Update observation space
        self._update_observation_space()

        # IMPORTANT: RslRlVecEnvWrapper.get_observations() bypasses gym wrappers and calls
        # `unwrapped.observation_manager.compute()`. Patch it so all callers see consistent dims.
        self._maybe_patch_observation_manager()
        
        print(f"[ForceSensorEnvWrapper] Initialized with mode='{self.cfg.force_obs_mode}' "
              f"({self._force_dim} dims), enabled={self.cfg.enabled}")

    def _maybe_patch_observation_manager(self) -> None:
        """Patch unwrapped.observation_manager.compute() to include force augmentation."""
        try:
            base_env = self._unwrapped
            obs_mgr = getattr(base_env, "observation_manager", None)
            if obs_mgr is None or not hasattr(obs_mgr, "compute"):
                return

            # Store original compute for potential debugging; don't patch twice.
            if getattr(obs_mgr, "_force_sensor_compute_patched", False):
                self._obs_manager_patched = True
                return

            self._orig_obs_manager_compute = obs_mgr.compute

            def _compute_with_force(*args, **kwargs):
                obs_dict = self._orig_obs_manager_compute(*args, **kwargs)  # type: ignore[misc]
                return self._augment_observations(obs_dict)

            obs_mgr.compute = _compute_with_force  # type: ignore[assignment]
            setattr(obs_mgr, "_force_sensor_compute_patched", True)
            self._obs_manager_patched = True
        except Exception:
            # Best-effort: wrapper will still augment reset/step path if patching fails.
            self._obs_manager_patched = False
    
    def _get_force_dim(self) -> int:
        """Get the number of force observation dimensions based on mode."""
        mode_dims = {
            "scalar": 1,
            "per_finger": 2,
            "with_closure": 3,
            "contact_estimate": 2,
            "grasp_indicator": 3,
        }
        return mode_dims.get(self.cfg.force_obs_mode, 1)
    
    def _update_observation_space(self):
        """Update the observation space to include force observations."""
        if not self.cfg.enabled:
            return
        
        # Get original observation space
        orig_space = self.env.observation_space
        
        def _append_last_axis(arr: np.ndarray, append_dim: int, fill_value: float) -> np.ndarray:
            arr = np.asarray(arr)
            pad_shape = (*arr.shape[:-1], append_dim)
            pad = np.full(pad_shape, fill_value, dtype=np.float32)
            return np.concatenate([arr, pad], axis=-1)

        if isinstance(orig_space, spaces.Dict):
            # Dict observation space - augment 'proprio' if it exists
            new_spaces = {}
            for key, space in orig_space.spaces.items():
                if key == "proprio" and isinstance(space, spaces.Box):
                    # Add force dimensions to proprio
                    new_low = _append_last_axis(space.low, self._force_dim, 0.0)
                    new_high = _append_last_axis(space.high, self._force_dim, 2.0)
                    new_spaces[key] = spaces.Box(low=new_low, high=new_high, dtype=np.float32)
                else:
                    new_spaces[key] = space
            
            # Add force as separate key if proprio doesn't exist
            if "proprio" not in new_spaces:
                new_spaces["gripper_force"] = spaces.Box(
                    low=np.zeros(self._force_dim, dtype=np.float32),
                    high=np.full(self._force_dim, 2.0, dtype=np.float32),
                    dtype=np.float32,
                )
            
            self.observation_space = spaces.Dict(new_spaces)
        else:
            # Flat observation space - append force dimensions
            if isinstance(orig_space, spaces.Box):
                new_low = _append_last_axis(orig_space.low, self._force_dim, 0.0)
                new_high = _append_last_axis(orig_space.high, self._force_dim, 2.0)
                self.observation_space = spaces.Box(low=new_low, high=new_high, dtype=np.float32)
    
    def _compute_force_obs(self) -> torch.Tensor:
        """Compute force observation based on configuration mode."""
        from . import gripper_force_observations as gfo
        
        env = self._unwrapped
        
        if self.cfg.force_obs_mode == "scalar":
            return gfo.gripper_force_scalar(
                env,
                robot_name=self.cfg.robot_name,
                normalize=self.cfg.normalize,
                effort_limit=self.cfg.effort_limit,
            )
        elif self.cfg.force_obs_mode == "per_finger":
            return gfo.gripper_force_obs(
                env,
                robot_name=self.cfg.robot_name,
                normalize=self.cfg.normalize,
                effort_limit=self.cfg.effort_limit,
            )
        elif self.cfg.force_obs_mode == "with_closure":
            return gfo.gripper_force_with_closure(
                env,
                robot_name=self.cfg.robot_name,
                normalize=self.cfg.normalize,
                effort_limit=self.cfg.effort_limit,
            )
        elif self.cfg.force_obs_mode == "contact_estimate":
            return gfo.gripper_contact_force_estimate(
                env,
                robot_name=self.cfg.robot_name,
                object_name=self.cfg.object_name,
                ee_frame_name=self.cfg.ee_frame_name,
                proximity_threshold=self.cfg.proximity_threshold,
                normalize=self.cfg.normalize,
                effort_limit=self.cfg.effort_limit,
            )
        elif self.cfg.force_obs_mode == "grasp_indicator":
            return gfo.gripper_grasp_force_indicator(
                env,
                robot_name=self.cfg.robot_name,
                object_name=self.cfg.object_name,
                ee_frame_name=self.cfg.ee_frame_name,
                proximity_threshold=self.cfg.proximity_threshold,
                normalize=self.cfg.normalize,
                effort_limit=self.cfg.effort_limit,
            )
        else:
            # Default fallback
            return gfo.gripper_force_scalar(
                env,
                robot_name=self.cfg.robot_name,
                normalize=self.cfg.normalize,
                effort_limit=self.cfg.effort_limit,
            )
    
    def _augment_observations(self, obs: dict | torch.Tensor) -> dict | torch.Tensor:
        """Augment observations with force data."""
        if not self.cfg.enabled:
            return obs
        
        force_obs = self._compute_force_obs()
        
        if isinstance(obs, dict):
            # Dict observations - augment 'proprio'
            if "proprio" in obs:
                obs["proprio"] = torch.cat([obs["proprio"], force_obs], dim=-1)
            else:
                # Add as separate key
                obs["gripper_force"] = force_obs
        elif isinstance(obs, torch.Tensor):
            # Flat tensor - concatenate
            obs = torch.cat([obs, force_obs], dim=-1)
        
        return obs
    
    def reset(self, **kwargs):
        """Reset environment and return augmented observations."""
        result = self.env.reset(**kwargs)
        
        # Handle both old (obs, info) and new (obs,) API
        if isinstance(result, tuple):
            obs, info = result
            if not self._obs_manager_patched:
                obs = self._augment_observations(obs)
            return obs, info
        else:
            return self._augment_observations(result) if not self._obs_manager_patched else result
    
    def step(self, action):
        """Step environment and return augmented observations."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        if not self._obs_manager_patched:
            obs = self._augment_observations(obs)
        return obs, reward, terminated, truncated, info
    
    def get_force_obs_dim(self) -> int:
        """Get the number of force observation dimensions."""
        return self._force_dim
    
    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to wrapped env."""
        return getattr(self.env, name)


class VecEnvForceSensorWrapper:
    """Alternative wrapper for IsaacLab VecEnv interface.
    
    Use this if the standard gym.Wrapper doesn't work with your training loop.
    This version uses __getattr__ forwarding for better compatibility.
    
    Usage:
        env = gym.make(...)
        env = VecEnvForceSensorWrapper(env, cfg=SimpleForceConfig())
    """
    
    def __init__(
        self,
        env,
        cfg: ForceSensorConfig | None = None,
    ):
        """Initialize wrapper."""
        self.env = env
        self.cfg = cfg if cfg is not None else SimpleForceConfig()
        self._obs_manager_patched: bool = False
        self._orig_obs_manager_compute = None
        
        # Get unwrapped env
        self._unwrapped = env.unwrapped if hasattr(env, 'unwrapped') else env
        
        # Calculate force observation dimensions
        self._force_dim = self._get_force_dim()
        
        # Update observation space
        self._update_observation_space()

        # IMPORTANT: RslRlVecEnvWrapper.get_observations() calls unwrapped.observation_manager.compute(),
        # so we patch that to return augmented proprio dims consistently (prevents 89 vs 92 mismatch).
        self._maybe_patch_observation_manager()
        
        print(f"[VecEnvForceSensorWrapper] Initialized with mode='{self.cfg.force_obs_mode}' "
              f"({self._force_dim} dims)")

    def _maybe_patch_observation_manager(self) -> None:
        """Patch unwrapped.observation_manager.compute() to include force augmentation."""
        try:
            base_env = self._unwrapped
            obs_mgr = getattr(base_env, "observation_manager", None)
            if obs_mgr is None or not hasattr(obs_mgr, "compute"):
                return

            if getattr(obs_mgr, "_force_sensor_compute_patched", False):
                self._obs_manager_patched = True
                return

            self._orig_obs_manager_compute = obs_mgr.compute

            def _compute_with_force(*args, **kwargs):
                obs_dict = self._orig_obs_manager_compute(*args, **kwargs)  # type: ignore[misc]
                return self._augment_observations(obs_dict)

            obs_mgr.compute = _compute_with_force  # type: ignore[assignment]
            setattr(obs_mgr, "_force_sensor_compute_patched", True)
            self._obs_manager_patched = True
        except Exception:
            self._obs_manager_patched = False
    
    def _get_force_dim(self) -> int:
        """Get the number of force observation dimensions based on mode."""
        mode_dims = {
            "scalar": 1,
            "per_finger": 2,
            "with_closure": 3,
            "contact_estimate": 2,
            "grasp_indicator": 3,
        }
        return mode_dims.get(self.cfg.force_obs_mode, 1)
    
    def _update_observation_space(self):
        """Update observation space to include force."""
        if not self.cfg.enabled:
            self.observation_space = self.env.observation_space
            return
        
        orig_space = self.env.observation_space
        
        def _append_last_axis(arr: np.ndarray, append_dim: int, fill_value: float) -> np.ndarray:
            """Append `append_dim` values to the last axis of a numpy array.

            IsaacLab/VecEnv spaces sometimes expose low/high as (D,) or (N, D).
            We always append along the last axis to preserve batch-like leading dims.
            """
            arr = np.asarray(arr)
            pad_shape = (*arr.shape[:-1], append_dim)
            pad = np.full(pad_shape, fill_value, dtype=np.float32)
            return np.concatenate([arr, pad], axis=-1)

        if isinstance(orig_space, spaces.Dict):
            new_spaces = {}
            for key, space in orig_space.spaces.items():
                if key == "proprio" and isinstance(space, spaces.Box):
                    # Append force bounds to the last dimension (handles (D,) and (N,D)).
                    new_low = _append_last_axis(space.low, self._force_dim, 0.0)
                    new_high = _append_last_axis(space.high, self._force_dim, 2.0)
                    new_spaces[key] = spaces.Box(low=new_low, high=new_high, dtype=np.float32)
                else:
                    new_spaces[key] = space
            self.observation_space = spaces.Dict(new_spaces)
        elif isinstance(orig_space, spaces.Box):
            new_low = _append_last_axis(orig_space.low, self._force_dim, 0.0)
            new_high = _append_last_axis(orig_space.high, self._force_dim, 2.0)
            self.observation_space = spaces.Box(low=new_low, high=new_high, dtype=np.float32)
        else:
            self.observation_space = orig_space
    
    def _compute_force_obs(self) -> torch.Tensor:
        """Compute force observation."""
        from . import gripper_force_observations as gfo
        
        env = self._unwrapped
        
        if self.cfg.force_obs_mode == "scalar":
            return gfo.gripper_force_scalar(env, self.cfg.robot_name, self.cfg.normalize, self.cfg.effort_limit)
        elif self.cfg.force_obs_mode == "per_finger":
            return gfo.gripper_force_obs(env, self.cfg.robot_name, self.cfg.normalize, self.cfg.effort_limit)
        elif self.cfg.force_obs_mode == "with_closure":
            return gfo.gripper_force_with_closure(env, self.cfg.robot_name, self.cfg.normalize, self.cfg.effort_limit)
        elif self.cfg.force_obs_mode == "contact_estimate":
            return gfo.gripper_contact_force_estimate(
                env, self.cfg.robot_name, self.cfg.object_name, self.cfg.ee_frame_name,
                self.cfg.proximity_threshold, self.cfg.normalize, self.cfg.effort_limit
            )
        elif self.cfg.force_obs_mode == "grasp_indicator":
            return gfo.gripper_grasp_force_indicator(
                env, self.cfg.robot_name, self.cfg.object_name, self.cfg.ee_frame_name,
                self.cfg.proximity_threshold, 0.1, self.cfg.normalize, self.cfg.effort_limit
            )
        else:
            return gfo.gripper_force_scalar(env, self.cfg.robot_name, self.cfg.normalize, self.cfg.effort_limit)
    
    def _augment_observations(self, obs):
        """Augment observations with force data."""
        if not self.cfg.enabled:
            return obs
        
        force_obs = self._compute_force_obs()
        
        if isinstance(obs, dict):
            if "proprio" in obs:
                obs["proprio"] = torch.cat([obs["proprio"], force_obs], dim=-1)
            else:
                obs["gripper_force"] = force_obs
        elif isinstance(obs, torch.Tensor):
            obs = torch.cat([obs, force_obs], dim=-1)
        
        return obs
    
    def reset(self, **kwargs):
        """Reset environment."""
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            obs, info = result
            if not self._obs_manager_patched:
                obs = self._augment_observations(obs)
            return obs, info
        return self._augment_observations(result) if not self._obs_manager_patched else result
    
    def step(self, action):
        """Step environment."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        if not self._obs_manager_patched:
            obs = self._augment_observations(obs)
        return obs, reward, terminated, truncated, info
    
    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to wrapped env."""
        return getattr(self.env, name)
    
    @property
    def unwrapped(self):
        """Get the unwrapped environment."""
        return self._unwrapped


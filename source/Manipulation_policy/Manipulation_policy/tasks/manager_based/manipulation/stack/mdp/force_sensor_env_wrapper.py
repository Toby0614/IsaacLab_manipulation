

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
    
    enabled: bool = True
    
    force_obs_mode: str = "scalar"
    
    normalize: bool = True
    
    effort_limit: float = 70.0
    
    proximity_threshold: float = 0.06
    
    robot_name: str = "robot"
    
    object_name: str = "cube_2"
    
    ee_frame_name: str = "ee_frame"


@configclass
class SimpleForceConfig(ForceSensorConfig):
    force_obs_mode: str = "scalar"


@configclass
class DetailedForceConfig(ForceSensorConfig):
    force_obs_mode: str = "grasp_indicator"


@configclass
class ContactForceConfig(ForceSensorConfig):
    force_obs_mode: str = "contact_estimate"


class ForceSensorEnvWrapper(gym.Wrapper):
    
    def __init__(
        self,
        env: gym.Env,
        cfg: ForceSensorConfig | None = None,
    ):
        super().__init__(env)
        
        self.cfg = cfg if cfg is not None else SimpleForceConfig()
        self._obs_manager_patched: bool = False
        self._orig_obs_manager_compute = None
        
        self._unwrapped = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
        
        self._force_dim = self._get_force_dim()
        
        self._update_observation_space()

        self._maybe_patch_observation_manager()
        
        print(f"[ForceSensorEnvWrapper] Initialized with mode='{self.cfg.force_obs_mode}' "
              f"({self._force_dim} dims), enabled={self.cfg.enabled}")

    def _maybe_patch_observation_manager(self) -> None:
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
        mode_dims = {
            "scalar": 1,
            "per_finger": 2,
            "with_closure": 3,
            "contact_estimate": 2,
            "grasp_indicator": 3,
        }
        return mode_dims.get(self.cfg.force_obs_mode, 1)
    
    def _update_observation_space(self):
        if not self.cfg.enabled:
            return
        
        orig_space = self.env.observation_space
        
        def _append_last_axis(arr: np.ndarray, append_dim: int, fill_value: float) -> np.ndarray:
            arr = np.asarray(arr)
            pad_shape = (*arr.shape[:-1], append_dim)
            pad = np.full(pad_shape, fill_value, dtype=np.float32)
            return np.concatenate([arr, pad], axis=-1)

        if isinstance(orig_space, spaces.Dict):
            new_spaces = {}
            for key, space in orig_space.spaces.items():
                if key == "proprio" and isinstance(space, spaces.Box):
                    new_low = _append_last_axis(space.low, self._force_dim, 0.0)
                    new_high = _append_last_axis(space.high, self._force_dim, 2.0)
                    new_spaces[key] = spaces.Box(low=new_low, high=new_high, dtype=np.float32)
                else:
                    new_spaces[key] = space
            
            if "proprio" not in new_spaces:
                new_spaces["gripper_force"] = spaces.Box(
                    low=np.zeros(self._force_dim, dtype=np.float32),
                    high=np.full(self._force_dim, 2.0, dtype=np.float32),
                    dtype=np.float32,
                )
            
            self.observation_space = spaces.Dict(new_spaces)
        else:
            if isinstance(orig_space, spaces.Box):
                new_low = _append_last_axis(orig_space.low, self._force_dim, 0.0)
                new_high = _append_last_axis(orig_space.high, self._force_dim, 2.0)
                self.observation_space = spaces.Box(low=new_low, high=new_high, dtype=np.float32)
    
    def _compute_force_obs(self) -> torch.Tensor:
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
            return gfo.gripper_force_scalar(
                env,
                robot_name=self.cfg.robot_name,
                normalize=self.cfg.normalize,
                effort_limit=self.cfg.effort_limit,
            )
    
    def _augment_observations(self, obs: dict | torch.Tensor) -> dict | torch.Tensor:
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
        result = self.env.reset(**kwargs)
        
        if isinstance(result, tuple):
            obs, info = result
            if not self._obs_manager_patched:
                obs = self._augment_observations(obs)
            return obs, info
        else:
            return self._augment_observations(result) if not self._obs_manager_patched else result
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if not self._obs_manager_patched:
            obs = self._augment_observations(obs)
        return obs, reward, terminated, truncated, info
    
    def get_force_obs_dim(self) -> int:
        return self._force_dim
    
    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)


class VecEnvForceSensorWrapper:
    
    def __init__(
        self,
        env,
        cfg: ForceSensorConfig | None = None,
    ):
        self.env = env
        self.cfg = cfg if cfg is not None else SimpleForceConfig()
        self._obs_manager_patched: bool = False
        self._orig_obs_manager_compute = None
        
        self._unwrapped = env.unwrapped if hasattr(env, 'unwrapped') else env
        
        self._force_dim = self._get_force_dim()
        
        self._update_observation_space()

        self._maybe_patch_observation_manager()
        
        print(f"[VecEnvForceSensorWrapper] Initialized with mode='{self.cfg.force_obs_mode}' "
              f"({self._force_dim} dims)")

    def _maybe_patch_observation_manager(self) -> None:
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
        mode_dims = {
            "scalar": 1,
            "per_finger": 2,
            "with_closure": 3,
            "contact_estimate": 2,
            "grasp_indicator": 3,
        }
        return mode_dims.get(self.cfg.force_obs_mode, 1)
    
    def _update_observation_space(self):
        if not self.cfg.enabled:
            self.observation_space = self.env.observation_space
            return
        
        orig_space = self.env.observation_space
        
        def _append_last_axis(arr: np.ndarray, append_dim: int, fill_value: float) -> np.ndarray:
            arr = np.asarray(arr)
            pad_shape = (*arr.shape[:-1], append_dim)
            pad = np.full(pad_shape, fill_value, dtype=np.float32)
            return np.concatenate([arr, pad], axis=-1)

        if isinstance(orig_space, spaces.Dict):
            new_spaces = {}
            for key, space in orig_space.spaces.items():
                if key == "proprio" and isinstance(space, spaces.Box):
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
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            obs, info = result
            if not self._obs_manager_patched:
                obs = self._augment_observations(obs)
            return obs, info
        return self._augment_observations(result) if not self._obs_manager_patched else result
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if not self._obs_manager_patched:
            obs = self._augment_observations(obs)
        return obs, reward, terminated, truncated, info
    
    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)
    
    @property
    def unwrapped(self):
        return self._unwrapped


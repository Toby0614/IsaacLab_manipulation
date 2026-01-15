

from __future__ import annotations

import torch
import gymnasium as gym
from typing import Any

from .modality_dropout_cfg import ModalityDropoutCfg
from .modality_dropout_manager import ModalityDropoutManager
from .phase_detector import PickPlacePhaseDetector


class DropoutEnvWrapper(gym.Wrapper):
    
    def __init__(
        self,
        env: gym.Env,
        dropout_cfg: ModalityDropoutCfg,
        enable_phase_detection: bool = False,
        phase_detector_kwargs: dict | None = None,
    ):
        super().__init__(env)
        
        self.dropout_cfg = dropout_cfg
        self.enable_phase_detection = enable_phase_detection or dropout_cfg.phase_aware
        
        self.dropout_manager = ModalityDropoutManager(
            cfg=dropout_cfg,
            num_envs=self.env.unwrapped.num_envs,
            device=str(self.env.unwrapped.device),
        )
        
        self.env.unwrapped.dropout_manager = self.dropout_manager
        
        if self.enable_phase_detection:
            phase_kwargs = phase_detector_kwargs or {}
            self.phase_detector = PickPlacePhaseDetector(**phase_kwargs)
        else:
            self.phase_detector = None
        
        print(f"[DropoutEnvWrapper] Initialized with mode={dropout_cfg.dropout_mode}, "
              f"enabled={dropout_cfg.enabled}, phase_aware={dropout_cfg.phase_aware}")
    
    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        
        all_env_ids = torch.arange(self.env.unwrapped.num_envs, device=self.env.unwrapped.device)
        self.dropout_manager.reset(all_env_ids)
        
        return result
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        if self.enable_phase_detection and self.phase_detector is not None:
            try:
                phases = self.phase_detector.detect_phases(self.env.unwrapped)
                self.dropout_manager.update_phases(phases)
            except Exception as e:
                print(f"[DropoutEnvWrapper] Phase detection failed: {e}")
        
        self.dropout_manager.step()
        
        done = terminated | truncated
        reset_env_ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.dropout_manager.reset(reset_env_ids)
        
        return obs, reward, terminated, truncated, info
    
    def get_dropout_stats(self) -> dict:
        return self.dropout_manager.get_stats()


class VecEnvDropoutWrapper:
    
    def __init__(
        self,
        env,
        dropout_cfg: ModalityDropoutCfg,
        enable_phase_detection: bool = False,
        phase_detector_kwargs: dict | None = None,
    ):
        self.env = env
        self.dropout_cfg = dropout_cfg
        self.enable_phase_detection = enable_phase_detection or dropout_cfg.phase_aware
        
        if hasattr(env, 'unwrapped'):
            base_env = env.unwrapped
        else:
            base_env = env
        
        self.dropout_manager = ModalityDropoutManager(
            cfg=dropout_cfg,
            num_envs=base_env.num_envs,
            device=str(base_env.device),
        )
        
        base_env.dropout_manager = self.dropout_manager
        
        if self.enable_phase_detection:
            phase_kwargs = phase_detector_kwargs or {}
            self.phase_detector = PickPlacePhaseDetector(**phase_kwargs)
        else:
            self.phase_detector = None
        
        print(f"[VecEnvDropoutWrapper] Initialized with mode={dropout_cfg.dropout_mode}, "
              f"enabled={dropout_cfg.enabled}")
    
    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        
        base_env = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
        all_env_ids = torch.arange(base_env.num_envs, device=base_env.device)
        self.dropout_manager.reset(all_env_ids)
        
        return result
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        if self.enable_phase_detection and self.phase_detector is not None:
            try:
                base_env = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
                phases = self.phase_detector.detect_phases(base_env)
                self.dropout_manager.update_phases(phases)
            except Exception:
                pass
        
        self.dropout_manager.step()
        
        base_env = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
        done = terminated | truncated
        reset_env_ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.dropout_manager.reset(reset_env_ids)
        
        return obs, reward, terminated, truncated, info
    
    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)
    
    def get_dropout_stats(self) -> dict:
        return self.dropout_manager.get_stats()


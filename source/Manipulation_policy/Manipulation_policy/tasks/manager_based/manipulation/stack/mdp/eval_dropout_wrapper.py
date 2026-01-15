

from __future__ import annotations

import torch
import gymnasium as gym
from typing import Any

from .eval_dropout_cfg import Variant1PhaseDropoutCfg, Variant2TimeDropoutCfg
from .phase_detector import PickPlacePhaseDetector


class EvalDropoutManagerBase:
    
    def __init__(
        self,
        num_envs: int,
        device: str,
        dropout_mode: str = "hard",
        dropout_rgb: bool = True,
        dropout_depth: bool = True,
        hard_dropout_value_rgb: float = 0.0,
        hard_dropout_value_depth: float = 0.0,
    ):
        self.num_envs = num_envs
        self.device = device
        self.dropout_mode = dropout_mode
        self.dropout_rgb = dropout_rgb
        self.dropout_depth = dropout_depth
        self.hard_dropout_value_rgb = hard_dropout_value_rgb
        self.hard_dropout_value_depth = hard_dropout_value_depth
        
        self.dropout_active = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.dropout_remaining_steps = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self.episode_step_count = torch.zeros(num_envs, dtype=torch.int32, device=device)
        
        self.dropout_triggered_count = torch.zeros(num_envs, dtype=torch.int32, device=device)
    
    def reset(self, env_ids: torch.Tensor):
        if env_ids is None or len(env_ids) == 0:
            return
        
        self.dropout_active[env_ids] = False
        self.dropout_remaining_steps[env_ids] = 0
        self.episode_step_count[env_ids] = 0
        self.dropout_triggered_count[env_ids] = 0
    
    def step(self):
        self.episode_step_count += 1
        
        self.dropout_remaining_steps = torch.clamp(self.dropout_remaining_steps - 1, min=0)
        
        expired = self.dropout_active & (self.dropout_remaining_steps <= 0)
        self.dropout_active[expired] = False
    
    def apply_dropout(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.dropout_active.any():
            return rgb, depth
        
        rgb_out = rgb.clone()
        depth_out = depth.clone()
        
        dropout_mask = self.dropout_active
        
        if self.dropout_mode == "hard":
            rgb_out, depth_out = self._apply_hard_dropout(rgb_out, depth_out, dropout_mask)
        
        return rgb_out, depth_out
    
    def _apply_hard_dropout(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        broadcast_mask = mask.view(-1, 1, 1, 1)
        
        if self.dropout_rgb:
            if rgb.dtype == torch.uint8:
                dropout_val = torch.tensor(self.hard_dropout_value_rgb, dtype=torch.uint8, device=self.device)
            else:
                dropout_val = torch.tensor(self.hard_dropout_value_rgb / 255.0, dtype=rgb.dtype, device=self.device)
            rgb = torch.where(broadcast_mask, dropout_val, rgb)
        
        if self.dropout_depth:
            dropout_val = torch.tensor(self.hard_dropout_value_depth, dtype=depth.dtype, device=self.device)
            depth = torch.where(broadcast_mask, dropout_val, depth)
        
        return rgb, depth
    
    def get_stats(self) -> dict:
        return {
            "dropout_active_count": self.dropout_active.sum().item(),
            "dropout_triggered_total": self.dropout_triggered_count.sum().item(),
        }


class Variant1PhaseDropoutManager(EvalDropoutManagerBase):
    
    def __init__(
        self,
        cfg: Variant1PhaseDropoutCfg,
        num_envs: int,
        device: str,
    ):
        super().__init__(
            num_envs=num_envs,
            device=device,
            dropout_mode=cfg.dropout_mode,
            dropout_rgb=cfg.dropout_rgb,
            dropout_depth=cfg.dropout_depth,
            hard_dropout_value_rgb=cfg.hard_dropout_value_rgb,
            hard_dropout_value_depth=cfg.hard_dropout_value_depth,
        )
        
        self.cfg = cfg
        self.target_phase = cfg.target_phase
        self.dropout_duration = cfg.dropout_duration_steps
        self.trigger_once = cfg.trigger_once_per_episode
        self.phase_entry_delay = cfg.phase_entry_delay
        self.require_stable = cfg.require_stable_phase
        self.stable_steps_required = cfg.stable_phase_steps
        
        self.current_phase = ["reach"] * num_envs
        self.previous_phase = ["reach"] * num_envs
        self.phase_stable_steps = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self.dropout_triggered_this_episode = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.phase_entry_step = torch.zeros(num_envs, dtype=torch.int32, device=device)
    
    def reset(self, env_ids: torch.Tensor):
        super().reset(env_ids)
        
        if env_ids is None or len(env_ids) == 0:
            return
        
        for idx in env_ids.cpu().tolist():
            self.current_phase[idx] = "reach"
            self.previous_phase[idx] = "reach"
        
        self.phase_stable_steps[env_ids] = 0
        self.dropout_triggered_this_episode[env_ids] = False
        self.phase_entry_step[env_ids] = 0
    
    def update_phases(self, phases: list[str]):
        assert len(phases) == self.num_envs
        
        for i, phase in enumerate(phases):
            if phase != self.current_phase[i]:
                self.previous_phase[i] = self.current_phase[i]
                self.current_phase[i] = phase
                self.phase_stable_steps[i] = 0
                self.phase_entry_step[i] = self.episode_step_count[i].item()
            else:
                self.phase_stable_steps[i] += 1
    
    def step(self):
        super().step()
        
        for i in range(self.num_envs):
            if self.dropout_active[i]:
                continue
            
            if self.trigger_once and self.dropout_triggered_this_episode[i]:
                continue
            
            if self.current_phase[i] != self.target_phase:
                continue
            
            if self.require_stable and self.phase_stable_steps[i] < self.stable_steps_required:
                continue
            
            steps_in_phase = self.episode_step_count[i].item() - self.phase_entry_step[i].item()
            if steps_in_phase < self.phase_entry_delay:
                continue
            
            self.dropout_active[i] = True
            self.dropout_remaining_steps[i] = self.dropout_duration
            self.dropout_triggered_this_episode[i] = True
            self.dropout_triggered_count[i] += 1


class Variant2TimeDropoutManager(EvalDropoutManagerBase):
    
    def __init__(
        self,
        cfg: Variant2TimeDropoutCfg,
        num_envs: int,
        device: str,
    ):
        super().__init__(
            num_envs=num_envs,
            device=device,
            dropout_mode=cfg.dropout_mode,
            dropout_rgb=cfg.dropout_rgb,
            dropout_depth=cfg.dropout_depth,
            hard_dropout_value_rgb=cfg.hard_dropout_value_rgb,
            hard_dropout_value_depth=cfg.hard_dropout_value_depth,
        )
        
        self.cfg = cfg
        self.onset_step = cfg.onset_step
        self.dropout_duration = cfg.dropout_duration_steps
        
        self.dropout_triggered_this_episode = torch.zeros(num_envs, dtype=torch.bool, device=device)
    
    def reset(self, env_ids: torch.Tensor):
        super().reset(env_ids)
        
        if env_ids is None or len(env_ids) == 0:
            return
        
        self.dropout_triggered_this_episode[env_ids] = False
    
    def step(self):
        super().step()
        
        should_trigger = (
            (self.episode_step_count == self.onset_step) &
            (~self.dropout_triggered_this_episode) &
            (~self.dropout_active)
        )
        
        if should_trigger.any():
            self.dropout_active[should_trigger] = True
            self.dropout_remaining_steps[should_trigger] = self.dropout_duration
            self.dropout_triggered_this_episode[should_trigger] = True
            self.dropout_triggered_count[should_trigger] += 1


class Variant1EvalWrapper(gym.Wrapper):
    
    def __init__(
        self,
        env: gym.Env,
        cfg: Variant1PhaseDropoutCfg,
        phase_detector_kwargs: dict | None = None,
    ):
        super().__init__(env)
        
        self.cfg = cfg
        
        self.dropout_manager = Variant1PhaseDropoutManager(
            cfg=cfg,
            num_envs=self.env.unwrapped.num_envs,
            device=str(self.env.unwrapped.device),
        )
        
        self.env.unwrapped.dropout_manager = self.dropout_manager
        
        phase_kwargs = phase_detector_kwargs or {}
        self.phase_detector = PickPlacePhaseDetector(**phase_kwargs)
        
        print(f"[Variant1EvalWrapper] Phase={cfg.target_phase}, Duration={cfg.dropout_duration_steps} steps")
    
    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        
        all_env_ids = torch.arange(self.env.unwrapped.num_envs, device=self.env.unwrapped.device)
        self.dropout_manager.reset(all_env_ids)
        
        return result
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        try:
            phases = self.phase_detector.detect_phases(self.env.unwrapped)
            self.dropout_manager.update_phases(phases)
        except Exception as e:
            pass  # Continue without phase update on error
        
        self.dropout_manager.step()
        
        done = terminated | truncated
        reset_env_ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.dropout_manager.reset(reset_env_ids)
        
        return obs, reward, terminated, truncated, info
    
    def get_dropout_stats(self) -> dict:
        return self.dropout_manager.get_stats()


class Variant2EvalWrapper(gym.Wrapper):
    
    def __init__(
        self,
        env: gym.Env,
        cfg: Variant2TimeDropoutCfg,
    ):
        super().__init__(env)
        
        self.cfg = cfg
        
        self.dropout_manager = Variant2TimeDropoutManager(
            cfg=cfg,
            num_envs=self.env.unwrapped.num_envs,
            device=str(self.env.unwrapped.device),
        )
        
        self.env.unwrapped.dropout_manager = self.dropout_manager
        
        print(f"[Variant2EvalWrapper] Onset={cfg.onset_step}, Duration={cfg.dropout_duration_steps} steps")
    
    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        
        all_env_ids = torch.arange(self.env.unwrapped.num_envs, device=self.env.unwrapped.device)
        self.dropout_manager.reset(all_env_ids)
        
        return result
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.dropout_manager.step()
        
        done = terminated | truncated
        reset_env_ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.dropout_manager.reset(reset_env_ids)
        
        return obs, reward, terminated, truncated, info
    
    def get_dropout_stats(self) -> dict:
        return self.dropout_manager.get_stats()



class VecEnvVariant1EvalWrapper:
    
    def __init__(
        self,
        env,
        cfg: Variant1PhaseDropoutCfg,
        phase_detector_kwargs: dict | None = None,
    ):
        self.env = env
        self.cfg = cfg
        
        base_env = env.unwrapped if hasattr(env, 'unwrapped') else env
        
        self.dropout_manager = Variant1PhaseDropoutManager(
            cfg=cfg,
            num_envs=base_env.num_envs,
            device=str(base_env.device),
        )
        
        base_env.dropout_manager = self.dropout_manager
        
        phase_kwargs = phase_detector_kwargs or {}
        self.phase_detector = PickPlacePhaseDetector(**phase_kwargs)
        
        print(f"[VecEnvVariant1EvalWrapper] Phase={cfg.target_phase}, Duration={cfg.dropout_duration_steps}")
    
    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        
        base_env = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
        all_env_ids = torch.arange(base_env.num_envs, device=base_env.device)
        self.dropout_manager.reset(all_env_ids)
        
        return result
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        base_env = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
        
        try:
            phases = self.phase_detector.detect_phases(base_env)
            self.dropout_manager.update_phases(phases)
        except Exception:
            pass
        
        self.dropout_manager.step()
        
        done = terminated | truncated
        reset_env_ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.dropout_manager.reset(reset_env_ids)
        
        return obs, reward, terminated, truncated, info
    
    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)
    
    @property
    def unwrapped(self):
        return self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env


class VecEnvVariant2EvalWrapper:
    
    def __init__(
        self,
        env,
        cfg: Variant2TimeDropoutCfg,
    ):
        self.env = env
        self.cfg = cfg
        
        base_env = env.unwrapped if hasattr(env, 'unwrapped') else env
        
        self.dropout_manager = Variant2TimeDropoutManager(
            cfg=cfg,
            num_envs=base_env.num_envs,
            device=str(base_env.device),
        )
        
        base_env.dropout_manager = self.dropout_manager
        
        print(f"[VecEnvVariant2EvalWrapper] Onset={cfg.onset_step}, Duration={cfg.dropout_duration_steps}")
    
    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        
        base_env = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
        all_env_ids = torch.arange(base_env.num_envs, device=base_env.device)
        self.dropout_manager.reset(all_env_ids)
        
        return result
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.dropout_manager.step()
        
        base_env = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
        done = terminated | truncated
        reset_env_ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.dropout_manager.reset(reset_env_ids)
        
        return obs, reward, terminated, truncated, info
    
    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)
    
    @property
    def unwrapped(self):
        return self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env


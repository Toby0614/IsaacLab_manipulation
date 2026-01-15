

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

from .modality_dropout_cfg import ModalityDropoutCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class ModalityDropoutManager:
    
    def __init__(
        self,
        cfg: ModalityDropoutCfg,
        num_envs: int,
        device: str,
    ):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        
        self.dropout_active = torch.zeros(num_envs, dtype=torch.bool, device=device)
        
        self.dropout_remaining_steps = torch.zeros(num_envs, dtype=torch.int32, device=device)
        
        self.episode_step_count = torch.zeros(num_envs, dtype=torch.int32, device=device)
        
        self.current_phase = ["reach"] * num_envs  # Default phase
        
    def reset(self, env_ids: torch.Tensor):
        if env_ids is None or len(env_ids) == 0:
            return
            
        self.dropout_active[env_ids] = False
        self.dropout_remaining_steps[env_ids] = 0
        self.episode_step_count[env_ids] = 0
        
        for idx in env_ids.cpu().tolist():
            self.current_phase[idx] = "reach"
    
    def update_phase(self, env_id: int, phase: str):
        if env_id < len(self.current_phase):
            self.current_phase[env_id] = phase
    
    def update_phases(self, phases: list[str]):
        assert len(phases) == self.num_envs, f"Expected {self.num_envs} phases, got {len(phases)}"
        self.current_phase = phases
    
    def step(self):
        if not self.cfg.enabled:
            return
        
        self.episode_step_count += 1
        
        if self.cfg.eval_mode:
            in_dropout_window = (
                (self.episode_step_count >= self.cfg.eval_dropout_start_step) &
                (self.episode_step_count < self.cfg.eval_dropout_start_step + self.cfg.eval_dropout_duration)
            )
            self.dropout_active = in_dropout_window
            return
        
        
        self.dropout_remaining_steps = torch.clamp(self.dropout_remaining_steps - 1, min=0)
        
        expired = (self.dropout_active) & (self.dropout_remaining_steps <= 0)
        self.dropout_active[expired] = False
        
        inactive_envs = ~self.dropout_active
        
        if self.cfg.phase_aware:
            dropout_probs = torch.tensor(
                [self.cfg.phase_dropout_config.get(self.current_phase[i], self.cfg.dropout_probability)
                 for i in range(self.num_envs)],
                device=self.device,
                dtype=torch.float32
            )
        else:
            dropout_probs = torch.full((self.num_envs,), self.cfg.dropout_probability, device=self.device)
        
        new_dropout = inactive_envs & (torch.rand(self.num_envs, device=self.device) < dropout_probs)
        
        if new_dropout.any():
            num_new = new_dropout.sum().item()
            min_dur, max_dur = self.cfg.dropout_duration_range
            durations = torch.randint(min_dur, max_dur + 1, (num_new,), device=self.device, dtype=torch.int32)
            
            self.dropout_active[new_dropout] = True
            self.dropout_remaining_steps[new_dropout] = durations
    
    def apply_dropout(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.cfg.enabled:
            return rgb, depth
        
        dropout_mask = self.dropout_active  # Shape: (B,)
        
        if not dropout_mask.any():
            return rgb, depth
        
        rgb_out = rgb.clone()
        depth_out = depth.clone()
        
        rgb_is_chw = (rgb.ndim == 4 and rgb.shape[1] in [1, 3, 4])
        depth_is_chw = (depth.ndim == 4 and depth.shape[1] == 1)
        
        if self.cfg.dropout_mode == "hard":
            rgb_out, depth_out = self._apply_hard_dropout(rgb_out, depth_out, dropout_mask, rgb_is_chw, depth_is_chw)
        elif self.cfg.dropout_mode == "soft":
            rgb_out, depth_out = self._apply_soft_dropout(rgb_out, depth_out, dropout_mask, rgb_is_chw, depth_is_chw)
        elif self.cfg.dropout_mode == "mixed":
            hard_mask = dropout_mask & (torch.rand(self.num_envs, device=self.device) < 0.5)
            soft_mask = dropout_mask & ~hard_mask
            
            if hard_mask.any():
                rgb_out, depth_out = self._apply_hard_dropout(rgb_out, depth_out, hard_mask, rgb_is_chw, depth_is_chw)
            if soft_mask.any():
                rgb_out, depth_out = self._apply_soft_dropout(rgb_out, depth_out, soft_mask, rgb_is_chw, depth_is_chw)
        
        return rgb_out, depth_out
    
    def _apply_hard_dropout(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        mask: torch.Tensor,
        rgb_is_chw: bool,
        depth_is_chw: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if rgb_is_chw:
            rgb_mask = mask.view(-1, 1, 1, 1)  # (B, 1, 1, 1)
        else:
            rgb_mask = mask.view(-1, 1, 1, 1)  # (B, 1, 1, 1)
        
        if depth_is_chw:
            depth_mask = mask.view(-1, 1, 1, 1)
        else:
            depth_mask = mask.view(-1, 1, 1, 1)
        
        if self.cfg.dropout_rgb:
            dropout_value = self.cfg.hard_dropout_value_rgb
            if rgb.dtype == torch.uint8:
                dropout_val_tensor = torch.tensor(dropout_value, dtype=torch.uint8, device=self.device)
            else:
                dropout_val_tensor = torch.tensor(dropout_value / 255.0, dtype=rgb.dtype, device=self.device)
            
            rgb = torch.where(rgb_mask, dropout_val_tensor, rgb)
        
        if self.cfg.dropout_depth:
            dropout_val_tensor = torch.tensor(self.cfg.hard_dropout_value_depth, dtype=depth.dtype, device=self.device)
            depth = torch.where(depth_mask, dropout_val_tensor, depth)
        
        return rgb, depth
    
    def _apply_soft_dropout(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        mask: torch.Tensor,
        rgb_is_chw: bool,
        depth_is_chw: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        if self.cfg.dropout_rgb and mask.any():
            rgb_float = rgb.float()
            is_uint8 = (rgb.dtype == torch.uint8)
            
            if self.cfg.gaussian_noise_std_rgb > 0:
                noise = torch.randn_like(rgb_float) * self.cfg.gaussian_noise_std_rgb
                if rgb_is_chw:
                    noise_mask = mask.view(-1, 1, 1, 1)
                else:
                    noise_mask = mask.view(-1, 1, 1, 1)
                rgb_float = rgb_float + torch.where(noise_mask, noise, torch.zeros_like(noise))
            
            if is_uint8:
                rgb_float = torch.clamp(rgb_float, 0.0, 255.0)
                rgb = rgb_float.to(torch.uint8)
            else:
                rgb_float = torch.clamp(rgb_float, 0.0, 1.0)
                rgb = rgb_float
            
            if self.cfg.cutout_prob > 0:
                rgb = self._apply_cutout(rgb, mask, rgb_is_chw)
        
        if self.cfg.dropout_depth and mask.any():
            if self.cfg.gaussian_noise_std_depth > 0:
                noise = torch.randn_like(depth) * self.cfg.gaussian_noise_std_depth
                if depth_is_chw:
                    noise_mask = mask.view(-1, 1, 1, 1)
                else:
                    noise_mask = mask.view(-1, 1, 1, 1)
                depth = depth + torch.where(noise_mask, noise, torch.zeros_like(noise))
                depth = torch.clamp(depth, 0.0, 1.0)
            
            if self.cfg.depth_speckle_prob > 0:
                depth = self._apply_speckle(depth, mask, depth_is_chw)
        
        return rgb, depth
    
    def _apply_cutout(self, img: torch.Tensor, mask: torch.Tensor, is_chw: bool) -> torch.Tensor:
        if is_chw:
            b, c, h, w = img.shape
        else:
            b, h, w, c = img.shape
        
        cut_h, cut_w = self.cfg.cutout_size
        cut_h = max(1, min(cut_h, h))
        cut_w = max(1, min(cut_w, w))
        
        for env_idx in torch.where(mask)[0].tolist():
            if torch.rand(1, device=self.device).item() < self.cfg.cutout_prob:
                y0 = torch.randint(0, h - cut_h + 1, (1,), device=self.device).item()
                x0 = torch.randint(0, w - cut_w + 1, (1,), device=self.device).item()
                
                if is_chw:
                    img[env_idx, :, y0:y0+cut_h, x0:x0+cut_w] = 0
                else:
                    img[env_idx, y0:y0+cut_h, x0:x0+cut_w, :] = 0
        
        return img
    
    def _apply_speckle(self, depth: torch.Tensor, mask: torch.Tensor, is_chw: bool) -> torch.Tensor:
        if is_chw:
            b, c, h, w = depth.shape
            speckle_shape = (b, 1, h, w)
        else:
            b, h, w, c = depth.shape
            speckle_shape = (b, h, w, 1)
        
        speckle_mask_rand = torch.rand(speckle_shape, device=self.device) < self.cfg.depth_speckle_prob
        
        env_mask = mask.view(-1, 1, 1, 1) if is_chw else mask.view(-1, 1, 1, 1)
        speckle_mask_final = speckle_mask_rand & env_mask
        
        depth = torch.where(speckle_mask_final, torch.zeros_like(depth), depth)
        return depth
    
    def get_dropout_indicator(self) -> torch.Tensor:
        return self.dropout_active.float().unsqueeze(-1)
    
    def get_stats(self) -> dict:
        return {
            "dropout_active_count": self.dropout_active.sum().item(),
            "dropout_active_fraction": self.dropout_active.float().mean().item(),
            "avg_remaining_steps": self.dropout_remaining_steps[self.dropout_active].float().mean().item()
                if self.dropout_active.any() else 0.0,
        }


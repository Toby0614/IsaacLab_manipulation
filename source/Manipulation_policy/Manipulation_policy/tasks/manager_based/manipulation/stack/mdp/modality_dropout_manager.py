# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Modality Dropout Manager for vision-based RL robustness studies.

Implements duration-based, phase-aware vision failure simulation as described in poe2.pdf.

Key features:
- Hard dropout: complete vision blackout
- Soft dropout: noise/corruption
- Duration control: continuous failures over multiple steps
- Phase-aware: different dropout rates per manipulation phase
- Per-environment dropout tracking
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

from .modality_dropout_cfg import ModalityDropoutCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class ModalityDropoutManager:
    """Manages modality dropout state and scheduling for vision-based RL environments.
    
    This class maintains per-environment dropout state and determines when/how to apply
    vision failures based on the configuration.
    
    Usage:
        # In environment __init__:
        from .mdp.modality_dropout_cfg import HardDropoutTrainingCfg
        from .mdp.modality_dropout_manager import ModalityDropoutManager
        
        dropout_cfg = HardDropoutTrainingCfg()
        self.dropout_manager = ModalityDropoutManager(cfg=dropout_cfg, num_envs=self.num_envs, device=self.device)
        
        # In observation function:
        rgb, depth = get_camera_data(...)
        if hasattr(env, 'dropout_manager'):
            rgb, depth = env.dropout_manager.apply_dropout(env, rgb, depth, env_step_count)
    """
    
    def __init__(
        self,
        cfg: ModalityDropoutCfg,
        num_envs: int,
        device: str,
    ):
        """Initialize the dropout manager.
        
        Args:
            cfg: Dropout configuration
            num_envs: Number of parallel environments
            device: Torch device (e.g., "cuda" or "cpu")
        """
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        
        # Per-environment dropout state
        # dropout_active[i] = True if env i is currently experiencing dropout
        self.dropout_active = torch.zeros(num_envs, dtype=torch.bool, device=device)
        
        # Remaining steps of current dropout event per environment
        self.dropout_remaining_steps = torch.zeros(num_envs, dtype=torch.int32, device=device)
        
        # Episode step counter (for eval mode deterministic schedule)
        self.episode_step_count = torch.zeros(num_envs, dtype=torch.int32, device=device)
        
        # Track current phase per environment (if phase_aware enabled)
        self.current_phase = ["reach"] * num_envs  # Default phase
        
    def reset(self, env_ids: torch.Tensor):
        """Reset dropout state for specified environments.
        
        Call this at episode reset.
        
        Args:
            env_ids: Tensor of environment indices to reset
        """
        if env_ids is None or len(env_ids) == 0:
            return
            
        self.dropout_active[env_ids] = False
        self.dropout_remaining_steps[env_ids] = 0
        self.episode_step_count[env_ids] = 0
        
        # Reset phase tracking
        for idx in env_ids.cpu().tolist():
            self.current_phase[idx] = "reach"
    
    def update_phase(self, env_id: int, phase: str):
        """Update the current manipulation phase for an environment.
        
        Args:
            env_id: Environment index
            phase: Phase name (e.g., "reach", "grasp", "lift", "transport", "place")
        """
        if env_id < len(self.current_phase):
            self.current_phase[env_id] = phase
    
    def update_phases(self, phases: list[str]):
        """Update phases for all environments.
        
        Args:
            phases: List of phase names, one per environment
        """
        assert len(phases) == self.num_envs, f"Expected {self.num_envs} phases, got {len(phases)}"
        self.current_phase = phases
    
    def step(self):
        """Update dropout state for all environments (call once per simulation step).
        
        This handles:
        - Decrementing remaining dropout duration
        - Starting new dropout events based on probability
        - Phase-aware dropout probability modulation
        """
        if not self.cfg.enabled:
            return
        
        # Increment episode step counter
        self.episode_step_count += 1
        
        # === Evaluation mode: deterministic dropout ===
        if self.cfg.eval_mode:
            # Start dropout at specific step, end after duration
            in_dropout_window = (
                (self.episode_step_count >= self.cfg.eval_dropout_start_step) &
                (self.episode_step_count < self.cfg.eval_dropout_start_step + self.cfg.eval_dropout_duration)
            )
            self.dropout_active = in_dropout_window
            return
        
        # === Training mode: stochastic dropout ===
        
        # Decrement remaining steps for active dropouts
        self.dropout_remaining_steps = torch.clamp(self.dropout_remaining_steps - 1, min=0)
        
        # Deactivate dropout when duration expires
        expired = (self.dropout_active) & (self.dropout_remaining_steps <= 0)
        self.dropout_active[expired] = False
        
        # Start new dropout events (only for envs not currently dropping out)
        inactive_envs = ~self.dropout_active
        
        # Get dropout probability (phase-aware if enabled)
        if self.cfg.phase_aware:
            dropout_probs = torch.tensor(
                [self.cfg.phase_dropout_config.get(self.current_phase[i], self.cfg.dropout_probability)
                 for i in range(self.num_envs)],
                device=self.device,
                dtype=torch.float32
            )
        else:
            dropout_probs = torch.full((self.num_envs,), self.cfg.dropout_probability, device=self.device)
        
        # Sample new dropout events
        new_dropout = inactive_envs & (torch.rand(self.num_envs, device=self.device) < dropout_probs)
        
        if new_dropout.any():
            # Sample dropout durations for new events
            num_new = new_dropout.sum().item()
            min_dur, max_dur = self.cfg.dropout_duration_range
            durations = torch.randint(min_dur, max_dur + 1, (num_new,), device=self.device, dtype=torch.int32)
            
            # Activate dropout
            self.dropout_active[new_dropout] = True
            self.dropout_remaining_steps[new_dropout] = durations
    
    def apply_dropout(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply dropout/corruption to RGB and depth images.
        
        Args:
            rgb: RGB images, shape (B, H, W, 3) or (B, 3, H, W), dtype uint8 or float
            depth: Depth images, shape (B, H, W, 1) or (B, 1, H, W), dtype float
            
        Returns:
            Corrupted (rgb, depth) tensors with same shape/dtype as input
        """
        if not self.cfg.enabled:
            return rgb, depth
        
        # Identify which environments need dropout applied
        dropout_mask = self.dropout_active  # Shape: (B,)
        
        if not dropout_mask.any():
            return rgb, depth
        
        # Clone to avoid in-place modification
        rgb_out = rgb.clone()
        depth_out = depth.clone()
        
        # Detect if images are channel-first or channel-last
        # Assume: (B, C, H, W) or (B, H, W, C)
        rgb_is_chw = (rgb.ndim == 4 and rgb.shape[1] in [1, 3, 4])
        depth_is_chw = (depth.ndim == 4 and depth.shape[1] == 1)
        
        # === Apply dropout based on mode ===
        if self.cfg.dropout_mode == "hard":
            rgb_out, depth_out = self._apply_hard_dropout(rgb_out, depth_out, dropout_mask, rgb_is_chw, depth_is_chw)
        elif self.cfg.dropout_mode == "soft":
            rgb_out, depth_out = self._apply_soft_dropout(rgb_out, depth_out, dropout_mask, rgb_is_chw, depth_is_chw)
        elif self.cfg.dropout_mode == "mixed":
            # Randomly choose hard or soft per affected environment
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
        """Apply hard dropout (complete blackout) to masked environments."""
        # Reshape mask for broadcasting
        if rgb_is_chw:
            rgb_mask = mask.view(-1, 1, 1, 1)  # (B, 1, 1, 1)
        else:
            rgb_mask = mask.view(-1, 1, 1, 1)  # (B, 1, 1, 1)
        
        if depth_is_chw:
            depth_mask = mask.view(-1, 1, 1, 1)
        else:
            depth_mask = mask.view(-1, 1, 1, 1)
        
        # Apply RGB dropout
        if self.cfg.dropout_rgb:
            dropout_value = self.cfg.hard_dropout_value_rgb
            # Handle both uint8 and float RGB
            if rgb.dtype == torch.uint8:
                dropout_val_tensor = torch.tensor(dropout_value, dtype=torch.uint8, device=self.device)
            else:
                dropout_val_tensor = torch.tensor(dropout_value / 255.0, dtype=rgb.dtype, device=self.device)
            
            rgb = torch.where(rgb_mask, dropout_val_tensor, rgb)
        
        # Apply depth dropout
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
        """Apply soft dropout (noise/corruption) to masked environments."""
        
        # === RGB corruption ===
        if self.cfg.dropout_rgb and mask.any():
            # Convert to float for noise addition
            rgb_float = rgb.float()
            is_uint8 = (rgb.dtype == torch.uint8)
            
            # Additive Gaussian noise
            if self.cfg.gaussian_noise_std_rgb > 0:
                noise = torch.randn_like(rgb_float) * self.cfg.gaussian_noise_std_rgb
                # Apply only to masked envs
                if rgb_is_chw:
                    noise_mask = mask.view(-1, 1, 1, 1)
                else:
                    noise_mask = mask.view(-1, 1, 1, 1)
                rgb_float = rgb_float + torch.where(noise_mask, noise, torch.zeros_like(noise))
            
            # Clamp and convert back
            if is_uint8:
                rgb_float = torch.clamp(rgb_float, 0.0, 255.0)
                rgb = rgb_float.to(torch.uint8)
            else:
                rgb_float = torch.clamp(rgb_float, 0.0, 1.0)
                rgb = rgb_float
            
            # Cutout (only for masked envs)
            if self.cfg.cutout_prob > 0:
                rgb = self._apply_cutout(rgb, mask, rgb_is_chw)
        
        # === Depth corruption ===
        if self.cfg.dropout_depth and mask.any():
            # Additive Gaussian noise
            if self.cfg.gaussian_noise_std_depth > 0:
                noise = torch.randn_like(depth) * self.cfg.gaussian_noise_std_depth
                if depth_is_chw:
                    noise_mask = mask.view(-1, 1, 1, 1)
                else:
                    noise_mask = mask.view(-1, 1, 1, 1)
                depth = depth + torch.where(noise_mask, noise, torch.zeros_like(noise))
                depth = torch.clamp(depth, 0.0, 1.0)
            
            # Speckle holes
            if self.cfg.depth_speckle_prob > 0:
                depth = self._apply_speckle(depth, mask, depth_is_chw)
        
        return rgb, depth
    
    def _apply_cutout(self, img: torch.Tensor, mask: torch.Tensor, is_chw: bool) -> torch.Tensor:
        """Apply random rectangular cutout to images from masked environments."""
        # Determine image dimensions
        if is_chw:
            b, c, h, w = img.shape
        else:
            b, h, w, c = img.shape
        
        cut_h, cut_w = self.cfg.cutout_size
        cut_h = max(1, min(cut_h, h))
        cut_w = max(1, min(cut_w, w))
        
        # Apply cutout to each masked env
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
        """Apply random speckle holes to depth from masked environments."""
        # Generate speckle pattern
        if is_chw:
            b, c, h, w = depth.shape
            speckle_shape = (b, 1, h, w)
        else:
            b, h, w, c = depth.shape
            speckle_shape = (b, h, w, 1)
        
        speckle_mask_rand = torch.rand(speckle_shape, device=self.device) < self.cfg.depth_speckle_prob
        
        # Apply only to envs in mask
        env_mask = mask.view(-1, 1, 1, 1) if is_chw else mask.view(-1, 1, 1, 1)
        speckle_mask_final = speckle_mask_rand & env_mask
        
        depth = torch.where(speckle_mask_final, torch.zeros_like(depth), depth)
        return depth
    
    def get_dropout_indicator(self) -> torch.Tensor:
        """Get binary indicator of current dropout state per environment.
        
        Returns:
            Tensor of shape (num_envs, 1) with 1.0 if dropout active, 0.0 otherwise
        """
        return self.dropout_active.float().unsqueeze(-1)
    
    def get_stats(self) -> dict:
        """Get statistics about dropout state (for logging/debugging).
        
        Returns:
            Dictionary with dropout statistics
        """
        return {
            "dropout_active_count": self.dropout_active.sum().item(),
            "dropout_active_fraction": self.dropout_active.float().mean().item(),
            "avg_remaining_steps": self.dropout_remaining_steps[self.dropout_active].float().mean().item()
                if self.dropout_active.any() else 0.0,
        }


"""Pose corruption manager for oracle object pose observations.

This mirrors the design of `ModalityDropoutManager` but operates on a low-dimensional pose estimate
(cube position) instead of images.
"""

from __future__ import annotations

import torch

from .pose_corruption_cfg import PoseCorruptionCfg


class PoseCorruptionManager:
    """Maintains per-env corruption event state and corrupts cube position (B,3)."""

    def __init__(self, cfg: PoseCorruptionCfg, num_envs: int, device: str):
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.device = device

        self.event_active = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self.remaining_steps = torch.zeros(self.num_envs, dtype=torch.int32, device=device)
        self.episode_step_count = torch.zeros(self.num_envs, dtype=torch.int32, device=device)
        # True for envs where an event started on the *current* step (used for correct freeze snapshot).
        self.event_just_started = torch.zeros(self.num_envs, dtype=torch.bool, device=device)

        # For freeze/delay
        self.last_valid_pose = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=device)

        # For delay: ring buffer of previous poses (max delay = cfg.delay_steps)
        self._delay_buf = None
        self._delay_idx = 0

        # For drift: persistent bias
        self._drift_bias = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=device)

        # Stats
        self.triggered_count = torch.zeros(self.num_envs, dtype=torch.int32, device=device)

        self._ensure_delay_buffer()

    def _ensure_delay_buffer(self):
        max_delay = max(int(getattr(self.cfg, "delay_steps", 0) or 0), 0)
        max_delay = max(max_delay, 1)
        self._delay_buf = torch.zeros(max_delay, self.num_envs, 3, dtype=torch.float32, device=self.device)
        self._delay_idx = 0

    def reset(self, env_ids: torch.Tensor):
        if env_ids is None or len(env_ids) == 0:
            return
        self.event_active[env_ids] = False
        self.remaining_steps[env_ids] = 0
        self.episode_step_count[env_ids] = 0
        self.event_just_started[env_ids] = False
        self.triggered_count[env_ids] = 0
        self.last_valid_pose[env_ids] = 0.0
        self._drift_bias[env_ids] = 0.0

        # Delay buffer resets for those envs
        if self._delay_buf is not None:
            self._delay_buf[:, env_ids, :] = 0.0

    def step(self):
        """Advance per-episode time and update event scheduling."""
        if not self.cfg.enabled:
            self.episode_step_count += 1
            return

        # Clear one-step pulse.
        self.event_just_started[:] = False

        self.episode_step_count += 1

        # Decrement remaining steps
        self.remaining_steps = torch.clamp(self.remaining_steps - 1, min=0)

        # End events
        expired = self.event_active & (self.remaining_steps <= 0)
        self.event_active[expired] = False

        # Possibly start new events where inactive
        can_start = ~self.event_active
        if self.cfg.event_probability > 0.0 and can_start.any():
            start_mask = (torch.rand((self.num_envs,), device=self.device) < float(self.cfg.event_probability)) & can_start
            if start_mask.any():
                dmin, dmax = self.cfg.duration_range
                dmin = int(dmin)
                dmax = int(dmax)
                if dmax < dmin:
                    dmax = dmin
                # randint is [low, high)
                dur = torch.randint(low=dmin, high=dmax + 1, size=(self.num_envs,), device=self.device, dtype=torch.int32)
                self.event_active[start_mask] = True
                self.remaining_steps[start_mask] = dur[start_mask]
                self.event_just_started[start_mask] = True
                self.triggered_count[start_mask] += 1

                # Reset drift bias at event start (optional but stable)
                self._drift_bias[start_mask] = 0.0

    def _choose_mode_per_env(self) -> list[str]:
        """Return per-env corruption mode string (length=num_envs) for the *current step*."""
        if self.cfg.mode != "mixed":
            return [self.cfg.mode] * self.num_envs

        # Mixed: sample mode per env (only matters when event_active)
        modes = []
        keys = list(self.cfg.mixed_mode_probs.keys())
        probs = torch.tensor([self.cfg.mixed_mode_probs[k] for k in keys], device=self.device, dtype=torch.float32)
        probs = probs / torch.clamp(probs.sum(), min=1e-8)
        cat = torch.distributions.Categorical(probs=probs)
        samples = cat.sample((self.num_envs,))
        for i in range(self.num_envs):
            modes.append(keys[int(samples[i].item())])
        return modes

    def apply(self, pose: torch.Tensor) -> torch.Tensor:
        """Apply pose corruption to a (B,3) pose tensor.

        Assumes `pose` is in env frame and float-like.
        """
        if not self.cfg.enabled:
            return pose

        # Choose per-env modes once for this step.
        modes = self._choose_mode_per_env()

        # Figure out which envs are currently in an active "freeze" event for THIS step.
        # This is required so that "freeze" actually becomes stale for L steps:
        # we must NOT keep updating last_valid_pose while a freeze event is active.
        if self.cfg.mode == "freeze":
            freeze_active = self.event_active.clone()
        elif self.cfg.mode == "mixed":
            freeze_active = torch.tensor([m == "freeze" for m in modes], device=self.device, dtype=torch.bool) & self.event_active
        else:
            freeze_active = torch.zeros_like(self.event_active, dtype=torch.bool)

        pose_f = pose.float() if torch.is_tensor(pose) else pose
        # Always update buffers, but do it in a way that preserves freeze semantics.
        self._update_buffers(pose_f, freeze_active=freeze_active)

        if not self.event_active.any():
            return pose

        out = pose_f.clone()
        active = self.event_active

        # Drift update for active envs (only used by noise)
        if float(self.cfg.drift_noise_std) > 0.0:
            eps = torch.randn_like(self._drift_bias) * float(self.cfg.drift_noise_std)
            alpha = float(self.cfg.drift_alpha)
            self._drift_bias = alpha * self._drift_bias + (1.0 - alpha) * eps

        hard_val = torch.tensor(self.cfg.hard_value, device=self.device, dtype=torch.float32).view(1, 3)

        for i in range(self.num_envs):
            if not bool(active[i].item()):
                continue
            m = modes[i]
            if m == "hard":
                out[i : i + 1, :] = hard_val
            elif m == "freeze":
                out[i, :] = self.last_valid_pose[i, :]
            elif m == "delay":
                out[i, :] = self._get_delayed_pose(i)
            elif m == "noise":
                noise = torch.randn((3,), device=self.device, dtype=torch.float32) * float(self.cfg.noise_std)
                out[i, :] = pose_f[i, :] + noise + self._drift_bias[i, :]
            else:
                # unknown -> no-op
                out[i, :] = pose_f[i, :]

        return out.to(dtype=pose.dtype) if pose.dtype != torch.float32 else out

    def _update_buffers(self, pose: torch.Tensor, *, freeze_active: torch.Tensor):
        """Update last_valid_pose and delay ring buffer.

        Correct freeze semantics:
        - When a freeze event is active, the measurement output should be stale for L steps.
          Therefore, we must NOT update `last_valid_pose` while (freeze_active == True),
          except on the *first* step of the event (event_just_started) to snapshot the reading.
        """
        pose_d = pose.detach()
        # Update last_valid_pose for envs that are NOT in an ongoing freeze,
        # and also for envs whose freeze event just started (snapshot).
        update_mask = (~freeze_active) | self.event_just_started
        if update_mask.any():
            self.last_valid_pose[update_mask] = pose_d[update_mask]
        if self._delay_buf is not None:
            self._delay_buf[self._delay_idx, :, :] = pose_d
            self._delay_idx = (self._delay_idx + 1) % self._delay_buf.shape[0]

    def _get_delayed_pose(self, env_i: int) -> torch.Tensor:
        if self._delay_buf is None:
            return self.last_valid_pose[env_i, :]
        k = int(self.cfg.delay_steps)
        k = max(k, 1)
        # current write index is next slot; delayed pose is k steps behind last written
        idx = (self._delay_idx - k) % self._delay_buf.shape[0]
        return self._delay_buf[idx, env_i, :]

    def get_stats(self) -> dict:
        return {
            "pose_corruption_active_count": int(self.event_active.sum().item()),
            "pose_corruption_triggered_total": int(self.triggered_count.sum().item()),
        }



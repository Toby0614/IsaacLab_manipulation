"""Deterministic pose-corruption managers for evaluation (poe3.pdf plan).

Implements:
- Variant 1: phase-triggered pose corruption (phase × duration heatmap)
- Variant 2: time-triggered pose corruption (onset × duration heatmap)

This mirrors the existing vision dropout evaluation design, but targets the oracle cube position
observation (pose estimate) instead of images.
"""

from __future__ import annotations

import torch

from .pose_corruption_cfg import PoseCorruptionCfg
from .pose_corruption_manager import PoseCorruptionManager


class PhaseTriggeredPoseCorruptionManager(PoseCorruptionManager):
    """Trigger a corruption event when entering a target phase (once per episode)."""

    def __init__(
        self,
        cfg: PoseCorruptionCfg,
        *,
        target_phase: str,
        duration_steps: int,
        trigger_once_per_episode: bool = True,
        require_stable_phase: bool = True,
        stable_phase_steps: int = 3,
        phase_entry_delay: int = 0,
        num_envs: int,
        device: str,
    ):
        # Ensure stochastic start is disabled for deterministic evaluation.
        cfg = PoseCorruptionCfg(**{**cfg.__dict__})
        cfg.enabled = True
        cfg.event_probability = 0.0
        super().__init__(cfg=cfg, num_envs=num_envs, device=device)

        self.target_phase = target_phase
        self.duration_steps = int(duration_steps)
        self.trigger_once = bool(trigger_once_per_episode)
        self.require_stable = bool(require_stable_phase)
        self.stable_steps_required = int(stable_phase_steps)
        self.phase_entry_delay = int(phase_entry_delay)

        self.current_phase = ["reach"] * self.num_envs
        self.previous_phase = ["reach"] * self.num_envs
        self.phase_stable_steps = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.phase_entry_step = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.triggered_this_episode = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: torch.Tensor):
        super().reset(env_ids)
        if env_ids is None or len(env_ids) == 0:
            return
        for idx in env_ids.cpu().tolist():
            self.current_phase[idx] = "reach"
            self.previous_phase[idx] = "reach"
        self.phase_stable_steps[env_ids] = 0
        self.phase_entry_step[env_ids] = 0
        self.triggered_this_episode[env_ids] = False

    def update_phases(self, phases: list[str]) -> None:
        assert len(phases) == self.num_envs
        for i, phase in enumerate(phases):
            if phase != self.current_phase[i]:
                self.previous_phase[i] = self.current_phase[i]
                self.current_phase[i] = phase
                self.phase_stable_steps[i] = 0
                self.phase_entry_step[i] = int(self.episode_step_count[i].item())
            else:
                self.phase_stable_steps[i] += 1

    def step(self):
        # base step handles counters, decrements, event expiry
        super().step()

        # Trigger logic
        for i in range(self.num_envs):
            if bool(self.event_active[i].item()):
                continue
            if self.trigger_once and bool(self.triggered_this_episode[i].item()):
                continue
            if self.current_phase[i] != self.target_phase:
                continue
            if self.require_stable and int(self.phase_stable_steps[i].item()) < self.stable_steps_required:
                continue
            steps_in_phase = int(self.episode_step_count[i].item()) - int(self.phase_entry_step[i].item())
            if steps_in_phase < self.phase_entry_delay:
                continue

            # Trigger deterministic event
            self.event_active[i] = True
            self.remaining_steps[i] = int(self.duration_steps)
            self.triggered_this_episode[i] = True
            self.triggered_count[i] += 1
            self._drift_bias[i] = 0.0


class TimeTriggeredPoseCorruptionManager(PoseCorruptionManager):
    """Trigger a corruption event at a fixed onset step (once per episode)."""

    def __init__(
        self,
        cfg: PoseCorruptionCfg,
        *,
        onset_step: int,
        duration_steps: int,
        num_envs: int,
        device: str,
    ):
        cfg = PoseCorruptionCfg(**{**cfg.__dict__})
        cfg.enabled = True
        cfg.event_probability = 0.0
        super().__init__(cfg=cfg, num_envs=num_envs, device=device)

        self.onset_step = int(onset_step)
        self.duration_steps = int(duration_steps)
        self.triggered_this_episode = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: torch.Tensor):
        super().reset(env_ids)
        if env_ids is None or len(env_ids) == 0:
            return
        self.triggered_this_episode[env_ids] = False

    def step(self):
        super().step()

        should_trigger = (
            (self.episode_step_count == self.onset_step)
            & (~self.triggered_this_episode)
            & (~self.event_active)
        )
        if should_trigger.any():
            self.event_active[should_trigger] = True
            self.remaining_steps[should_trigger] = int(self.duration_steps)
            self.triggered_this_episode[should_trigger] = True
            self.triggered_count[should_trigger] += 1
            self._drift_bias[should_trigger] = 0.0



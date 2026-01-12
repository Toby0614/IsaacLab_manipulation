"""Pose corruption configuration for oracle object-pose observations.

Implements the `poe3.pdf` pivot:
- Treat cube position in the policy observation as the output of a perception module (oracle pose).
- During training/evaluation, corrupt that pose estimate with controlled outages:
  hard dropout, freeze (stale), delay, noise+drift.

This file only defines configs; the wrapper/manager apply the corruption.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


CorruptionMode = Literal["hard", "freeze", "delay", "noise", "mixed"]


@dataclass
class PoseCorruptionCfg:
    """Configuration for pose corruption events."""

    enabled: bool = False

    # Which corruption to apply when an event is active.
    mode: CorruptionMode = "freeze"

    # Event schedule (similar to modality dropout):
    # - When not active, start an event with probability `event_probability` each step.
    # - When active, keep it for `duration_steps`, then stop.
    event_probability: float = 0.02
    duration_range: tuple[int, int] = (5, 40)  # steps

    # Mixed mode selection
    mixed_mode_probs: dict[str, float] = field(
        default_factory=lambda: {"freeze": 0.5, "hard": 0.25, "noise": 0.25}
    )

    # Hard dropout parameters
    hard_value: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Delay parameters (if mode="delay" or chosen under "mixed")
    delay_steps: int = 5

    # Noise parameters (meters)
    noise_std: float = 0.01  # 10mm

    # Drift parameters: persistent bias updated each step during an active event.
    drift_alpha: float = 0.99
    drift_noise_std: float = 0.001  # 1mm per step equivalent (small)

    # Optional curriculum knobs (you can override these externally in scripts if desired)
    # Here we just keep a placeholder; curriculum policy lives in training script logic.
    curriculum_stage: int = 0



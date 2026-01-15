
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


CorruptionMode = Literal["hard", "freeze", "delay", "noise", "mixed"]


@dataclass
class PoseCorruptionCfg:

    enabled: bool = False

    mode: CorruptionMode = "freeze"

    event_probability: float = 0.02
    duration_range: tuple[int, int] = (5, 40)  # steps

    mixed_mode_probs: dict[str, float] = field(
        default_factory=lambda: {"freeze": 0.5, "hard": 0.25, "noise": 0.25}
    )

    hard_value: tuple[float, float, float] = (0.0, 0.0, 0.0)

    delay_steps: int = 5

    noise_std: float = 0.01  # 10mm

    drift_alpha: float = 0.99
    drift_noise_std: float = 0.001  # 1mm per step equivalent (small)

    curriculum_stage: int = 0



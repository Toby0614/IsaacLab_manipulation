

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ModalityDropoutCfg:
    
    enabled: bool = False
    
    dropout_mode: Literal["hard", "soft", "mixed"] = "hard"
    
    dropout_duration_range: tuple[int, int] = (5, 50)  # 0.25s to 2.5s at 20Hz
    dropout_probability: float = 0.02  # 2% chance per step to start new event
    
    phase_aware: bool = False
    phase_dropout_config: dict[str, float] = field(default_factory=lambda: {
        "reach": 0.01,      # Low dropout during reach
        "grasp": 0.03,      # Higher during grasp
        "lift": 0.02,       # Moderate during lift
        "transport": 0.04,  # Highest during transport
        "place": 0.03,      # High during place
    })
    
    hard_dropout_value_rgb: float = 0.0  # Black screen
    hard_dropout_value_depth: float = 0.0  # Invalid depth
    
    gaussian_noise_std_rgb: float = 20.0  # On 0-255 scale
    gaussian_noise_std_depth: float = 0.05  # On normalized scale
    cutout_prob: float = 0.3
    cutout_size: tuple[int, int] = (24, 24)  # For 64x64 images
    depth_speckle_prob: float = 0.15
    
    dropout_rgb: bool = True
    dropout_depth: bool = True
    
    provide_dropout_indicator: bool = False  # If True, policy observes dropout state
    
    eval_mode: bool = False
    eval_dropout_start_step: int = 25  # ~1.25s into episode
    eval_dropout_duration: int = 20  # 1 second dropout


@dataclass
class PhaseBasedDropoutCfg(ModalityDropoutCfg):
    
    enabled: bool = True
    phase_aware: bool = True
    dropout_mode: Literal["hard", "soft", "mixed"] = "hard"
    dropout_duration_range: tuple[int, int] = (10, 40)  # 0.5-2.0s
    
    phase_dropout_config: dict[str, float] = field(default_factory=lambda: {
        "reach": 0.015,
        "grasp": 0.05,      # Most critical
        "lift": 0.03,
        "transport": 0.06,  # Highest risk
        "place": 0.04,
    })


@dataclass
class HardDropoutTrainingCfg(ModalityDropoutCfg):
    
    enabled: bool = True
    dropout_mode: Literal["hard", "soft", "mixed"] = "hard"
    dropout_probability: float = 0.025  # ~2.5% chance per step
    dropout_duration_range: tuple[int, int] = (10, 60)  # 0.5-3.0s
    provide_dropout_indicator: bool = False  # Unannounced (policy must handle blindly)


@dataclass
class SoftDropoutTrainingCfg(ModalityDropoutCfg):
    
    enabled: bool = True
    dropout_mode: Literal["hard", "soft", "mixed"] = "soft"
    dropout_probability: float = 0.05  # More frequent but less severe
    dropout_duration_range: tuple[int, int] = (5, 30)
    
    gaussian_noise_std_rgb: float = 30.0
    gaussian_noise_std_depth: float = 0.08
    cutout_prob: float = 0.5
    depth_speckle_prob: float = 0.25


@dataclass
class MixedDropoutTrainingCfg(ModalityDropoutCfg):
    
    enabled: bool = True
    dropout_mode: Literal["hard", "soft", "mixed"] = "mixed"
    dropout_probability: float = 0.03
    dropout_duration_range: tuple[int, int] = (10, 50)
    
    gaussian_noise_std_rgb: float = 25.0
    gaussian_noise_std_depth: float = 0.06
    cutout_prob: float = 0.4
    depth_speckle_prob: float = 0.20


@dataclass
class EvalDropoutCfg(ModalityDropoutCfg):
    
    enabled: bool = True
    eval_mode: bool = True
    dropout_mode: Literal["hard", "soft", "mixed"] = "hard"
    
    eval_dropout_start_step: int = 25  # Start at 1.25s
    eval_dropout_duration: int = 20    # 1.0s duration
    
    phase_aware: bool = False


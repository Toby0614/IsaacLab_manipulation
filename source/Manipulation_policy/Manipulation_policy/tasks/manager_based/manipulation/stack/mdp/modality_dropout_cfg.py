# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration classes for modality dropout system.

Based on poe2.pdf recommendations for vision-based RL robustness studies.
Supports:
- Hard dropout (complete vision blackout)
- Soft dropout (noise/corruption)
- Duration-based dropout (continuous failures over multiple steps)
- Phase-aware dropout (different dropout during reach/grasp/lift/place phases)
- Announced vs unannounced dropout modes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ModalityDropoutCfg:
    """Configuration for modality dropout system.
    
    This implements the vision failure simulation described in poe2.pdf page 7+.
    
    Attributes:
        enabled: Master toggle for dropout system. Set to False to disable completely.
        
        # Dropout Type
        dropout_mode: "hard" (complete blackout), "soft" (noise), or "mixed"
        
        # Duration Control
        dropout_duration_range: (min_steps, max_steps) for dropout events
        dropout_probability: Probability of starting a new dropout event each step (when not already dropping)
        
        # Phase-Based Dropout
        phase_aware: If True, apply different dropout rates per task phase
        phase_dropout_config: Dict mapping phase names to dropout probabilities
        
        # Hard Dropout Parameters
        hard_dropout_value_rgb: Value to set RGB pixels to during hard dropout (0-255)
        hard_dropout_value_depth: Value to set depth pixels to during hard dropout (normalized)
        
        # Soft Dropout Parameters (Noise/Corruption)
        gaussian_noise_std_rgb: Std dev of additive Gaussian noise for RGB (0-255 scale)
        gaussian_noise_std_depth: Std dev of additive Gaussian noise for depth (normalized)
        cutout_prob: Probability of random rectangular cutout per image
        cutout_size: (height, width) of cutout region
        depth_speckle_prob: Probability of per-pixel holes in depth
        
        # Modality Selection
        dropout_rgb: Whether to apply dropout to RGB channels
        dropout_depth: Whether to apply dropout to depth channel
        
        # Announced vs Unannounced
        provide_dropout_indicator: If True, add binary observation indicating dropout state
        
        # Evaluation/Analysis Mode
        eval_mode: If True, use deterministic dropout schedule for reproducible evaluation
        eval_dropout_start_step: Step to start dropout in eval mode
        eval_dropout_duration: Fixed duration in eval mode
    """
    
    # Master toggle
    enabled: bool = False
    
    # Dropout type
    dropout_mode: Literal["hard", "soft", "mixed"] = "hard"
    
    # Duration control (continuous dropout over multiple steps)
    dropout_duration_range: tuple[int, int] = (5, 50)  # 0.25s to 2.5s at 20Hz
    dropout_probability: float = 0.02  # 2% chance per step to start new event
    
    # Phase-based dropout
    phase_aware: bool = False
    phase_dropout_config: dict[str, float] = field(default_factory=lambda: {
        "reach": 0.01,      # Low dropout during reach
        "grasp": 0.03,      # Higher during grasp
        "lift": 0.02,       # Moderate during lift
        "transport": 0.04,  # Highest during transport
        "place": 0.03,      # High during place
    })
    
    # Hard dropout parameters
    hard_dropout_value_rgb: float = 0.0  # Black screen
    hard_dropout_value_depth: float = 0.0  # Invalid depth
    
    # Soft dropout parameters (noise/corruption)
    gaussian_noise_std_rgb: float = 20.0  # On 0-255 scale
    gaussian_noise_std_depth: float = 0.05  # On normalized scale
    cutout_prob: float = 0.3
    cutout_size: tuple[int, int] = (24, 24)  # For 64x64 images
    depth_speckle_prob: float = 0.15
    
    # Modality selection
    dropout_rgb: bool = True
    dropout_depth: bool = True
    
    # Announced vs unannounced
    provide_dropout_indicator: bool = False  # If True, policy observes dropout state
    
    # Evaluation mode (deterministic dropout for reproducible testing)
    eval_mode: bool = False
    eval_dropout_start_step: int = 25  # ~1.25s into episode
    eval_dropout_duration: int = 20  # 1 second dropout


@dataclass
class PhaseBasedDropoutCfg(ModalityDropoutCfg):
    """Preset: Phase-aware dropout for systematic evaluation.
    
    This configuration enables phase-based dropout with different failure rates
    during different manipulation phases, as recommended in poe2.pdf.
    """
    
    enabled: bool = True
    phase_aware: bool = True
    dropout_mode: Literal["hard", "soft", "mixed"] = "hard"
    dropout_duration_range: tuple[int, int] = (10, 40)  # 0.5-2.0s
    
    # Higher dropout during critical phases
    phase_dropout_config: dict[str, float] = field(default_factory=lambda: {
        "reach": 0.015,
        "grasp": 0.05,      # Most critical
        "lift": 0.03,
        "transport": 0.06,  # Highest risk
        "place": 0.04,
    })


@dataclass
class HardDropoutTrainingCfg(ModalityDropoutCfg):
    """Preset: Hard dropout for robust training (M3 baseline in poe2.pdf).
    
    Train with intermittent complete vision loss to learn robust policies.
    """
    
    enabled: bool = True
    dropout_mode: Literal["hard", "soft", "mixed"] = "hard"
    dropout_probability: float = 0.025  # ~2.5% chance per step
    dropout_duration_range: tuple[int, int] = (10, 60)  # 0.5-3.0s
    provide_dropout_indicator: bool = False  # Unannounced (policy must handle blindly)


@dataclass
class SoftDropoutTrainingCfg(ModalityDropoutCfg):
    """Preset: Soft dropout (noise/corruption) for visual robustness.
    
    Train with noisy/corrupted vision rather than complete blackout.
    """
    
    enabled: bool = True
    dropout_mode: Literal["hard", "soft", "mixed"] = "soft"
    dropout_probability: float = 0.05  # More frequent but less severe
    dropout_duration_range: tuple[int, int] = (5, 30)
    
    # Aggressive noise parameters
    gaussian_noise_std_rgb: float = 30.0
    gaussian_noise_std_depth: float = 0.08
    cutout_prob: float = 0.5
    depth_speckle_prob: float = 0.25


@dataclass
class MixedDropoutTrainingCfg(ModalityDropoutCfg):
    """Preset: Mixed hard + soft dropout for maximum robustness.
    
    Combines complete failures with noisy observations.
    """
    
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
    """Preset: Deterministic dropout for evaluation.
    
    Use for reproducible testing of policy robustness under vision failures.
    """
    
    enabled: bool = True
    eval_mode: bool = True
    dropout_mode: Literal["hard", "soft", "mixed"] = "hard"
    
    # Deterministic schedule
    eval_dropout_start_step: int = 25  # Start at 1.25s
    eval_dropout_duration: int = 20    # 1.0s duration
    
    # Can be overridden to test different phases/durations
    phase_aware: bool = False


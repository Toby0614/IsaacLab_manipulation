# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluation-specific dropout configurations for Variant 1 (phase-based) and Variant 2 (time-based).

Based on poe2.pdf recommendations for systematic robustness evaluation:
- Variant 1: Phase-based dropout - triggers dropout at specific manipulation phases
- Variant 2: Time-based dropout - triggers dropout at specific onset times (steps)

Both variants support configurable dropout duration for creating 2D sensitivity maps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class EvalDropoutBaseCfg:
    """Base configuration for evaluation dropout.
    
    Common parameters for both Variant 1 and Variant 2.
    """
    
    # Master toggle
    enabled: bool = True
    
    # Dropout mode
    dropout_mode: Literal["hard", "soft", "mixed"] = "hard"
    
    # Modality selection
    dropout_rgb: bool = True
    dropout_depth: bool = True
    
    # Hard dropout values
    hard_dropout_value_rgb: float = 0.0
    hard_dropout_value_depth: float = 0.0
    
    # Soft dropout parameters (for soft/mixed modes)
    gaussian_noise_std_rgb: float = 25.0
    gaussian_noise_std_depth: float = 0.06
    
    # Dropout indicator (for "announced" dropout evaluation)
    provide_dropout_indicator: bool = False


@dataclass
class Variant1PhaseDropoutCfg(EvalDropoutBaseCfg):
    """Variant 1: Phase-based dropout configuration.
    
    Triggers dropout when the task enters a specific manipulation phase.
    
    Phases (from poe2.pdf):
    - A / "reach": Before first contact - moving toward object
    - B / "grasp": Grasp closure - first contact to grasp confirmed
    - C / "lift" or "transport": Object grasped and moving toward goal
    - D / "place": At goal, lowering/releasing object
    
    Usage:
        cfg = Variant1PhaseDropoutCfg(
            target_phase="grasp",
            dropout_duration_steps=40,  # 2 seconds at 20Hz
        )
    """
    
    # Target phase to trigger dropout
    target_phase: Literal["reach", "grasp", "lift", "transport", "place"] = "grasp"
    
    # Duration of dropout once triggered (in simulation steps)
    dropout_duration_steps: int = 20  # 1 second at 20Hz
    
    # Whether to trigger dropout only on first entry to phase (per episode)
    trigger_once_per_episode: bool = True
    
    # Delay after phase entry before dropout starts (steps)
    phase_entry_delay: int = 0
    
    # Whether to require stable phase (avoid flicker triggers)
    require_stable_phase: bool = True
    stable_phase_steps: int = 3  # Must be in phase for N steps


@dataclass 
class Variant2TimeDropoutCfg(EvalDropoutBaseCfg):
    """Variant 2: Time-based (onset-time) dropout configuration.
    
    Triggers dropout at a specific timestep (onset time) in the episode.
    
    Usage:
        cfg = Variant2TimeDropoutCfg(
            onset_step=50,  # Start dropout at step 50 (~2.5s at 20Hz)
            dropout_duration_steps=30,  # 1.5 seconds
        )
    """
    
    # Step at which to start dropout
    onset_step: int = 25  # 1.25 seconds at 20Hz
    
    # Duration of dropout (in simulation steps)
    dropout_duration_steps: int = 20  # 1 second at 20Hz


# =============================================================================
# Preset configurations for common evaluation scenarios
# =============================================================================

@dataclass
class PhaseA_ReachDropoutCfg(Variant1PhaseDropoutCfg):
    """Phase A dropout: During reach phase (before object contact)."""
    target_phase: Literal["reach", "grasp", "lift", "transport", "place"] = "reach"


@dataclass
class PhaseB_GraspDropoutCfg(Variant1PhaseDropoutCfg):
    """Phase B dropout: During grasp closure."""
    target_phase: Literal["reach", "grasp", "lift", "transport", "place"] = "grasp"


@dataclass
class PhaseC_LiftDropoutCfg(Variant1PhaseDropoutCfg):
    """Phase C dropout: During lift phase."""
    target_phase: Literal["reach", "grasp", "lift", "transport", "place"] = "lift"


@dataclass
class PhaseC_TransportDropoutCfg(Variant1PhaseDropoutCfg):
    """Phase C dropout: During transport phase."""
    target_phase: Literal["reach", "grasp", "lift", "transport", "place"] = "transport"


@dataclass
class PhaseD_PlaceDropoutCfg(Variant1PhaseDropoutCfg):
    """Phase D dropout: During place/release phase."""
    target_phase: Literal["reach", "grasp", "lift", "transport", "place"] = "place"


# =============================================================================
# Grid configuration for systematic evaluation
# =============================================================================

@dataclass
class EvalGridConfig:
    """Configuration for systematic evaluation grid.
    
    Defines the parameter space for 2D sensitivity analysis.
    """
    
    # Variant 1: Phase-based evaluation
    phases: list = field(default_factory=lambda: ["reach", "grasp", "lift", "transport", "place"])
    
    # Variant 2: Time-based evaluation (onset steps)
    # At 20Hz: 25 steps = 1.25s, 50 = 2.5s, 100 = 5s, etc.
    onset_steps: list = field(default_factory=lambda: [10, 25, 50, 75, 100, 125, 150])
    
    # Duration grid (steps) - same for both variants
    # At 20Hz: 5 = 0.25s, 10 = 0.5s, 20 = 1s, 40 = 2s, 80 = 4s
    durations: list = field(default_factory=lambda: [5, 10, 20, 40, 60, 80, 100])
    
    # Evaluation parameters
    num_eval_episodes: int = 100  # Episodes per condition
    num_envs: int = 64  # Parallel environments
    episode_length: int = 250  # Max steps per episode
    
    # Random seeds for reproducibility
    seeds: list = field(default_factory=lambda: [42, 123, 456])


# Default evaluation grid
DEFAULT_EVAL_GRID = EvalGridConfig()

# Fine-grained grid for publication
PUBLICATION_EVAL_GRID = EvalGridConfig(
    phases=["reach", "grasp", "lift", "transport", "place"],
    onset_steps=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200],
    durations=[5, 10, 15, 20, 30, 40, 50, 60, 80, 100, 120],
    num_eval_episodes=200,
    seeds=[42, 123, 456, 789, 1011],
)

# Quick grid for debugging
DEBUG_EVAL_GRID = EvalGridConfig(
    phases=["reach", "grasp", "place"],
    onset_steps=[25, 75, 125],
    durations=[10, 30, 60],
    num_eval_episodes=20,
    num_envs=16,
    seeds=[42],
)


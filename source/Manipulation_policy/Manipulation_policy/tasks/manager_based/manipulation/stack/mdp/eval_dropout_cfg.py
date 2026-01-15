

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class EvalDropoutBaseCfg:
    
    enabled: bool = True
    
    dropout_mode: Literal["hard", "soft", "mixed"] = "hard"
    
    dropout_rgb: bool = True
    dropout_depth: bool = True
    
    hard_dropout_value_rgb: float = 0.0
    hard_dropout_value_depth: float = 0.0
    
    gaussian_noise_std_rgb: float = 25.0
    gaussian_noise_std_depth: float = 0.06
    
    provide_dropout_indicator: bool = False


@dataclass
class Variant1PhaseDropoutCfg(EvalDropoutBaseCfg):
    
    target_phase: Literal["reach", "grasp", "lift", "transport", "place"] = "grasp"
    
    dropout_duration_steps: int = 20  # 1 second at 20Hz
    
    trigger_once_per_episode: bool = True
    
    phase_entry_delay: int = 0
    
    require_stable_phase: bool = True
    stable_phase_steps: int = 3  # Must be in phase for N steps


@dataclass 
class Variant2TimeDropoutCfg(EvalDropoutBaseCfg):
    
    onset_step: int = 25  # 1.25 seconds at 20Hz
    
    dropout_duration_steps: int = 20  # 1 second at 20Hz



@dataclass
class PhaseA_ReachDropoutCfg(Variant1PhaseDropoutCfg):
    target_phase: Literal["reach", "grasp", "lift", "transport", "place"] = "reach"


@dataclass
class PhaseB_GraspDropoutCfg(Variant1PhaseDropoutCfg):
    target_phase: Literal["reach", "grasp", "lift", "transport", "place"] = "grasp"


@dataclass
class PhaseC_LiftDropoutCfg(Variant1PhaseDropoutCfg):
    target_phase: Literal["reach", "grasp", "lift", "transport", "place"] = "lift"


@dataclass
class PhaseC_TransportDropoutCfg(Variant1PhaseDropoutCfg):
    target_phase: Literal["reach", "grasp", "lift", "transport", "place"] = "transport"


@dataclass
class PhaseD_PlaceDropoutCfg(Variant1PhaseDropoutCfg):
    target_phase: Literal["reach", "grasp", "lift", "transport", "place"] = "place"



@dataclass
class EvalGridConfig:
    
    phases: list = field(default_factory=lambda: ["reach", "grasp", "lift", "transport", "place"])
    
    onset_steps: list = field(default_factory=lambda: [10, 25, 50, 75, 100, 125, 150])
    
    durations: list = field(default_factory=lambda: [5, 10, 20, 40, 60, 80, 100])
    
    num_eval_episodes: int = 100  # Episodes per condition
    num_envs: int = 64  # Parallel environments
    episode_length: int = 250  # Max steps per episode
    
    seeds: list = field(default_factory=lambda: [42, 123, 456])


DEFAULT_EVAL_GRID = EvalGridConfig()

PUBLICATION_EVAL_GRID = EvalGridConfig(
    phases=["reach", "grasp", "lift", "transport", "place"],
    onset_steps=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200],
    durations=[5, 10, 15, 20, 30, 40, 50, 60, 80, 100, 120],
    num_eval_episodes=200,
    seeds=[42, 123, 456, 789, 1011],
)

DEBUG_EVAL_GRID = EvalGridConfig(
    phases=["reach", "grasp", "place"],
    onset_steps=[25, 75, 125],
    durations=[10, 30, 60],
    num_eval_episodes=20,
    num_envs=16,
    seeds=[42],
)


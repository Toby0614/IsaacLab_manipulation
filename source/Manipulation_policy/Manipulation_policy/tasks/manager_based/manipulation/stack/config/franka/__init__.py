# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka pick-and-place task registration."""

import gymnasium as gym

from . import agents

# Register the pick-and-place environment
gym.register(
    id="Isaac-Franka-PickPlace-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pickplace_env_cfg:FrankaPickPlaceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cnn_cfg:PPORunnerCfg",
    },
    disable_env_checker=True,
)

# =============================================================================
# FORCE SENSING VARIANTS
# =============================================================================
# Import force-sensing configs to trigger their gym.register() calls
from . import pickplace_env_cfg_with_force  # noqa: F401

# The following environments are now available:
# - Isaac-Franka-PickPlace-Force-v0: Force + closure info (3 dims) [recommended]
# - Isaac-Franka-PickPlace-Force-Scalar-v0: Single scalar force (1 dim)
# - Isaac-Franka-PickPlace-Force-GraspIndicator-v0: Grasp quality indicator (3 dims)

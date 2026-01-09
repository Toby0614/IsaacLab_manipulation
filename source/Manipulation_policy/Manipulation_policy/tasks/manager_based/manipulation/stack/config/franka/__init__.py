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

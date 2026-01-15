

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Franka-PickPlace-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pickplace_env_cfg:FrankaPickPlaceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cnn_cfg:PPORunnerCfg",
    },
    disable_env_checker=True,
)

from . import pickplace_env_cfg_with_force  # noqa: F401


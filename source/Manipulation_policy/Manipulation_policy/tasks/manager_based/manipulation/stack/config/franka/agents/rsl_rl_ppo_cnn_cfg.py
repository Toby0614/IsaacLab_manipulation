# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO config using ActorCriticCNN (IsaacLab 2.3 + RSL-RL visual support).

This file intentionally mirrors the standard PPO runner cfg, but switches the *policy module class*
to `ActorCriticCNN` so RSL-RL uses its CNN vision encoder.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class RslRlPpoActorCriticCNNCfg(RslRlPpoActorCriticCfg):
    """Extends the base actor-critic cfg with CNN encoder configs required by ActorCriticCNN."""

    # Per-observation-group CNN configs (see rsl_rl.networks.CNN signature)
    actor_cnn_cfg: dict = None  # type: ignore
    critic_cnn_cfg: dict = None  # type: ignore


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 4000
    save_interval = 100
    experiment_name = "franka_grasp_visuomotor_rslrl_cnn"

    # Map environment observation groups to algorithm observation sets (required by rsl_rl>=3.2).
    # Our env exposes top-level observation groups: "proprio" (B,D) and "rgbd" (B,4,H,W).
    obs_groups = {
        "policy": ["proprio", "rgbd"],
        "critic": ["proprio", "rgbd"],
    }

    # CNN + MLP fusion policy
    policy = RslRlPpoActorCriticCNNCfg(
        class_name="ActorCriticCNN",
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 256],
        critic_hidden_dims=[256, 256],
        activation="elu",
        # A simple starting CNN for 128x128 images
        actor_cnn_cfg={
            # if a single dict is provided, ActorCriticCNN applies it to each 2D obs group
            "output_channels": [32, 64, 64],
            "kernel_size": [8, 4, 3],
            "stride": [4, 2, 1],
            "padding": "zeros",
            "norm": "none",
            "activation": "elu",
            "max_pool": False,
            "global_pool": "avg",
            "flatten": True,
        },
        critic_cnn_cfg={
            "output_channels": [32, 64, 64],
            "kernel_size": [8, 4, 3],
            "stride": [4, 2, 1],
            "padding": "zeros",
            "norm": "none",
            "activation": "elu",
            "max_pool": False,
            "global_pool": "avg",
            "flatten": True,
        },
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.003,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    # no __post_init__ needed

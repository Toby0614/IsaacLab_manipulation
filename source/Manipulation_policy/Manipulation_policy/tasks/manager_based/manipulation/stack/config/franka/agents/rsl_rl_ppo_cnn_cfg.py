

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class RslRlPpoActorCriticCNNCfg(RslRlPpoActorCriticCfg):
    actor_cnn_cfg: dict = None
    critic_cnn_cfg: dict = None


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    
    num_steps_per_env = 32
    max_iterations = 8000
    save_interval = 500
    experiment_name = "franka_pickplace"

    obs_groups = {
        "policy": ["proprio", "multi_cam"],
        "critic": ["proprio", "multi_cam"],
    }

    policy = RslRlPpoActorCriticCNNCfg(
        class_name="ActorCriticCNN",
        init_noise_std=0.3,  # Low noise for stable learning
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        actor_cnn_cfg={
            "output_channels": [32, 64, 128],
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
            "output_channels": [32, 64, 128],
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
        entropy_coef=0.01,
        num_learning_epochs=8,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.97,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )

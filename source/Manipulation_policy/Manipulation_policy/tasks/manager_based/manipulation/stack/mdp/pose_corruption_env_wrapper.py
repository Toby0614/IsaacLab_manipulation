"""Environment wrapper that corrupts oracle object-pose observations.

This wrapper does NOT change rewards/terminations or physics.
It only edits the observation passed to the policy by corrupting the cube position signal that
is included in the `proprio` group (see `pickplace_env_cfg.py`).

Designed to implement the `poe3.pdf` plan: pose-dropout/noise/delay as a proxy for perception outages.
"""

from __future__ import annotations

import torch
import gymnasium as gym
from typing import Any

from .pose_corruption_cfg import PoseCorruptionCfg
from .pose_corruption_manager import PoseCorruptionManager


class PoseCorruptionEnvWrapper(gym.Wrapper):
    """Gym wrapper that corrupts cube position inside the `proprio` observation vector."""

    def __init__(self, env: gym.Env, cfg: PoseCorruptionCfg):
        super().__init__(env)
        self.cfg = cfg

        base_env = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
        self.manager = PoseCorruptionManager(cfg=cfg, num_envs=base_env.num_envs, device=str(base_env.device))

        # Attach for debugging/introspection
        base_env.pose_corruption_manager = self.manager

        # Patch observation manager so all callers (including RslRlVecEnvWrapper.get_observations)
        # see corrupted proprio consistently.
        self._maybe_patch_observation_manager()

        print(
            f"[PoseCorruptionEnvWrapper] enabled={cfg.enabled}, mode={cfg.mode}, "
            f"p_start={cfg.event_probability}, duration_range={cfg.duration_range}"
        )

    def _maybe_patch_observation_manager(self) -> None:
        base_env = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
        obs_mgr = getattr(base_env, "observation_manager", None)
        if obs_mgr is None or not hasattr(obs_mgr, "compute"):
            return

        # Avoid double patching
        if getattr(obs_mgr, "_pose_corruption_patched", False):
            return

        orig_compute = obs_mgr.compute

        def compute_patched(*args, **kwargs):
            obs = orig_compute(*args, **kwargs)
            return self._corrupt_obs(obs)

        obs_mgr.compute = compute_patched  # type: ignore[assignment]
        obs_mgr._pose_corruption_patched = True  # type: ignore[attr-defined]

    def _corrupt_obs(self, obs: Any) -> Any:
        """Corrupt cube_position slice in obs['proprio'] if configured."""
        if not self.cfg.enabled:
            return obs

        # Support dict-like and TensorDict-like objects
        if hasattr(obs, "get") and hasattr(obs, "keys"):
            # TensorDict behaves similarly to dict for key access
            if "proprio" not in obs:
                return obs
            proprio = obs["proprio"]
            if not torch.is_tensor(proprio) or proprio.ndim != 2 or proprio.shape[1] < 3:
                return obs

            # Assumption by design: `cube_position` is appended at the END of the base proprio vector.
            # This wrapper is intended to be applied BEFORE the force wrapper (which appends to proprio),
            # so the last 3 dims correspond to cube_position at this stage.
            cube_pos = proprio[:, -3:]
            self.manager.step()
            cube_pos_corrupt = self.manager.apply(cube_pos)
            obs["proprio"] = torch.cat([proprio[:, :-3], cube_pos_corrupt], dim=1)
            return obs

        return obs

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        base_env = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
        all_env_ids = torch.arange(base_env.num_envs, device=base_env.device)
        self.manager.reset(all_env_ids)
        return out

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        done = terminated | truncated
        reset_env_ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.manager.reset(reset_env_ids)

        # Note: actual corruption is applied via observation_manager.compute patch.
        return obs, reward, terminated, truncated, info


class VecEnvPoseCorruptionWrapper:
    """VecEnv-style wrapper variant (for symmetry with other wrappers)."""

    def __init__(self, env, cfg: PoseCorruptionCfg):
        self.env = env
        self.cfg = cfg

        base_env = env.unwrapped if hasattr(env, "unwrapped") else env
        self.manager = PoseCorruptionManager(cfg=cfg, num_envs=base_env.num_envs, device=str(base_env.device))
        base_env.pose_corruption_manager = self.manager

        self._maybe_patch_observation_manager()

        print(
            f"[VecEnvPoseCorruptionWrapper] enabled={cfg.enabled}, mode={cfg.mode}, "
            f"p_start={cfg.event_probability}, duration_range={cfg.duration_range}"
        )

    def _maybe_patch_observation_manager(self) -> None:
        base_env = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
        obs_mgr = getattr(base_env, "observation_manager", None)
        if obs_mgr is None or not hasattr(obs_mgr, "compute"):
            return
        if getattr(obs_mgr, "_pose_corruption_patched", False):
            return

        orig_compute = obs_mgr.compute

        def compute_patched(*args, **kwargs):
            obs = orig_compute(*args, **kwargs)
            return self._corrupt_obs(obs)

        obs_mgr.compute = compute_patched  # type: ignore[assignment]
        obs_mgr._pose_corruption_patched = True  # type: ignore[attr-defined]

    def _corrupt_obs(self, obs: Any) -> Any:
        if not self.cfg.enabled:
            return obs
        if hasattr(obs, "get") and hasattr(obs, "keys"):
            if "proprio" not in obs:
                return obs
            proprio = obs["proprio"]
            if not torch.is_tensor(proprio) or proprio.ndim != 2 or proprio.shape[1] < 3:
                return obs
            cube_pos = proprio[:, -3:]
            self.manager.step()
            cube_pos_corrupt = self.manager.apply(cube_pos)
            obs["proprio"] = torch.cat([proprio[:, :-3], cube_pos_corrupt], dim=1)
            return obs
        return obs

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        base_env = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
        all_env_ids = torch.arange(base_env.num_envs, device=base_env.device)
        self.manager.reset(all_env_ids)
        return out

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated | truncated
        reset_env_ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.manager.reset(reset_env_ids)
        return obs, reward, terminated, truncated, info

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)



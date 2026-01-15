
from __future__ import annotations

import torch
import gymnasium as gym
from typing import Any

from .pose_corruption_cfg import PoseCorruptionCfg
from .pose_corruption_manager import PoseCorruptionManager


class PoseCorruptionEnvWrapper(gym.Wrapper):

    def __init__(self, env: gym.Env, cfg: PoseCorruptionCfg):
        super().__init__(env)
        self.cfg = cfg

        base_env = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
        self.manager = PoseCorruptionManager(cfg=cfg, num_envs=base_env.num_envs, device=str(base_env.device))

        base_env.pose_corruption_manager = self.manager

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

            base_env = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env

            idxs = getattr(base_env, "__pose_corruption_cube_pos_indices", None)
            if idxs is None:
                try:
                    obj = base_env.scene["cube_2"]
                    cube_pos_oracle = obj.data.root_pos_w - base_env.scene.env_origins  # (B,3)
                    b = int(min(64, proprio.shape[0]))
                    p = proprio[:b].float()
                    c = cube_pos_oracle[:b].float()

                    err = torch.empty((3, p.shape[1]), device=p.device, dtype=torch.float32)
                    for d in range(3):
                        err[d] = (p - c[:, d : d + 1]).abs().mean(dim=0)
                    best = torch.argmin(err, dim=1)
                    best_err = err[torch.arange(3, device=p.device), best]

                    if torch.any(best_err > 1e-3):
                        idxs = None
                    else:
                        idxs = [int(best[0].item()), int(best[1].item()), int(best[2].item())]
                except Exception:
                    idxs = None

                setattr(base_env, "__pose_corruption_cube_pos_indices", idxs)

            if idxs is None:
                cube_pos = proprio[:, -3:]
                write_mode = "slice"
            else:
                idx = torch.as_tensor(idxs, device=proprio.device, dtype=torch.long)
                cube_pos = proprio.index_select(dim=1, index=idx)
                write_mode = "indices"

            self.manager.step()
            cube_pos_corrupt = self.manager.apply(cube_pos)
            if write_mode == "slice":
                obs["proprio"] = torch.cat([proprio[:, :-3], cube_pos_corrupt], dim=1)
            else:
                proprio_out = proprio.clone()
                proprio_out[:, idx] = cube_pos_corrupt
                obs["proprio"] = proprio_out
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


class VecEnvPoseCorruptionWrapper:

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



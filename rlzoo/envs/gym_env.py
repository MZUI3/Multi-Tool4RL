"""Single-agent Gymnasium environments and a minimal synchronous vector env."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np


@dataclass
class EnvSpec:
    obs_dim: int
    discrete: bool
    n_actions: int = 0
    act_dim: int = 0
    act_low: Optional[np.ndarray] = None
    act_high: Optional[np.ndarray] = None
    num_envs: int = 1
    obs_shape: Optional[Tuple[int, ...]] = None  # original observation shape (images: H, W, C)
    obs_dtype: type = np.float32  # np.uint8 for image observations (stored compactly, scaled by the encoder)
    goal_dim: int = 0  # > 0 for goal-conditioned envs (obs = [observation, achieved_goal, desired_goal])
    env_id: str = ""


def make_gym_env(env_cfg, render_mode: Optional[str] = None):
    import gymnasium as gym

    from rlzoo.envs import builtin  # noqa: F401 - registers BitFlip-v0, PointReach-v0, PixelGrid-v0

    kwargs = dict(env_cfg.get("kwargs") or {})
    if render_mode:
        kwargs["render_mode"] = render_mode
    return gym.make(env_cfg["id"], **kwargs)


def spec_from_env(env, num_envs: int = 1) -> EnvSpec:
    from gymnasium import spaces

    obs_space = env.observation_space
    goal_dim = 0
    if isinstance(obs_space, spaces.Dict):  # goal-conditioned (gymnasium-robotics style)
        keys = ("observation", "achieved_goal", "desired_goal")
        if not all(k in obs_space.spaces for k in keys):
            raise TypeError(f"Dict observation spaces need keys {keys}, got {list(obs_space.spaces)}")
        goal_dim = int(np.prod(obs_space["desired_goal"].shape))
        obs_dim = int(sum(np.prod(obs_space[k].shape) for k in keys))
        obs_shape, obs_dtype = (obs_dim,), np.float32
    elif isinstance(obs_space, spaces.Box):
        obs_dim = int(np.prod(obs_space.shape))
        obs_shape = tuple(int(s) for s in obs_space.shape)
        obs_dtype = np.uint8 if obs_space.dtype == np.uint8 and len(obs_shape) == 3 else np.float32
    else:
        raise TypeError(f"Only Box / goal-Dict observation spaces are supported, got {obs_space}")
    env_id = getattr(getattr(env, "spec", None), "id", "") or ""
    space = env.action_space
    if isinstance(space, spaces.Discrete):
        return EnvSpec(obs_dim=obs_dim, discrete=True, n_actions=int(space.n), num_envs=num_envs, obs_shape=obs_shape,
                       obs_dtype=obs_dtype, goal_dim=goal_dim, env_id=env_id)
    if isinstance(space, spaces.Box):
        return EnvSpec(obs_dim=obs_dim, discrete=False, act_dim=int(np.prod(space.shape)),
                       act_low=space.low.astype(np.float32).ravel(), act_high=space.high.astype(np.float32).ravel(),
                       num_envs=num_envs, obs_shape=obs_shape, obs_dtype=obs_dtype, goal_dim=goal_dim, env_id=env_id)
    raise TypeError(f"Unsupported action space {space}")


def to_env_action(action, spec: EnvSpec):
    if spec.discrete:
        return int(action)
    return np.clip(np.asarray(action, dtype=np.float32), spec.act_low, spec.act_high)


def flatten_obs(o, spec: Optional[EnvSpec] = None) -> np.ndarray:
    """Observation -> flat array. Dict goal observations are concatenated [obs, achieved, desired]; uint8
    images stay uint8 (the encoder rescales them)."""
    if isinstance(o, dict):
        return np.concatenate([np.asarray(o[k], dtype=np.float32).ravel()
                               for k in ("observation", "achieved_goal", "desired_goal")])
    arr = np.asarray(o)
    if spec is not None and spec.obs_dtype == np.uint8 and arr.dtype == np.uint8:
        return arr.ravel()
    return arr.astype(np.float32).ravel()


class VecEnv:
    """Steps N gymnasium envs sequentially and auto-resets finished ones.

    step() returns (obs, rewards, terminated, truncated, info) where info["final_obs"][i] holds the last
    observation of an env that just finished and info["episodes"] lists finished-episode statistics.
    """

    def __init__(self, env_fns: List[Callable], seed: int = 0):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        self.spec = spec_from_env(self.envs[0], self.num_envs)
        self.seed = seed
        self._seeded = False
        self._ret = np.zeros(self.num_envs)
        self._len = np.zeros(self.num_envs, dtype=np.int64)

    def _flat(self, o) -> np.ndarray:
        return flatten_obs(o, self.spec)

    def reset(self) -> np.ndarray:
        obs = []
        for i, env in enumerate(self.envs):
            seed = None if self._seeded else self.seed + i
            if seed is not None:
                env.action_space.seed(seed)
            o, _ = env.reset(seed=seed)
            obs.append(self._flat(o))
        self._seeded = True
        self._ret[:] = 0
        self._len[:] = 0
        return np.stack(obs)

    def sample_actions(self) -> np.ndarray:
        return np.stack([np.asarray(env.action_space.sample()) for env in self.envs])

    def compute_reward(self, achieved: np.ndarray, desired: np.ndarray, info=None) -> np.ndarray:
        """Vectorised goal reward of the underlying goal env (HER relabelling)."""
        env = self.envs[0].unwrapped
        return np.asarray(env.compute_reward(achieved, desired, info), dtype=np.float32)

    def step(self, actions):
        obs, rews, terms, truncs = [], np.zeros(self.num_envs, np.float32), np.zeros(self.num_envs, bool), \
            np.zeros(self.num_envs, bool)
        final_obs, episodes = [None] * self.num_envs, []
        for i, env in enumerate(self.envs):
            o, r, term, trunc, step_info = env.step(to_env_action(actions[i], self.spec))
            self._ret[i] += r
            self._len[i] += 1
            rews[i], terms[i], truncs[i] = r, term, trunc
            if term or trunc:
                final_obs[i] = self._flat(o)
                ep = {"return": float(self._ret[i]), "length": float(self._len[i])}
                if "is_success" in step_info:
                    ep["success"] = float(step_info["is_success"])
                episodes.append(ep)
                self._ret[i], self._len[i] = 0.0, 0
                o, _ = env.reset()
            obs.append(self._flat(o))
        return np.stack(obs), rews, terms, truncs, {"final_obs": final_obs, "episodes": episodes}

    def close(self) -> None:
        for env in self.envs:
            env.close()


def evaluate_rl(algo, env_fn: Callable, episodes: int, seed: int):
    env = env_fn()
    spec = spec_from_env(env)
    returns, lengths, successes = [], [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, ret, length, info = False, 0.0, 0, {}
        if hasattr(algo, "reset_hidden"):
            algo.reset_hidden(1)
        while not done:
            a = algo.act(flatten_obs(obs, spec)[None], deterministic=True)[0]
            obs, r, term, trunc, info = env.step(to_env_action(a, spec))
            if hasattr(algo, "observe"):
                algo.observe(float(r))
            ret += r
            length += 1
            done = term or trunc
        returns.append(ret)
        lengths.append(length)
        if "is_success" in info:
            successes.append(float(info["is_success"]))
    env.close()
    out = {"return": float(np.mean(returns)), "return_std": float(np.std(returns)), "length": float(np.mean(lengths))}
    if successes:
        out["success"] = float(np.mean(successes))
    return out

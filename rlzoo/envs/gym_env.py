"""Single-agent Gymnasium environments and a minimal synchronous vector env."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

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


def make_gym_env(env_cfg, render_mode: Optional[str] = None):
    import gymnasium as gym

    kwargs = dict(env_cfg.get("kwargs") or {})
    if render_mode:
        kwargs["render_mode"] = render_mode
    return gym.make(env_cfg["id"], **kwargs)


def spec_from_env(env, num_envs: int = 1) -> EnvSpec:
    from gymnasium import spaces

    if not isinstance(env.observation_space, spaces.Box):
        raise TypeError(f"Only Box observation spaces are supported, got {env.observation_space}")
    obs_dim = int(np.prod(env.observation_space.shape))
    space = env.action_space
    if isinstance(space, spaces.Discrete):
        return EnvSpec(obs_dim=obs_dim, discrete=True, n_actions=int(space.n), num_envs=num_envs)
    if isinstance(space, spaces.Box):
        return EnvSpec(obs_dim=obs_dim, discrete=False, act_dim=int(np.prod(space.shape)),
                       act_low=space.low.astype(np.float32).ravel(), act_high=space.high.astype(np.float32).ravel(),
                       num_envs=num_envs)
    raise TypeError(f"Unsupported action space {space}")


def to_env_action(action, spec: EnvSpec):
    if spec.discrete:
        return int(action)
    return np.clip(np.asarray(action, dtype=np.float32), spec.act_low, spec.act_high)


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

    @staticmethod
    def _flat(o) -> np.ndarray:
        return np.asarray(o, dtype=np.float32).ravel()

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

    def step(self, actions):
        obs, rews, terms, truncs = [], np.zeros(self.num_envs, np.float32), np.zeros(self.num_envs, bool), \
            np.zeros(self.num_envs, bool)
        final_obs, episodes = [None] * self.num_envs, []
        for i, env in enumerate(self.envs):
            o, r, term, trunc, _ = env.step(to_env_action(actions[i], self.spec))
            self._ret[i] += r
            self._len[i] += 1
            rews[i], terms[i], truncs[i] = r, term, trunc
            if term or trunc:
                final_obs[i] = self._flat(o)
                episodes.append({"return": float(self._ret[i]), "length": float(self._len[i])})
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
    returns, lengths = [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, ret, length = False, 0.0, 0
        while not done:
            a = algo.act(VecEnv._flat(obs)[None], deterministic=True)[0]
            obs, r, term, trunc, _ = env.step(to_env_action(a, spec))
            ret += r
            length += 1
            done = term or trunc
        returns.append(ret)
        lengths.append(length)
    env.close()
    return {"return": float(np.mean(returns)), "return_std": float(np.std(returns)), "length": float(np.mean(lengths))}

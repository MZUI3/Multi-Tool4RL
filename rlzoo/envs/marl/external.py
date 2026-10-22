"""Adapters for optional third-party MARL benchmarks (PettingZoo parallel envs, SMAC)."""
from __future__ import annotations

import importlib

import numpy as np

from rlzoo.envs.marl.base import MultiAgentEnv


class PettingZooEnv(MultiAgentEnv):
    """Wraps a PettingZoo *parallel* env with Discrete actions, e.g. id "pz:mpe2.simple_spread_v3".

    Observations are zero-padded to a common size, the team reward is the mean of agent rewards and the
    global state comes from env.state() when available (else concatenated observations).
    """

    def __init__(self, module: str, episode_limit: int | None = None, **kwargs):
        from gymnasium import spaces

        self.env = importlib.import_module(module).parallel_env(**kwargs)
        self.agents = list(self.env.possible_agents)
        self.n_agents = len(self.agents)
        obs_spaces = [self.env.observation_space(a) for a in self.agents]
        act_spaces = [self.env.action_space(a) for a in self.agents]
        if not all(isinstance(s, spaces.Discrete) for s in act_spaces):
            raise TypeError("PettingZooEnv adapter requires Discrete action spaces")
        self._act_n = [int(s.n) for s in act_spaces]
        self.n_actions = max(self._act_n)
        self.obs_dim = max(int(np.prod(s.shape)) for s in obs_spaces)
        try:  # probe once: some envs declare state_space but do not implement state()
            self.env.reset(seed=0)
            self.state_dim = int(np.asarray(self.env.state(), np.float32).size)
            self._has_state = True
        except Exception:
            self.state_dim = self.obs_dim * self.n_agents
            self._has_state = False
        self.episode_limit = int(episode_limit or kwargs.get("max_cycles", 25))

    def _pack(self, obs_dict):
        obs = np.zeros((self.n_agents, self.obs_dim), np.float32)
        for i, a in enumerate(self.agents):
            if a in obs_dict:
                o = np.asarray(obs_dict[a], np.float32).ravel()
                obs[i, : o.size] = o
        return obs

    def _state(self, obs):
        if self._has_state:
            return np.asarray(self.env.state(), np.float32).ravel()
        return obs.ravel()

    def _avail(self):
        avail = np.zeros((self.n_agents, self.n_actions), np.float32)
        for i, a in enumerate(self.agents):
            if a in self.env.agents:
                avail[i, : self._act_n[i]] = 1
            else:
                avail[i, 0] = 1  # finished agents may only no-op
        return avail

    def reset(self, seed=None):
        obs_dict, _ = self.env.reset(seed=seed)
        obs = self._pack(obs_dict)
        return obs, self._state(obs), self._avail()

    def step(self, actions):
        act = {a: int(actions[i]) for i, a in enumerate(self.agents) if a in self.env.agents}
        obs_dict, rew, term, trunc, _ = self.env.step(act)
        obs = self._pack(obs_dict)
        reward = float(np.mean(list(rew.values()))) if rew else 0.0
        done = len(self.env.agents) == 0 or all(term.get(a, False) or trunc.get(a, False) for a in act)
        terminated = done and any(term.values())
        truncated = done and not terminated
        return obs, self._state(obs), self._avail(), reward, terminated, truncated, {}

    def close(self):
        self.env.close()


class SMACEnv(MultiAgentEnv):
    """StarCraft Multi-Agent Challenge adapter, e.g. id "smac:3m". Requires `smac` and StarCraft II."""

    def __init__(self, map_name: str, **kwargs):
        from smac.env import StarCraft2Env

        self.env = StarCraft2Env(map_name=map_name, **kwargs)
        info = self.env.get_env_info()
        self.n_agents = info["n_agents"]
        self.n_actions = info["n_actions"]
        self.obs_dim = info["obs_shape"]
        self.state_dim = info["state_shape"]
        self.episode_limit = info["episode_limit"]

    def _get(self):
        return (np.asarray(self.env.get_obs(), np.float32), np.asarray(self.env.get_state(), np.float32),
                np.asarray(self.env.get_avail_actions(), np.float32))

    def reset(self, seed=None):
        self.env.reset()
        return self._get()

    def step(self, actions):
        reward, terminated, info = self.env.step([int(a) for a in actions])
        truncated = bool(info.get("episode_limit", False))
        terminated = bool(terminated) and not truncated
        obs, state, avail = self._get()
        extra = {"battle_won": float(info.get("battle_won", False))}
        return obs, state, avail, float(reward), terminated, truncated, extra

    def close(self):
        self.env.close()

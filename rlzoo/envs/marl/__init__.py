"""Multi-agent environment registry, vector env and evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import numpy as np

from rlzoo.envs.marl.base import MultiAgentEnv
from rlzoo.envs.marl.matrix_game import MatrixGameEnv
from rlzoo.envs.marl.predator_prey import PredatorPreyEnv
from rlzoo.envs.marl.spread import SpreadEnv

BUILTIN_ENVS = {
    "spread": SpreadEnv,
    "predator_prey": PredatorPreyEnv,
    "matrix_game": MatrixGameEnv,
}


@dataclass
class MAEnvSpec:
    n_agents: int
    obs_dim: int
    state_dim: int
    n_actions: int
    episode_limit: int
    num_envs: int = 1


def make_marl_env(env_cfg) -> MultiAgentEnv:
    """env id: one of BUILTIN_ENVS, "pz:<python.module>" (PettingZoo parallel env) or "smac:<map>"."""
    env_id = env_cfg["id"]
    kwargs = dict(env_cfg.get("kwargs") or {})
    if env_id in BUILTIN_ENVS:
        return BUILTIN_ENVS[env_id](**kwargs)
    if env_id.startswith("pz:"):
        from rlzoo.envs.marl.external import PettingZooEnv
        return PettingZooEnv(env_id[3:], **kwargs)
    if env_id.startswith("smac:"):
        from rlzoo.envs.marl.external import SMACEnv
        return SMACEnv(env_id[5:], **kwargs)
    raise ValueError(f"Unknown MARL env '{env_id}'. Built-ins: {sorted(BUILTIN_ENVS)}, or use 'pz:<module>' / "
                     f"'smac:<map>'")


def spec_from_marl_env(env: MultiAgentEnv, num_envs: int = 1) -> MAEnvSpec:
    return MAEnvSpec(env.n_agents, env.obs_dim, env.state_dim, env.n_actions, env.episode_limit, num_envs)


class MAVecEnv:
    """Steps several multi-agent envs and auto-resets them.

    step(actions (E, n)) -> obs (E,n,d), state (E,s), avail (E,n,A), rewards (E,), terminated (E,),
    truncated (E,), info with "final" [(obs, state, avail) or None] and finished "episodes".
    """

    def __init__(self, env_fns: List[Callable[[], MultiAgentEnv]], seed: int = 0):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        self.spec = spec_from_marl_env(self.envs[0], self.num_envs)
        self.seed = seed
        self._seeded = False
        self._ret = np.zeros(self.num_envs)
        self._len = np.zeros(self.num_envs, dtype=np.int64)

    def reset(self):
        outs = []
        for i, env in enumerate(self.envs):
            outs.append(env.reset(seed=None if self._seeded else self.seed + i))
        self._seeded = True
        self._ret[:] = 0
        self._len[:] = 0
        return tuple(np.stack(x) for x in zip(*outs))

    def step(self, actions):
        E = self.num_envs
        obs, state, avail = [], [], []
        rews, terms, truncs = np.zeros(E, np.float32), np.zeros(E, bool), np.zeros(E, bool)
        final, episodes = [None] * E, []
        for i, env in enumerate(self.envs):
            o, s, av, r, term, trunc, info = env.step(actions[i])
            self._ret[i] += r
            self._len[i] += 1
            rews[i], terms[i], truncs[i] = r, term, trunc
            if term or trunc:
                final[i] = (o, s, av)
                ep = {"return": float(self._ret[i]), "length": float(self._len[i])}
                ep.update({k: float(v) for k, v in info.items() if np.isscalar(v)})
                episodes.append(ep)
                self._ret[i], self._len[i] = 0.0, 0
                o, s, av = env.reset()
            obs.append(o)
            state.append(s)
            avail.append(av)
        return np.stack(obs), np.stack(state), np.stack(avail), rews, terms, truncs, \
            {"final": final, "episodes": episodes}

    def close(self):
        for env in self.envs:
            env.close()


def evaluate_marl(algo, env_fn: Callable[[], MultiAgentEnv], episodes: int, seed: int):
    env = env_fn()
    returns, lengths, extras = [], [], {}
    for ep in range(episodes):
        obs, state, avail = env.reset(seed=seed + ep)
        done, ret, length, info = False, 0.0, 0, {}
        while not done:
            a = algo.act(obs[None], state[None], avail[None], deterministic=True)[0]
            obs, state, avail, r, term, trunc, info = env.step(a)
            ret += r
            length += 1
            done = term or trunc
        returns.append(ret)
        lengths.append(length)
        for k, v in info.items():
            if np.isscalar(v):
                extras.setdefault(k, []).append(float(v))
    env.close()
    out = {"return": float(np.mean(returns)), "return_std": float(np.std(returns)), "length": float(np.mean(lengths))}
    out.update({k: float(np.mean(v)) for k, v in extras.items()})
    return out

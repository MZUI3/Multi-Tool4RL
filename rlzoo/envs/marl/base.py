from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


class MultiAgentEnv:
    """Cooperative multi-agent env interface (SMAC style).

    reset(seed) -> obs (n, obs_dim), state (state_dim,), avail (n, n_actions)
    step(actions) -> obs, state, avail, team_reward (float), terminated, truncated, info (dict of numbers)
    """

    n_agents: int
    obs_dim: int
    state_dim: int
    n_actions: int
    episode_limit: int

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError

    def step(self, actions):
        raise NotImplementedError

    def close(self) -> None:
        pass

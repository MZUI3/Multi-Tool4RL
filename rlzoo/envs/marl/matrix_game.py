"""One-step cooperative matrix games, the classic test for value-factorisation expressiveness."""
from __future__ import annotations

import numpy as np

from rlzoo.envs.marl.base import MultiAgentEnv

PAYOFFS = {
    # Non-monotonic game from QTRAN / QPLEX papers: optimum (0,0)=8 is hard for VDN/QMIX.
    "qtran": [[8, -12, -12], [-12, 0, 0], [-12, 0, 0]],
    # Climbing game (Claus & Boutilier, 1998).
    "climbing": [[11, -30, 0], [-30, 7, 6], [0, 0, 5]],
    # Penalty game.
    "penalty": [[10, 0, -10], [0, 2, 0], [-10, 0, 10]],
}


class MatrixGameEnv(MultiAgentEnv):
    def __init__(self, payoff="qtran", seed=None):
        matrix = PAYOFFS[payoff] if isinstance(payoff, str) else payoff
        self.payoff = np.asarray(matrix, dtype=np.float32)
        if self.payoff.ndim != 2:
            raise ValueError("payoff must be a 2D matrix (2 agents)")
        self.n_agents = 2
        self.n_actions = int(max(self.payoff.shape))
        self.obs_dim = 1
        self.state_dim = 1
        self.episode_limit = 1

    def reset(self, seed=None):
        avail = np.zeros((2, self.n_actions), np.float32)
        avail[0, : self.payoff.shape[0]] = 1
        avail[1, : self.payoff.shape[1]] = 1
        return np.ones((2, 1), np.float32), np.ones(1, np.float32), avail

    def step(self, actions):
        a0, a1 = int(actions[0]), int(actions[1])
        reward = float(self.payoff[a0, a1])
        obs, state, avail = self.reset()
        return obs, state, avail, reward, True, False, {"optimal": float(reward == self.payoff.max())}

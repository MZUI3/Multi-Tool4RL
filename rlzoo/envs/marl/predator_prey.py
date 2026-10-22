"""Grid-world predator-prey with partial observability, pure numpy."""
from __future__ import annotations

import numpy as np

from rlzoo.envs.marl.base import MultiAgentEnv

_MOVES = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int64)  # stay, up, down, left, right


class PredatorPreyEnv(MultiAgentEnv):
    """Predators (agents) capture randomly moving prey on a grid.

    A prey is captured when at least `agents_to_capture` predators are orthogonally adjacent to it, so
    agents must coordinate. Each agent sees a (2*sight+1)^2 window with channels [predators, prey, walls].
    """

    def __init__(self, grid_size: int = 7, n_agents: int = 3, n_prey: int = 2, sight: int = 2,
                 episode_limit: int = 50, capture_reward: float = 10.0, step_penalty: float = 0.1,
                 agents_to_capture: int = 2, seed=None):
        self.G, self.n_agents, self.n_prey, self.sight = int(grid_size), int(n_agents), int(n_prey), int(sight)
        self.episode_limit = int(episode_limit)
        self.capture_reward, self.step_penalty = capture_reward, step_penalty
        self.k = int(agents_to_capture)
        self.n_actions = 5
        view = 2 * self.sight + 1
        self.obs_dim = 3 * view * view + 2
        self.state_dim = 2 * self.n_agents + 3 * self.n_prey
        self.rng = np.random.default_rng(seed)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        cells = self.rng.choice(self.G * self.G, self.n_agents + self.n_prey, replace=False)
        coords = np.stack([cells // self.G, cells % self.G], axis=1)
        self.agents = coords[: self.n_agents].copy()
        self.prey = coords[self.n_agents:].copy()
        self.alive = np.ones(self.n_prey, bool)
        self.t = 0
        return self._obs(), self._state(), self._avail()

    def _inside(self, p) -> bool:
        return 0 <= p[0] < self.G and 0 <= p[1] < self.G

    def _occupied(self):
        occ = {tuple(a) for a in self.agents}
        occ |= {tuple(p) for p, alive in zip(self.prey, self.alive) if alive}
        return occ

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.int64)
        occ = self._occupied()
        for i in self.rng.permutation(self.n_agents):
            target = self.agents[i] + _MOVES[actions[i]]
            if actions[i] != 0 and self._inside(target) and tuple(target) not in occ:
                occ.remove(tuple(self.agents[i]))
                occ.add(tuple(target))
                self.agents[i] = target
        reward = -self.step_penalty
        for j in range(self.n_prey):
            if not self.alive[j]:
                continue
            adjacent = int((np.abs(self.agents - self.prey[j]).sum(1) == 1).sum())
            if adjacent >= self.k:
                self.alive[j] = False
                occ.discard(tuple(self.prey[j]))
                reward += self.capture_reward
        for j in range(self.n_prey):
            if not self.alive[j]:
                continue
            target = self.prey[j] + _MOVES[self.rng.integers(5)]
            if self._inside(target) and tuple(target) not in occ:
                occ.remove(tuple(self.prey[j]))
                occ.add(tuple(target))
                self.prey[j] = target
        self.t += 1
        terminated = not self.alive.any()
        truncated = (not terminated) and self.t >= self.episode_limit
        info = {"captured": float(self.n_prey - self.alive.sum()), "success": float(terminated)}
        return self._obs(), self._state(), self._avail(), float(reward), terminated, truncated, info

    def _obs(self):
        s, G = self.sight, self.G
        grid = np.zeros((3, G + 2 * s, G + 2 * s), np.float32)
        grid[2] = 1.0
        grid[2, s:s + G, s:s + G] = 0.0
        for a in self.agents:
            grid[0, a[0] + s, a[1] + s] = 1.0
        for p, alive in zip(self.prey, self.alive):
            if alive:
                grid[1, p[0] + s, p[1] + s] = 1.0
        obs = []
        for a in self.agents:
            window = grid[:, a[0]:a[0] + 2 * s + 1, a[1]:a[1] + 2 * s + 1].ravel()
            obs.append(np.concatenate([window, a / max(G - 1, 1)]))
        return np.stack(obs).astype(np.float32)

    def _state(self):
        norm = max(self.G - 1, 1)
        prey = np.concatenate([self.prey / norm * self.alive[:, None], self.alive[:, None]], axis=1)
        return np.concatenate([(self.agents / norm).ravel(), prey.ravel()]).astype(np.float32)

    def _avail(self):
        return np.ones((self.n_agents, self.n_actions), np.float32)

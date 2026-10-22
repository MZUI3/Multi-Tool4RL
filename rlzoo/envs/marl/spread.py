"""Cooperative navigation (MPE simple_spread-like), pure numpy."""
from __future__ import annotations

import numpy as np

from rlzoo.envs.marl.base import MultiAgentEnv

_MOVES = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.float32)


class SpreadEnv(MultiAgentEnv):
    """N agents must cover N landmarks while avoiding collisions.

    Team reward per step: -sum_l min_i dist(agent_i, landmark_l) - collision_penalty * #colliding pairs.
    Actions: 0 noop, 1 left, 2 right, 3 down, 4 up (force with damping, like MPE).
    """

    def __init__(self, n_agents: int = 3, n_landmarks: int | None = None, episode_limit: int = 25,
                 agent_size: float = 0.15, collision_penalty: float = 1.0, world_bound: float | None = None,
                 seed: int | None = None):
        self.n_agents = int(n_agents)
        self.n_landmarks = int(n_landmarks or n_agents)
        self.episode_limit = int(episode_limit)
        self.agent_size = agent_size
        self.collision_penalty = collision_penalty
        self.world_bound = world_bound  # clip positions to [-bound, bound] (None = unbounded plane, as in MPE)
        self.n_actions = 5
        self.obs_dim = 4 + 2 * self.n_landmarks + 2 * (self.n_agents - 1)
        self.state_dim = 4 * self.n_agents + 2 * self.n_landmarks
        self.rng = np.random.default_rng(seed)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.pos = self.rng.uniform(-1, 1, (self.n_agents, 2)).astype(np.float32)
        self.vel = np.zeros((self.n_agents, 2), np.float32)
        self.landmarks = self.rng.uniform(-1, 1, (self.n_landmarks, 2)).astype(np.float32)
        self.t = 0
        return self._obs(), self._state(), self._avail()

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.int64)
        self.vel = self.vel * 0.75 + _MOVES[actions] * 0.5
        self.pos = self.pos + self.vel * 0.1
        if self.world_bound is not None:
            clipped = np.clip(self.pos, -self.world_bound, self.world_bound)
            self.vel[clipped != self.pos] = 0.0
            self.pos = clipped
        self.t += 1
        dists = np.linalg.norm(self.pos[:, None] - self.landmarks[None], axis=-1)  # (n, L)
        min_d = dists.min(0)
        reward = -float(min_d.sum())
        pair = np.linalg.norm(self.pos[:, None] - self.pos[None], axis=-1)
        n_coll = int(np.triu(pair < 2 * self.agent_size, k=1).sum())
        reward -= self.collision_penalty * n_coll
        truncated = self.t >= self.episode_limit
        info = {"covered": float((min_d < self.agent_size).mean()), "collisions": float(n_coll)}
        return self._obs(), self._state(), self._avail(), reward, False, truncated, info

    def _obs(self):
        obs = []
        for i in range(self.n_agents):
            rel_l = (self.landmarks - self.pos[i]).ravel()
            others = (np.delete(self.pos, i, axis=0) - self.pos[i]).ravel()
            obs.append(np.concatenate([self.vel[i], self.pos[i], rel_l, others]))
        return np.stack(obs).astype(np.float32)

    def _state(self):
        return np.concatenate([self.pos.ravel(), self.vel.ravel(), self.landmarks.ravel()]).astype(np.float32)

    def _avail(self):
        return np.ones((self.n_agents, self.n_actions), np.float32)

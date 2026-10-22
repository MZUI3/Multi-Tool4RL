from __future__ import annotations

from collections import deque
from typing import Dict, Tuple

import numpy as np


class ReplayBuffer:
    """Ring buffer storing named fields. fields: name -> (shape, dtype)."""

    def __init__(self, capacity: int, fields: Dict[str, Tuple[tuple, type]]):
        self.capacity = int(capacity)
        self.data = {k: np.zeros((self.capacity, *shape), dtype=dtype) for k, (shape, dtype) in fields.items()}
        self.ptr = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(self, **items) -> int:
        idx = self.ptr
        for k, arr in self.data.items():
            arr[idx] = items[k]
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return idx

    def get(self, idx: np.ndarray) -> Dict[str, np.ndarray]:
        return {k: v[idx] for k, v in self.data.items()}

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return idx, self.get(idx)


class SumTree:
    """Binary sum tree with leaves at [capacity, 2*capacity)."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float64)

    @property
    def total(self) -> float:
        return float(self.tree[1])

    def update(self, idx: int, value: float) -> None:
        i = idx + self.capacity
        delta = value - self.tree[i]
        while i >= 1:
            self.tree[i] += delta
            i //= 2

    def leaf(self, idx) -> np.ndarray:
        return self.tree[np.asarray(idx) + self.capacity]

    def find(self, value: float) -> int:
        i = 1
        while i < self.capacity:
            left = 2 * i
            if value <= self.tree[left]:
                i = left
            else:
                value -= self.tree[left]
                i = left + 1
        return i - self.capacity


class PrioritizedReplayBuffer(ReplayBuffer):
    """Proportional prioritized experience replay (Schaul et al., 2016)."""

    def __init__(self, capacity, fields, alpha: float = 0.6, eps: float = 1e-6):
        super().__init__(capacity, fields)
        self.tree = SumTree(self.capacity)
        self.alpha, self.eps = alpha, eps
        self.max_priority = 1.0

    def add(self, **items) -> int:
        idx = super().add(**items)
        self.tree.update(idx, self.max_priority ** self.alpha)
        return idx

    def sample(self, batch_size: int, beta: float = 0.4):
        total = self.tree.total
        segment = total / batch_size
        idx = np.empty(batch_size, dtype=np.int64)
        for i in range(batch_size):
            v = np.random.uniform(segment * i, segment * (i + 1))
            j = self.tree.find(min(v, total - 1e-9))
            idx[i] = min(j, self.size - 1)
        probs = np.maximum(self.tree.leaf(idx) / total, 1e-12)
        weights = (self.size * probs) ** (-beta)
        weights = (weights / weights.max()).astype(np.float32)
        return idx, self.get(idx), weights

    def update_priorities(self, idx: np.ndarray, priorities: np.ndarray) -> None:
        priorities = np.abs(priorities) + self.eps
        for i, p in zip(idx, priorities):
            self.tree.update(int(i), float(p) ** self.alpha)
        self.max_priority = max(self.max_priority, float(priorities.max()))


class NStepAccumulator:
    """Turns 1-step transitions of one environment into n-step transitions."""

    def __init__(self, n: int, gamma: float):
        self.n, self.gamma = int(n), gamma
        self.buf = deque()

    def push(self, obs, action, reward, next_obs, terminated: bool, done: bool):
        self.buf.append((obs, action, float(reward), next_obs, bool(terminated)))
        out = []
        if len(self.buf) == self.n:
            out.append(self._make())
            self.buf.popleft()
        if done:
            while self.buf:
                out.append(self._make())
                self.buf.popleft()
        return out

    def _make(self):
        ret, disc = 0.0, 1.0
        for (_, _, r, _, _) in self.buf:
            ret += disc * r
            disc *= self.gamma
        obs, action = self.buf[0][0], self.buf[0][1]
        _, _, _, next_obs, terminated = self.buf[-1]
        return dict(obs=obs, action=action, reward=ret, next_obs=next_obs, done=float(terminated), discount=disc)


def compute_gae(rewards, values, dones, last_values, gamma: float, lam: float):
    """GAE over arrays shaped (T, ...). dones[t]=1 means the episode ended after step t."""
    T = rewards.shape[0]
    adv = np.zeros_like(rewards, dtype=np.float32)
    last = np.zeros_like(rewards[0], dtype=np.float32)
    for t in reversed(range(T)):
        next_v = last_values if t == T - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    return adv, adv + values

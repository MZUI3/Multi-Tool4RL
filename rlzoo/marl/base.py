"""Shared pieces for cooperative MARL algorithms (discrete actions, team reward, CTDE)."""
from __future__ import annotations

import numpy as np
import torch

from rlzoo.common.base import Algorithm
from rlzoo.common.buffers import ReplayBuffer


class MAAlgorithm(Algorithm):
    kind = "marl"
    default_hparams = dict(gamma=0.99, hidden_sizes=[64, 64], activation="relu",
                           agent_id=True)  # append a one-hot agent id to each agent's observation

    def __init__(self, spec, cfg, device, logger=None):
        super().__init__(spec, cfg, device, logger)
        self.n = spec.n_agents
        self.A = spec.n_actions
        self.state_dim = spec.state_dim
        self.obs_dim = spec.obs_dim + (self.n if self.hp["agent_id"] else 0)  # actor input size
        self.eye = torch.eye(self.n, device=device)

    def agent_inputs(self, obs: torch.Tensor) -> torch.Tensor:
        """(..., n, obs_dim) -> (..., n, obs_dim [+ n])"""
        if not self.hp["agent_id"]:
            return obs
        return torch.cat([obs, self.eye.expand(*obs.shape[:-2], self.n, self.n)], -1)

    def state_inputs(self, state: torch.Tensor) -> torch.Tensor:
        """(..., state_dim) -> per-agent centralized input (..., n, state_dim [+ n])"""
        s = state.unsqueeze(-2).expand(*state.shape[:-1], self.n, state.shape[-1])
        if not self.hp["agent_id"]:
            return s
        return torch.cat([s, self.eye.expand(*state.shape[:-1], self.n, self.n)], -1)


class MAOffPolicyAlgo(MAAlgorithm):
    default_hparams = dict(buffer_size=50_000, batch_size=64, learning_starts=1_000, train_freq=1, gradient_steps=1)

    def make_buffer(self):
        sp = self.spec
        n, d, s, A = sp.n_agents, sp.obs_dim, sp.state_dim, sp.n_actions
        fields = {
            "obs": ((n, d), np.float32), "state": ((s,), np.float32), "avail": ((n, A), np.float32),
            "actions": ((n,), np.int64), "reward": ((), np.float32),
            "next_obs": ((n, d), np.float32), "next_state": ((s,), np.float32), "next_avail": ((n, A), np.float32),
            "done": ((), np.float32),
        }
        return ReplayBuffer(int(self.hp["buffer_size"]), fields)

    def update(self):
        raise NotImplementedError

    def train(self, venv, total_steps):
        hp = self.hp
        obs, state, avail = venv.reset()
        it = 0
        while self.global_step < total_steps:
            actions = self.act(obs, state, avail, deterministic=False)
            nobs, nstate, navail, rewards, terms, truncs, info = venv.step(actions)
            for e in range(venv.num_envs):
                done = bool(terms[e] or truncs[e])
                o2, s2, a2 = info["final"][e] if done else (nobs[e], nstate[e], navail[e])
                self.buffer.add(obs=obs[e], state=state[e], avail=avail[e], actions=actions[e], reward=rewards[e],
                                next_obs=o2, next_state=s2, next_avail=a2, done=float(terms[e]))
            obs, state, avail = nobs, nstate, navail
            self.global_step += venv.num_envs
            it += 1
            self.record_episodes(info["episodes"])
            if (self.global_step >= hp["learning_starts"] and it % int(hp["train_freq"]) == 0
                    and len(self.buffer) >= hp["batch_size"]):
                for _ in range(int(hp["gradient_steps"])):
                    self.train_metrics = self.update()
            self.maybe_log()


class MAOnPolicyAlgo(MAAlgorithm):
    """Rollout collection for on-policy MARL. Subclasses implement policy_step / bootstrap_value / update."""

    default_hparams = dict(n_steps=100, gae_lambda=0.95)

    def policy_step(self, obs, state, avail):
        """-> actions (E,n) long, logp (E,n), values (E,n)"""
        raise NotImplementedError

    def bootstrap_value(self, obs, state, avail):
        """-> value estimate (E,n) used for truncated episodes and the rollout tail."""
        raise NotImplementedError

    def collect(self, venv, carry):
        obs, state, avail = carry
        gamma, E = self.hp["gamma"], venv.num_envs
        keys = ("obs", "state", "avail", "actions", "logp", "values", "rewards", "dones")
        buf = {k: [] for k in keys}
        episodes = []
        for _ in range(int(self.hp["n_steps"])):
            with torch.no_grad():
                actions, logp, values = self.policy_step(self.t(obs), self.t(state), self.t(avail))
            a_np = actions.cpu().numpy()
            nobs, nstate, navail, rew, terms, truncs, info = venv.step(a_np)
            rewards = np.repeat(rew[:, None], self.n, axis=1).astype(np.float32)
            for e in range(E):
                if truncs[e] and not terms[e]:
                    fo, fs, fa = info["final"][e]
                    with torch.no_grad():
                        v = self.bootstrap_value(self.t(fo[None]), self.t(fs[None]), self.t(fa[None]))
                    rewards[e] += gamma * v[0].cpu().numpy()
            for k, v in (("obs", obs), ("state", state), ("avail", avail), ("actions", a_np),
                         ("logp", logp.cpu().numpy()), ("values", values.cpu().numpy()), ("rewards", rewards),
                         ("dones", np.logical_or(terms, truncs).astype(np.float32))):
                buf[k].append(v)
            episodes.extend(info["episodes"])
            obs, state, avail = nobs, nstate, navail
        with torch.no_grad():
            last = self.bootstrap_value(self.t(obs), self.t(state), self.t(avail)).cpu().numpy()
        data = {k: np.stack(v) for k, v in buf.items()}  # (T, E, ...)
        data["last_values"] = last
        return data, (obs, state, avail), episodes

    def update(self, data):
        raise NotImplementedError

    def train(self, venv, total_steps):
        carry = venv.reset()
        while self.global_step < total_steps:
            data, carry, episodes = self.collect(venv, carry)
            self.global_step += int(self.hp["n_steps"]) * venv.num_envs
            self.record_episodes(episodes)
            self.train_metrics = self.update(data)
            self.maybe_log()


def flatten_time(data, keys):
    """(T, E, ...) numpy arrays -> (T*E, ...)."""
    return {k: data[k].reshape(-1, *data[k].shape[2:]) for k in keys}

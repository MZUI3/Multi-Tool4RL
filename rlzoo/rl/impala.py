"""IMPALA / V-trace (Espeholt et al., 2018), synchronous variant.

Rollouts are collected with a *behaviour* copy of the policy that is refreshed only every `policy_lag`
learner updates, so the learner is off-policy by a few updates exactly as with asynchronous actors; the
V-trace importance weights (rho, c) correct for that. With `policy_lag=0` it degenerates to on-policy
actor-critic with V-trace targets.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from rlzoo.rl.base import OnPolicyAlgo


class IMPALA(OnPolicyAlgo):
    name = "impala"
    default_hparams = dict(lr=6e-4, n_steps=20, rho_bar=1.0, c_bar=1.0, ent_coef=0.01, vf_coef=0.5, max_grad_norm=40.0,
                           policy_lag=2, rms_alpha=0.99, rms_eps=0.01, activation="relu")

    def __init__(self, spec, cfg, device, logger=None, accel=None):
        super().__init__(spec, cfg, device, logger, accel)
        hp = self.hp
        self.behavior = copy.deepcopy(self.ac)
        self.opt = torch.optim.RMSprop(self.ac.parameters(), lr=hp["lr"], alpha=hp["rms_alpha"], eps=hp["rms_eps"])
        self.n_updates = 0

    def collect(self, venv, obs):
        E, T, gamma = venv.num_envs, int(self.hp["n_steps"]), self.hp["gamma"]
        buf = {k: [] for k in ("obs", "actions", "logp", "rewards", "dones")}
        episodes = []
        for _ in range(T):
            with torch.no_grad():
                dist = self.behavior.dist(self.t(obs))
                action = dist.sample()
                logp = dist.log_prob(action)
            a_np = action.cpu().numpy()
            next_obs, rewards, terms, truncs, info = venv.step(a_np)
            rewards = rewards.astype(np.float32).copy()
            for i in range(E):
                if truncs[i] and not terms[i]:
                    with torch.no_grad():
                        rewards[i] += gamma * self.ac.value(self.t(info["final_obs"][i][None])).item()
            buf["obs"].append(obs)
            buf["actions"].append(a_np)
            buf["logp"].append(logp.cpu().numpy())
            buf["rewards"].append(rewards)
            buf["dones"].append(np.logical_or(terms, truncs).astype(np.float32))
            episodes.extend(info["episodes"])
            obs = next_obs
        data = {k: np.stack(v) for k, v in buf.items()}
        data["last_obs"] = obs
        return data, obs, episodes

    def vtrace_update(self, data):
        hp = self.hp
        T, E = data["rewards"].shape
        b = self.to_torch({k: v for k, v in data.items() if k != "last_obs"})
        obs = b["obs"].reshape(T * E, -1)
        dist = self.ac.dist(obs)
        logp = dist.log_prob(b["actions"].reshape(T * E, *b["actions"].shape[2:])).view(T, E)
        entropy = dist.entropy().mean()
        values = self.ac.value(obs).view(T, E)
        with torch.no_grad():
            bootstrap = self.ac.value(self.t(data["last_obs"]))
            v_next = torch.cat([values[1:], bootstrap[None]], 0).detach()
            v_det = values.detach()
            log_rho = logp.detach() - b["logp"]
            rhos = log_rho.exp().clamp(max=hp["rho_bar"])
            cs = log_rho.exp().clamp(max=hp["c_bar"])
            nonterminal = 1.0 - b["dones"]
            deltas = rhos * (b["rewards"] + hp["gamma"] * nonterminal * v_next - v_det)
            vs = torch.zeros_like(v_det)
            acc = torch.zeros(E, device=self.device)
            for t in reversed(range(T)):
                acc = deltas[t] + hp["gamma"] * nonterminal[t] * cs[t] * acc
                vs[t] = v_det[t] + acc
            vs_next = torch.cat([vs[1:], bootstrap[None]], 0)
            pg_adv = rhos * (b["rewards"] + hp["gamma"] * nonterminal * vs_next - v_det)
        policy_loss = -(logp * pg_adv).mean()
        value_loss = 0.5 * F.mse_loss(values, vs)
        loss = policy_loss + hp["vf_coef"] * value_loss - hp["ent_coef"] * entropy
        self.clip_and_step(self.opt, loss, self.ac.parameters(), hp["max_grad_norm"])
        return {"policy_loss": policy_loss.item(), "value_loss": value_loss.item(), "entropy": entropy.item(),
                "mean_rho": rhos.mean().item()}

    def train(self, venv, total_steps):
        hp = self.hp
        obs = venv.reset()
        while self.global_step < total_steps:
            data, obs, episodes = self.collect(venv, obs)
            self.global_step += int(hp["n_steps"]) * venv.num_envs * self.step_scale
            self.record_episodes(episodes)
            self.train_metrics = self.vtrace_update(data)
            self.n_updates += 1
            if int(hp["policy_lag"]) <= 0 or self.n_updates % int(hp["policy_lag"]) == 0:
                self.behavior.load_state_dict(self.ac.state_dict())
            self.maybe_log()

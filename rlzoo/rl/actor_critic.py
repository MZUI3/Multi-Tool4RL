"""REINFORCE, A2C and A3C."""
from __future__ import annotations

import copy
import threading

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rlzoo.rl.base import OnPolicyAlgo


class REINFORCE(OnPolicyAlgo):
    """Monte-Carlo policy gradient (Williams, 1992) with an optional learned value baseline."""

    name = "reinforce"
    default_hparams = dict(lr=1e-3, episodes_per_update=4, use_baseline=True, max_grad_norm=0.0)

    def __init__(self, spec, cfg, device, logger=None):
        super().__init__(spec, cfg, device, logger)
        self.opt = torch.optim.Adam(self.ac.parameters(), lr=self.hp["lr"])

    def train(self, venv, total_steps):
        gamma = self.hp["gamma"]
        obs = venv.reset()
        trajs = [[] for _ in range(venv.num_envs)]
        completed = []
        while self.global_step < total_steps:
            actions = self.act(obs)
            next_obs, rewards, terms, truncs, info = venv.step(actions)
            for i in range(venv.num_envs):
                trajs[i].append((obs[i], actions[i], rewards[i]))
                if terms[i] or truncs[i]:
                    completed.append(trajs[i])
                    trajs[i] = []
            obs = next_obs
            self.global_step += venv.num_envs
            self.record_episodes(info["episodes"])
            if len(completed) >= self.hp["episodes_per_update"]:
                self.train_metrics = self.update_episodes(completed, gamma)
                completed = []
                if any(trajs):  # drop partial (now off-policy) episodes to stay on-policy
                    trajs = [[] for _ in range(venv.num_envs)]
                    obs = venv.reset()
            self.maybe_log()

    def update_episodes(self, episodes, gamma):
        obs, acts, rets = [], [], []
        for ep in episodes:
            g, returns = 0.0, []
            for (_, _, r) in reversed(ep):
                g = r + gamma * g
                returns.append(g)
            returns.reverse()
            obs += [o for (o, _, _) in ep]
            acts += [a for (_, a, _) in ep]
            rets += returns
        b = self.to_torch({"obs": np.stack(obs), "actions": np.stack(acts), "returns": np.asarray(rets, np.float32)})
        dist = self.ac.dist(b["obs"])
        logp = dist.log_prob(b["actions"])
        entropy = dist.entropy().mean()
        adv, value_loss = b["returns"], torch.zeros((), device=self.device)
        if self.hp["use_baseline"]:
            v = self.ac.value(b["obs"])
            value_loss = F.mse_loss(v, b["returns"])
            adv = b["returns"] - v.detach()
        if self.hp["normalize_advantage"] and adv.numel() > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        policy_loss = -(logp * adv).mean()
        loss = policy_loss + self.hp["vf_coef"] * value_loss - self.hp["ent_coef"] * entropy
        self.clip_and_step(self.opt, loss, self.ac.parameters(), self.hp["max_grad_norm"])
        return {"policy_loss": policy_loss.item(), "value_loss": value_loss.item(), "entropy": entropy.item()}


class A2C(OnPolicyAlgo):
    """Synchronous advantage actor-critic (Mnih et al., 2016; OpenAI baselines A2C)."""

    name = "a2c"
    default_hparams = dict(lr=7e-4, n_steps=5, gae_lambda=1.0, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
                           normalize_advantage=False, rms_alpha=0.99, rms_eps=1e-5)

    def __init__(self, spec, cfg, device, logger=None):
        super().__init__(spec, cfg, device, logger)
        hp = self.hp
        self.opt = torch.optim.RMSprop(self.ac.parameters(), lr=hp["lr"], alpha=hp["rms_alpha"], eps=hp["rms_eps"])

    def update(self, b):
        loss, metrics = self.actor_critic_loss(self.ac, b)
        self.clip_and_step(self.opt, loss, self.ac.parameters(), self.hp["max_grad_norm"])
        return metrics


class A3C(OnPolicyAlgo):
    """Asynchronous advantage actor-critic (Mnih et al., 2016) with Python threads and a shared optimizer.

    Each worker owns an env and a local copy of the network, computes n-step gradients and applies them to
    the shared global network.
    """

    name = "a3c"
    default_hparams = dict(lr=7e-4, n_workers=4, n_steps=5, gae_lambda=1.0, ent_coef=0.01, vf_coef=0.5,
                           max_grad_norm=40.0, normalize_advantage=False)

    def __init__(self, spec, cfg, device, logger=None):
        super().__init__(spec, cfg, device, logger)
        self.opt = torch.optim.Adam(self.ac.parameters(), lr=self.hp["lr"])
        self.lock = threading.Lock()
        self._stop = False
        self._errors = []

    def train(self, venv, total_steps):
        if self.make_venv is None:
            raise RuntimeError("A3C needs algo.make_venv to create per-worker environments")
        workers = [threading.Thread(target=self._worker, args=(w, total_steps), daemon=True)
                   for w in range(int(self.hp["n_workers"]))]
        for w in workers:
            w.start()
        for w in workers:
            w.join()
        if self._errors:
            raise self._errors[0]

    def _worker(self, wid, total_steps):
        try:
            env = self.make_venv(1, int(self.cfg["seed"]) + 1000 * (wid + 1))
            with self.lock:
                local = copy.deepcopy(self.ac)
            obs = env.reset()
            while not self._stop and self.global_step < total_steps:
                with self.lock:
                    local.load_state_dict(self.ac.state_dict())
                batch, obs, episodes = self.collect_rollout(local, env, obs, self.hp["n_steps"])
                loss, metrics = self.actor_critic_loss(local, self.to_torch(batch))
                local.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(local.parameters(), self.hp["max_grad_norm"])
                with self.lock:
                    for gp, lp in zip(self.ac.parameters(), local.parameters()):
                        gp.grad = None if lp.grad is None else lp.grad.clone()
                    self.opt.step()
                    self.global_step += int(self.hp["n_steps"])
                    self.record_episodes(episodes)
                    self.train_metrics = metrics
                    self.maybe_log()
            env.close()
        except Exception as e:  # surface worker errors in the main thread
            self._errors.append(e)
            self._stop = True

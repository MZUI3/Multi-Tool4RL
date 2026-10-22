"""Training loops shared by single-agent off-policy and on-policy algorithms."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from rlzoo.common.base import Algorithm
from rlzoo.common.buffers import ReplayBuffer, compute_gae
from rlzoo.common.nets import ActorCritic


class RLAlgorithm(Algorithm):
    kind = "rl"
    default_hparams = dict(gamma=0.99, hidden_sizes=[64, 64], activation="relu")


class OffPolicyAlgo(RLAlgorithm):
    default_hparams = dict(
        buffer_size=100_000,
        batch_size=64,
        learning_starts=1_000,  # env steps before the first gradient update
        train_freq=1,  # vector-env steps between update rounds
        gradient_steps=1,  # gradient updates per round
        random_steps=0,  # env steps with uniformly random actions
    )

    def buffer_fields(self):
        spec = self.spec
        act = ((), np.int64) if spec.discrete else ((spec.act_dim,), np.float32)
        return {
            "obs": ((spec.obs_dim,), np.float32),
            "action": act,
            "reward": ((), np.float32),
            "next_obs": ((spec.obs_dim,), np.float32),
            "done": ((), np.float32),
            "discount": ((), np.float32),
        }

    def make_buffer(self):
        return ReplayBuffer(int(self.hp["buffer_size"]), self.buffer_fields())

    def store(self, env_idx, obs, action, reward, next_obs, terminated, done):
        self.buffer.add(obs=obs, action=action, reward=reward, next_obs=next_obs, done=float(terminated),
                        discount=self.hp["gamma"])

    def update(self):
        raise NotImplementedError

    def train(self, venv, total_steps):
        hp = self.hp
        obs = venv.reset()
        it = 0
        while self.global_step < total_steps:
            if self.global_step < hp["random_steps"]:
                actions = venv.sample_actions()
            else:
                actions = self.act(obs, deterministic=False)
            next_obs, rewards, terms, truncs, info = venv.step(actions)
            for i in range(venv.num_envs):
                done = bool(terms[i] or truncs[i])
                real_next = info["final_obs"][i] if done else next_obs[i]
                self.store(i, obs[i], actions[i], rewards[i], real_next, bool(terms[i]), done)
            obs = next_obs
            self.global_step += venv.num_envs
            it += 1
            self.record_episodes(info["episodes"])
            if (self.global_step >= hp["learning_starts"] and it % int(hp["train_freq"]) == 0
                    and len(self.buffer) >= hp["batch_size"]):
                for _ in range(int(hp["gradient_steps"])):
                    self.train_metrics = self.update()
            self.maybe_log()


class OnPolicyAlgo(RLAlgorithm):
    default_hparams = dict(
        lr=3e-4,
        n_steps=128,  # rollout length per env
        gae_lambda=0.95,
        activation="tanh",
        normalize_advantage=True,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        log_std_init=0.0,
        ortho_init=True,
    )

    def __init__(self, spec, cfg, device, logger=None):
        super().__init__(spec, cfg, device, logger)
        hp = self.hp
        self.ac = ActorCritic(spec.obs_dim, spec, hp["hidden_sizes"], hp["activation"], hp["log_std_init"],
                              hp["ortho_init"]).to(device)

    def networks(self):
        return {"ac": self.ac}

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        dist = self.ac.dist(self.t(obs))
        a = self.ac.mode(dist) if deterministic else dist.sample()
        return a.cpu().numpy()

    def collect_rollout(self, ac, venv, obs, n_steps):
        """Collect n_steps x num_envs transitions with policy `ac`; returns (flat batch, last obs, episodes)."""
        gamma, E = self.hp["gamma"], venv.num_envs
        keys = ("obs", "actions", "logp", "values", "rewards", "dones")
        buf = {k: [] for k in keys}
        episodes = []
        for _ in range(int(n_steps)):
            with torch.no_grad():
                obs_t = self.t(obs)
                dist = ac.dist(obs_t)
                action = dist.sample()
                logp = dist.log_prob(action)
                value = ac.value(obs_t)
            a_np = action.cpu().numpy()
            next_obs, rewards, terms, truncs, info = venv.step(a_np)
            rewards = rewards.astype(np.float32).copy()
            for i in range(E):  # bootstrap time-limit truncations
                if truncs[i] and not terms[i]:
                    with torch.no_grad():
                        rewards[i] += gamma * ac.value(self.t(info["final_obs"][i][None])).item()
            buf["obs"].append(obs)
            buf["actions"].append(a_np)
            buf["logp"].append(logp.cpu().numpy())
            buf["values"].append(value.cpu().numpy())
            buf["rewards"].append(rewards)
            buf["dones"].append(np.logical_or(terms, truncs).astype(np.float32))
            episodes.extend(info["episodes"])
            obs = next_obs
        with torch.no_grad():
            last_value = ac.value(self.t(obs)).cpu().numpy()
        data = {k: np.stack(v) for k, v in buf.items()}
        adv, ret = compute_gae(data["rewards"], data["values"], data["dones"], last_value, gamma,
                               self.hp["gae_lambda"])
        data["advantages"], data["returns"] = adv, ret
        T = data["rewards"].shape[0]
        batch = {k: v.reshape(T * E, *v.shape[2:]) for k, v in data.items() if k not in ("rewards", "dones")}
        return batch, obs, episodes

    def actor_critic_loss(self, ac, b):
        """Vanilla advantage actor-critic loss (A2C / A3C)."""
        dist = ac.dist(b["obs"])
        logp = dist.log_prob(b["actions"])
        entropy = dist.entropy().mean()
        adv = b["advantages"]
        if self.hp["normalize_advantage"] and adv.numel() > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        policy_loss = -(logp * adv).mean()
        value_loss = F.mse_loss(ac.value(b["obs"]), b["returns"])
        loss = policy_loss + self.hp["vf_coef"] * value_loss - self.hp["ent_coef"] * entropy
        return loss, {"policy_loss": policy_loss.item(), "value_loss": value_loss.item(), "entropy": entropy.item()}

    def update(self, batch):
        raise NotImplementedError

    def train(self, venv, total_steps):
        obs = venv.reset()
        while self.global_step < total_steps:
            batch, obs, episodes = self.collect_rollout(self.ac, venv, obs, self.hp["n_steps"])
            self.global_step += int(self.hp["n_steps"]) * venv.num_envs
            self.record_episodes(episodes)
            self.train_metrics = self.update(self.to_torch(batch))
            self.maybe_log()

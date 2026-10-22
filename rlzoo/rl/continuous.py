"""Off-policy continuous control: DDPG, TD3, SAC."""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rlzoo.common.nets import DeterministicActor, QCritic, SquashedGaussianActor
from rlzoo.common.utils import soft_update
from rlzoo.rl.base import OffPolicyAlgo


class _ContinuousBase(OffPolicyAlgo):
    default_hparams = dict(actor_lr=1e-3, critic_lr=1e-3, tau=0.005, buffer_size=200_000, batch_size=256,
                           learning_starts=1_000, random_steps=1_000, hidden_sizes=[256, 256], n_critics=1,
                           max_grad_norm=0.0)

    def __init__(self, spec, cfg, device, logger=None):
        super().__init__(spec, cfg, device, logger)
        if spec.discrete:
            raise TypeError(f"{self.name} requires a continuous (Box) action space")
        hp = self.hp
        self.low = torch.as_tensor(spec.act_low, device=device)
        self.high = torch.as_tensor(spec.act_high, device=device)
        self.scale = (self.high - self.low) / 2
        self.critics = nn.ModuleList([QCritic(spec.obs_dim, spec.act_dim, hp["hidden_sizes"], hp["activation"])
                                      for _ in range(int(hp["n_critics"]))]).to(device)
        self.critics_target = copy.deepcopy(self.critics)
        self.critic_opt = torch.optim.Adam(self.critics.parameters(), lr=hp["critic_lr"])
        self.buffer = self.make_buffer()
        self.n_updates = 0

    def critic_step(self, b, target):
        loss = sum(F.mse_loss(c(b["obs"], b["action"]), target) for c in self.critics)
        self.clip_and_step(self.critic_opt, loss, self.critics.parameters(), self.hp["max_grad_norm"])
        return loss.item()


class DDPG(_ContinuousBase):
    """Lillicrap et al., 2016. TD3 reuses this with twin critics, delayed actor and target smoothing."""

    name = "ddpg"
    default_hparams = dict(exploration_noise=0.1, policy_delay=1, target_policy_noise=0.0, target_noise_clip=0.5)

    def __init__(self, spec, cfg, device, logger=None):
        super().__init__(spec, cfg, device, logger)
        hp = self.hp
        self.actor = DeterministicActor(spec.obs_dim, spec.act_dim, hp["hidden_sizes"], hp["activation"],
                                        spec.act_low, spec.act_high).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=hp["actor_lr"])
        self._actor_loss = 0.0

    def networks(self):
        return {"actor": self.actor, "critics": self.critics, "actor_target": self.actor_target,
                "critics_target": self.critics_target}

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        a = self.actor(self.t(obs)).cpu().numpy()
        if not deterministic:
            scale = (self.spec.act_high - self.spec.act_low) / 2
            a = a + np.random.normal(0, self.hp["exploration_noise"], a.shape) * scale
        return np.clip(a, self.spec.act_low, self.spec.act_high)

    def update(self):
        hp = self.hp
        _, batch = self.buffer.sample(int(hp["batch_size"]))
        b = self.to_torch(batch)
        with torch.no_grad():
            next_a = self.actor_target(b["next_obs"])
            if hp["target_policy_noise"] > 0:
                noise = (torch.randn_like(next_a) * hp["target_policy_noise"]).clamp(
                    -hp["target_noise_clip"], hp["target_noise_clip"]) * self.scale
                next_a = torch.max(torch.min(next_a + noise, self.high), self.low)
            next_q = torch.min(torch.stack([c(b["next_obs"], next_a) for c in self.critics_target]), 0).values
            target = b["reward"] + b["discount"] * (1 - b["done"]) * next_q
        critic_loss = self.critic_step(b, target)
        self.n_updates += 1
        if self.n_updates % int(hp["policy_delay"]) == 0:
            actor_loss = -self.critics[0](b["obs"], self.actor(b["obs"])).mean()
            self.clip_and_step(self.actor_opt, actor_loss, self.actor.parameters(), hp["max_grad_norm"])
            self._actor_loss = actor_loss.item()
            soft_update(self.actor_target, self.actor, hp["tau"])
            soft_update(self.critics_target, self.critics, hp["tau"])
        return {"critic_loss": critic_loss, "actor_loss": self._actor_loss}


class TD3(DDPG):
    """Fujimoto et al., 2018."""

    name = "td3"
    default_hparams = dict(n_critics=2, policy_delay=2, target_policy_noise=0.2, target_noise_clip=0.5)


class SAC(_ContinuousBase):
    """Soft Actor-Critic with automatic entropy tuning (Haarnoja et al., 2018)."""

    name = "sac"
    default_hparams = dict(actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4, n_critics=2, alpha=0.2, autotune=True,
                           target_entropy=None)

    def __init__(self, spec, cfg, device, logger=None):
        super().__init__(spec, cfg, device, logger)
        hp = self.hp
        self.actor = SquashedGaussianActor(spec.obs_dim, spec.act_dim, hp["hidden_sizes"], hp["activation"],
                                           spec.act_low, spec.act_high).to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=hp["actor_lr"])
        self.target_entropy = hp["target_entropy"] if hp["target_entropy"] is not None else -float(spec.act_dim)
        self.log_alpha = torch.tensor(np.log(hp["alpha"]), dtype=torch.float32, device=device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=hp["alpha_lr"])

    @property
    def alpha(self):
        return self.log_alpha.exp().detach()

    def networks(self):
        return {"actor": self.actor, "critics": self.critics, "critics_target": self.critics_target}

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        a, _ = self.actor(self.t(obs), deterministic=deterministic)
        return a.cpu().numpy()

    def update(self):
        hp = self.hp
        _, batch = self.buffer.sample(int(hp["batch_size"]))
        b = self.to_torch(batch)
        with torch.no_grad():
            next_a, next_logp = self.actor(b["next_obs"])
            next_q = torch.min(torch.stack([c(b["next_obs"], next_a) for c in self.critics_target]), 0).values
            target = b["reward"] + b["discount"] * (1 - b["done"]) * (next_q - self.alpha * next_logp)
        critic_loss = self.critic_step(b, target)
        a, logp = self.actor(b["obs"])
        q = torch.min(torch.stack([c(b["obs"], a) for c in self.critics]), 0).values
        actor_loss = (self.alpha * logp - q).mean()
        self.clip_and_step(self.actor_opt, actor_loss, self.actor.parameters(), hp["max_grad_norm"])
        if hp["autotune"]:
            alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
        soft_update(self.critics_target, self.critics, hp["tau"])
        return {"critic_loss": critic_loss, "actor_loss": actor_loss.item(), "alpha": self.alpha.item()}

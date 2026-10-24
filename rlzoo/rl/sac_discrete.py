"""Soft Actor-Critic for discrete actions (Christodoulou, 2019)."""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rlzoo.common.nets import CategoricalActor, DiscreteQCritic
from rlzoo.common.utils import soft_update
from rlzoo.rl.base import OffPolicyAlgo


class SACDiscrete(OffPolicyAlgo):
    """Categorical policy + twin Q(s, .) critics. Expectations over actions are computed exactly, so no
    reparameterisation is needed; the entropy target is a fraction of the maximum entropy log|A|."""

    name = "sac_discrete"
    default_hparams = dict(actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4, tau=0.005, alpha=0.2, autotune=True,
                           target_entropy_ratio=0.98, n_critics=2, batch_size=64, buffer_size=100_000,
                           learning_starts=1_000, max_grad_norm=0.0)

    def __init__(self, spec, cfg, device, logger=None, accel=None):
        super().__init__(spec, cfg, device, logger, accel)
        if not spec.discrete:
            raise TypeError("sac_discrete requires a discrete action space (use sac for continuous actions)")
        hp = self.hp
        self.actor = CategoricalActor(spec.obs_dim, spec.n_actions, hp["hidden_sizes"], hp["activation"],
                                      encoder=self.make_encoder()).to(device)
        self.critics = nn.ModuleList([DiscreteQCritic(spec.obs_dim, spec.n_actions, hp["hidden_sizes"], hp["activation"],
                                                      encoder=self.make_encoder()) for _ in range(int(hp["n_critics"]))]).to(device)
        self.critics_target = copy.deepcopy(self.critics)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=hp["actor_lr"])
        self.critic_opt = torch.optim.Adam(self.critics.parameters(), lr=hp["critic_lr"])
        self.target_entropy = -np.log(1.0 / spec.n_actions) * hp["target_entropy_ratio"]
        self.log_alpha = torch.tensor(np.log(hp["alpha"]), dtype=torch.float32, device=device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=hp["alpha_lr"])
        self.buffer = self.make_buffer()

    @property
    def alpha(self):
        return self.log_alpha.exp().detach()

    def networks(self):
        return {"actor": self.actor, "critics": self.critics, "critics_target": self.critics_target}

    def min_q(self, critics, obs):
        return torch.min(torch.stack([c(obs) for c in critics]), 0).values

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        logits = self.actor(self.t(obs))
        a = logits.argmax(-1) if deterministic else torch.distributions.Categorical(logits=logits).sample()
        return a.cpu().numpy()

    @torch.no_grad()
    def explain(self, obs):
        obs_t = self.t(np.asarray(obs)[None])
        return {"probs": F.softmax(self.actor(obs_t), -1)[0].cpu().numpy(), "q": self.min_q(self.critics, obs_t)[0].cpu().numpy()}

    def update(self):
        hp = self.hp
        _, batch = self.buffer.sample(int(hp["batch_size"]))
        b = self.to_torch(batch)
        with torch.no_grad():
            logits_next = self.actor(b["next_obs"])
            pi_next, logp_next = F.softmax(logits_next, -1), F.log_softmax(logits_next, -1)
            v_next = (pi_next * (self.min_q(self.critics_target, b["next_obs"]) - self.alpha * logp_next)).sum(-1)
            target = b["reward"] + b["discount"] * (1 - b["done"]) * v_next
        critic_loss = sum(F.mse_loss(c(b["obs"]).gather(1, b["action"][:, None]).squeeze(1), target) for c in self.critics)
        self.clip_and_step(self.critic_opt, critic_loss, self.critics.parameters(), hp["max_grad_norm"])

        logits = self.actor(b["obs"])
        pi, logp = F.softmax(logits, -1), F.log_softmax(logits, -1)
        with torch.no_grad():
            q = self.min_q(self.critics, b["obs"])
        actor_loss = (pi * (self.alpha * logp - q)).sum(-1).mean()
        self.clip_and_step(self.actor_opt, actor_loss, self.actor.parameters(), hp["max_grad_norm"])
        entropy = -(pi * logp).sum(-1).detach()
        if hp["autotune"]:
            alpha_loss = (self.log_alpha * (entropy - self.target_entropy)).mean()
            self.clip_and_step(self.alpha_opt, alpha_loss, [self.log_alpha])
        soft_update(self.critics_target, self.critics, hp["tau"])
        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item(), "alpha": self.alpha.item(),
                "entropy": entropy.mean().item()}

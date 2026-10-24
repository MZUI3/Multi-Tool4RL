"""Multi-agent SAC for discrete actions (CTDE): shared categorical actors, centralized per-agent critics
Q_i(s, o_i, i, u_-i, .) as in COMA, soft targets with exact expectations over the agent's own action."""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from rlzoo.common.nets import mlp
from rlzoo.common.utils import masked_logits, soft_update
from rlzoo.marl.base import MAOffPolicyAlgo


class MASAC(MAOffPolicyAlgo):
    name = "masac"
    default_hparams = dict(actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4, alpha=0.05, autotune=True, target_entropy_ratio=0.5,
                           tau=0.005, n_critics=2, batch_size=128, buffer_size=100_000, learning_starts=2_000,
                           critic_hidden=[128, 128], max_grad_norm=10.0)

    def __init__(self, spec, cfg, device, logger=None, accel=None):
        super().__init__(spec, cfg, device, logger, accel)
        hp = self.hp
        self.actor = mlp(self.obs_dim, self.A, hp["hidden_sizes"], hp["activation"]).to(device)
        critic_in = self.state_dim + spec.obs_dim + self.n + self.n * self.A
        self.critics = nn.ModuleList([mlp(critic_in, self.A, hp["critic_hidden"], hp["activation"])
                                      for _ in range(int(hp["n_critics"]))]).to(device)
        self.critics_target = copy.deepcopy(self.critics)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=hp["actor_lr"])
        self.critic_opt = torch.optim.Adam(self.critics.parameters(), lr=hp["critic_lr"])
        self.target_entropy = np.log(self.A) * hp["target_entropy_ratio"]
        self.log_alpha = torch.tensor(np.log(hp["alpha"]), dtype=torch.float32, device=device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=hp["alpha_lr"])
        self.others_mask = (1 - self.eye).repeat_interleave(self.A, dim=1)
        self.buffer = self.make_buffer()

    @property
    def alpha(self):
        return self.log_alpha.exp().detach()

    def networks(self):
        return {"actor": self.actor, "critics": self.critics}

    def dist(self, obs, avail):
        return Categorical(logits=masked_logits(self.actor(self.agent_inputs(obs)), avail))

    @torch.no_grad()
    def act(self, obs, state, avail, deterministic=False):
        d = self.dist(self.t(obs), self.t(avail))
        a = d.probs.argmax(-1) if deterministic else d.sample()
        return a.cpu().numpy()

    @torch.no_grad()
    def explain(self, obs, state, avail):
        return {"probs": self.dist(self.t(obs[None]), self.t(avail[None])).probs[0].cpu().numpy()}

    def critic_inputs(self, obs, state, actions):
        B = obs.shape[0]
        joint = F.one_hot(actions, self.A).float().reshape(B, 1, -1).expand(B, self.n, self.n * self.A)
        s = state.unsqueeze(1).expand(B, self.n, state.shape[-1])
        return torch.cat([s, obs, self.eye.expand(B, self.n, self.n), joint * self.others_mask], -1)

    def min_q(self, critics, x):
        return torch.min(torch.stack([c(x) for c in critics]), 0).values  # (B, n, A)

    def update(self):
        hp = self.hp
        _, batch = self.buffer.sample(int(hp["batch_size"]))
        b = self.to_torch(batch)
        with torch.no_grad():
            d_next = self.dist(b["next_obs"], b["next_avail"])
            u_next = d_next.sample()
            q_next = masked_logits(self.min_q(self.critics_target, self.critic_inputs(b["next_obs"], b["next_state"], u_next)),
                                   b["next_avail"])
            probs, logp = d_next.probs, torch.log(d_next.probs.clamp_min(1e-8))
            v_next = (probs * (q_next - self.alpha * logp)).sum(-1)  # (B, n)
            y = b["reward"][:, None] + hp["gamma"] * (1 - b["done"])[:, None] * v_next
        x = self.critic_inputs(b["obs"], b["state"], b["actions"])
        critic_loss = sum(F.mse_loss(c(x).gather(-1, b["actions"][..., None]).squeeze(-1), y) for c in self.critics)
        self.clip_and_step(self.critic_opt, critic_loss, self.critics.parameters(), hp["max_grad_norm"])
        d = self.dist(b["obs"], b["avail"])
        probs, logp = d.probs, torch.log(d.probs.clamp_min(1e-8))
        with torch.no_grad():
            q = masked_logits(self.min_q(self.critics, x), b["avail"])
        actor_loss = (probs * (self.alpha * logp - q)).sum(-1).mean()
        self.clip_and_step(self.actor_opt, actor_loss, self.actor.parameters(), hp["max_grad_norm"])
        entropy = -(probs * logp).sum(-1).detach()
        if hp["autotune"]:
            alpha_loss = (self.log_alpha * (entropy - self.target_entropy)).mean()
            self.clip_and_step(self.alpha_opt, alpha_loss, [self.log_alpha])
        soft_update(self.critics_target, self.critics, hp["tau"])
        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item(), "alpha": self.alpha.item(),
                "entropy": entropy.mean().item()}

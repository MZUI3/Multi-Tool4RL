"""MADDPG and MATD3 for discrete actions via Gumbel-Softmax (as in the MPE experiments of Lowe et al.)."""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from rlzoo.common.nets import mlp
from rlzoo.common.utils import masked_logits, soft_update
from rlzoo.marl.base import MAOffPolicyAlgo


class MADDPG(MAOffPolicyAlgo):
    """Decentralized actors + centralized critics Q_i(s, u_1..u_n).

    Actor and critic parameters are shared across agents (agent id in the input). The actor gradient for
    agent i flows through a straight-through Gumbel-Softmax sample of its own action while other agents'
    actions come from the replay buffer.
    """

    name = "maddpg"
    default_hparams = dict(actor_lr=5e-4, critic_lr=1e-3, tau=0.01, gumbel_temperature=1.0, batch_size=256,
                           buffer_size=100_000, learning_starts=2_000, critic_hidden=[128, 128], policy_reg=1e-3,
                           max_grad_norm=0.5, n_critics=1, policy_delay=1, target_smoothing=False)

    def __init__(self, spec, cfg, device, logger=None):
        super().__init__(spec, cfg, device, logger)
        hp = self.hp
        self.actor = mlp(self.obs_dim, self.A, hp["hidden_sizes"], hp["activation"]).to(device)
        self.target_actor = copy.deepcopy(self.actor)
        critic_in = self.state_dim + self.n * self.A + (self.n if hp["agent_id"] else 0)
        self.critics = nn.ModuleList([mlp(critic_in, 1, hp["critic_hidden"], hp["activation"])
                                      for _ in range(int(hp["n_critics"]))]).to(device)
        self.target_critics = copy.deepcopy(self.critics)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=hp["actor_lr"])
        self.critic_opt = torch.optim.Adam(self.critics.parameters(), lr=hp["critic_lr"])
        self.buffer = self.make_buffer()
        self.n_updates = 0
        self._actor_loss = 0.0

    def networks(self):
        return {"actor": self.actor, "critics": self.critics}

    def logits(self, net, obs, avail):
        raw = net(self.agent_inputs(obs))
        return raw, masked_logits(raw, avail)

    @torch.no_grad()
    def act(self, obs, state, avail, deterministic=False):
        _, logits = self.logits(self.actor, self.t(obs), self.t(avail))
        a = logits.argmax(-1) if deterministic else Categorical(logits=logits).sample()
        return a.cpu().numpy()

    def q(self, critic, state, joint):
        """state (B, s), joint one-hot actions per critic-agent (B, n, n, A) -> (B, n)"""
        x = torch.cat([self.state_inputs(state), joint.flatten(-2)], -1)
        return critic(x).squeeze(-1)

    def joint_for_all(self, joint):
        """(B, n, A) -> (B, n, n, A): every agent's critic sees the same joint action."""
        return joint.unsqueeze(1).expand(joint.shape[0], self.n, self.n, self.A)

    def update(self):
        hp = self.hp
        _, batch = self.buffer.sample(int(hp["batch_size"]))
        b = self.to_torch(batch)
        with torch.no_grad():
            _, next_logits = self.logits(self.target_actor, b["next_obs"], b["next_avail"])
            if hp["target_smoothing"]:
                next_joint = F.gumbel_softmax(next_logits, tau=hp["gumbel_temperature"], hard=True)
            else:
                next_joint = F.one_hot(next_logits.argmax(-1), self.A).float()
            next_q = torch.stack([self.q(c, b["next_state"], self.joint_for_all(next_joint))
                                  for c in self.target_critics]).min(0).values
            y = b["reward"][:, None] + hp["gamma"] * (1 - b["done"])[:, None] * next_q
        joint = F.one_hot(b["actions"], self.A).float()
        critic_loss = sum(F.mse_loss(self.q(c, b["state"], self.joint_for_all(joint)), y) for c in self.critics)
        self.clip_and_step(self.critic_opt, critic_loss, self.critics.parameters(), hp["max_grad_norm"])
        self.n_updates += 1
        if self.n_updates % int(hp["policy_delay"]) == 0:
            raw, logits = self.logits(self.actor, b["obs"], b["avail"])
            pi = F.gumbel_softmax(logits, tau=hp["gumbel_temperature"], hard=True)  # (B, n, A)
            eye = self.eye[None, :, :, None]
            joint_i = self.joint_for_all(joint) * (1 - eye) + pi.unsqueeze(1) * eye  # agent i uses its policy
            actor_loss = -self.q(self.critics[0], b["state"], joint_i).mean() + hp["policy_reg"] * (raw ** 2).mean()
            self.clip_and_step(self.actor_opt, actor_loss, self.actor.parameters(), hp["max_grad_norm"])
            self._actor_loss = actor_loss.item()
            soft_update(self.target_actor, self.actor, hp["tau"])
            soft_update(self.target_critics, self.critics, hp["tau"])
        return {"critic_loss": critic_loss.item(), "actor_loss": self._actor_loss}


class MATD3(MADDPG):
    """MATD3 (Ackermann et al., 2019): twin critics, delayed policy updates, target policy smoothing."""

    name = "matd3"
    default_hparams = dict(n_critics=2, policy_delay=2, target_smoothing=True)

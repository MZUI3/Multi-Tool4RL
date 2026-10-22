"""COMA: Counterfactual Multi-Agent policy gradients (Foerster et al., 2018)."""
from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from rlzoo.common.nets import mlp
from rlzoo.common.utils import masked_logits
from rlzoo.marl.base import MAOnPolicyAlgo, flatten_time


class COMA(MAOnPolicyAlgo):
    """Shared decentralized actor + centralized critic Q(s, o_i, i, u_-i, .) returning Q for each own action.

    Advantage_i = Q(s, u) - sum_a pi_i(a) Q(s, (u_-i, a)); the critic regresses TD(lambda) targets built from
    a target critic.
    """

    name = "coma"
    default_hparams = dict(lr=5e-4, critic_lr=5e-4, n_steps=50, td_lambda=0.8, ent_coef=0.01, critic_epochs=10,
                           target_update_interval=20, max_grad_norm=10.0, critic_hidden=[128, 128],
                           normalize_advantage=True)

    def __init__(self, spec, cfg, device, logger=None):
        super().__init__(spec, cfg, device, logger)
        hp = self.hp
        self.actor = mlp(self.obs_dim, self.A, hp["hidden_sizes"], hp["activation"]).to(device)
        critic_in = self.state_dim + spec.obs_dim + self.n + self.n * self.A
        self.critic = mlp(critic_in, self.A, hp["critic_hidden"], hp["activation"]).to(device)
        self.target_critic = copy.deepcopy(self.critic)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=hp["lr"])
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=hp["critic_lr"])
        # zero out each agent's own action block in the joint-action input
        self.others_mask = (1 - self.eye).repeat_interleave(self.A, dim=1)  # (n, n*A)
        self.n_updates = 0

    def networks(self):
        return {"actor": self.actor, "critic": self.critic}

    def dist(self, obs, avail):
        return Categorical(logits=masked_logits(self.actor(self.agent_inputs(obs)), avail))

    @torch.no_grad()
    def act(self, obs, state, avail, deterministic=False):
        d = self.dist(self.t(obs), self.t(avail))
        a = d.probs.argmax(-1) if deterministic else d.sample()
        return a.cpu().numpy()

    def critic_inputs(self, obs, state, actions):
        B = obs.shape[0]
        joint = F.one_hot(actions, self.A).float().reshape(B, 1, -1).expand(B, self.n, self.n * self.A)
        s = state.unsqueeze(1).expand(B, self.n, state.shape[-1])
        ids = self.eye.expand(B, self.n, self.n)
        return torch.cat([s, obs, ids, joint * self.others_mask], -1)

    def policy_step(self, obs, state, avail):
        d = self.dist(obs, avail)
        a = d.sample()
        return a, d.log_prob(a), torch.zeros_like(a, dtype=torch.float32)

    def bootstrap_value(self, obs, state, avail):
        d = self.dist(obs, avail)
        q = self.target_critic(self.critic_inputs(obs, state, d.sample()))
        return (d.probs * q).sum(-1)

    def update(self, data):
        hp = self.hp
        T, E = data["rewards"].shape[:2]
        b = self.to_torch(flatten_time(data, ("obs", "state", "avail", "actions")))
        rewards, dones = self.t(data["rewards"]), self.t(data["dones"])
        inputs = self.critic_inputs(b["obs"], b["state"], b["actions"])
        act_idx = b["actions"][..., None]
        with torch.no_grad():
            q_target = self.target_critic(inputs).gather(-1, act_idx).squeeze(-1).view(T, E, self.n)
            returns = torch.zeros_like(rewards)
            g = next_v = self.t(data["last_values"])
            for t in reversed(range(T)):
                nonterminal = (1 - dones[t])[:, None]
                g = rewards[t] + hp["gamma"] * nonterminal * ((1 - hp["td_lambda"]) * next_v + hp["td_lambda"] * g)
                returns[t] = g
                next_v = q_target[t]
            returns = returns.view(T * E, self.n)
        for _ in range(int(hp["critic_epochs"])):
            q_taken = self.critic(inputs).gather(-1, act_idx).squeeze(-1)
            critic_loss = F.mse_loss(q_taken, returns)
            self.clip_and_step(self.critic_opt, critic_loss, self.critic.parameters(), hp["max_grad_norm"])
        with torch.no_grad():
            q_all = self.critic(inputs)
        d = self.dist(b["obs"], b["avail"])
        baseline = (d.probs.detach() * q_all).sum(-1)
        adv = q_all.gather(-1, act_idx).squeeze(-1) - baseline
        if hp["normalize_advantage"]:
            adv = adv / (adv.std() + 1e-8)
        entropy = d.entropy().mean()
        actor_loss = -(d.log_prob(b["actions"]) * adv).mean() - hp["ent_coef"] * entropy
        self.clip_and_step(self.actor_opt, actor_loss, self.actor.parameters(), hp["max_grad_norm"])
        self.n_updates += 1
        if self.n_updates % int(hp["target_update_interval"]) == 0:
            self.target_critic.load_state_dict(self.critic.state_dict())
        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item(), "entropy": entropy.item()}

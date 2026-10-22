"""PPO-based MARL: IPPO, MAPPO, HAPPO, and CommNet / G2ANet communication actors trained with MAPPO.

Variants differ only in the actor architecture (shared MLP, per-agent, CommNet, G2ANet), the critic input
(local observation vs. global state) and HAPPO's sequential per-agent update.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.distributions import Categorical

from rlzoo.common.buffers import compute_gae
from rlzoo.common.nets import mlp
from rlzoo.common.utils import RunningMeanStd, masked_logits
from rlzoo.marl.base import MAOnPolicyAlgo, flatten_time
from rlzoo.marl.networks import CommNetActor, G2ANetActor, PerAgentActor


class MAPPO(MAOnPolicyAlgo):
    """MAPPO (Yu et al., 2022): shared actor, centralized value V(s, i)."""

    name = "mappo"
    actor_type = "mlp"  # mlp | separate | commnet | g2anet
    critic_type = "central"  # central (global state) | local (own observation)
    default_hparams = dict(lr=5e-4, critic_lr=5e-4, n_steps=100, gae_lambda=0.95, clip_range=0.2, n_epochs=5,
                           n_minibatches=2, ent_coef=0.01, max_grad_norm=10.0, normalize_advantage=True,
                           value_clip=0.2, value_norm=True)  # value_norm: running return normalisation

    def __init__(self, spec, cfg, device, logger=None):
        super().__init__(spec, cfg, device, logger)
        hp = self.hp
        self.actor = self.build_actor().to(device)
        critic_in = (self.state_dim + (self.n if hp["agent_id"] else 0)) if self.critic_type == "central" \
            else self.obs_dim
        self.critic = mlp(critic_in, 1, hp["hidden_sizes"], hp["activation"]).to(device)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=hp["critic_lr"], eps=1e-5)
        self.vnorm = RunningMeanStd() if hp["value_norm"] else None
        self.setup_actor_optimizer()

    def build_actor(self):
        return mlp(self.obs_dim, self.A, self.hp["hidden_sizes"], self.hp["activation"])

    def setup_actor_optimizer(self):
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.hp["lr"], eps=1e-5)

    def networks(self):
        return {"actor": self.actor, "critic": self.critic}

    # ---------------------------------------------------------------- model
    def dist(self, obs, avail):
        return Categorical(logits=masked_logits(self.actor(self.agent_inputs(obs)), avail))

    def values(self, obs, state):
        """Raw critic output (normalised scale when value_norm is on)."""
        x = self.state_inputs(state) if self.critic_type == "central" else self.agent_inputs(obs)
        return self.critic(x).squeeze(-1)

    def real_values(self, obs, state):
        v = self.values(obs, state)
        return self.vnorm.denormalize(v) if self.vnorm is not None else v

    @torch.no_grad()
    def act(self, obs, state, avail, deterministic=False):
        d = self.dist(self.t(obs), self.t(avail))
        a = d.probs.argmax(-1) if deterministic else d.sample()
        return a.cpu().numpy()

    def policy_step(self, obs, state, avail):
        d = self.dist(obs, avail)
        a = d.sample()
        return a, d.log_prob(a), self.real_values(obs, state)

    def bootstrap_value(self, obs, state, avail):
        return self.real_values(obs, state)

    # ---------------------------------------------------------------- update
    def prepare(self, data):
        hp = self.hp
        dones = np.repeat(data["dones"][..., None], self.n, axis=-1)
        adv, ret = compute_gae(data["rewards"], data["values"], dones, data["last_values"], hp["gamma"],
                               hp["gae_lambda"])
        data = dict(data, advantages=adv, returns=ret)
        b = self.to_torch(flatten_time(data, ("obs", "state", "avail", "actions", "logp", "values", "advantages",
                                              "returns")))
        if hp["normalize_advantage"]:
            b["advantages"] = (b["advantages"] - b["advantages"].mean()) / (b["advantages"].std() + 1e-8)
        if self.vnorm is not None:  # critic regresses normalised returns
            self.vnorm.update(ret)
            b["returns"] = self.vnorm.normalize(b["returns"])
            b["values"] = self.vnorm.normalize(b["values"])
        return b

    def minibatches(self, N):
        mb = max(1, N // int(self.hp["n_minibatches"]))
        for _ in range(int(self.hp["n_epochs"])):
            perm = torch.randperm(N, device=self.device)
            for s in range(0, N, mb):
                yield perm[s:s + mb]

    def value_loss(self, b, idx):
        v = self.values(b["obs"][idx], b["state"][idx])
        ret = b["returns"][idx]
        clip = self.hp["value_clip"]
        if clip:
            old = b["values"][idx]
            v_clip = old + (v - old).clamp(-clip, clip)
            return 0.5 * torch.max((v - ret) ** 2, (v_clip - ret) ** 2).mean()
        return 0.5 * ((v - ret) ** 2).mean()

    def critic_update(self, b, idx):
        loss = self.value_loss(b, idx)
        self.clip_and_step(self.critic_opt, loss, self.critic.parameters(), self.hp["max_grad_norm"])
        return loss.item()

    def update(self, data):
        hp = self.hp
        b = self.prepare(data)
        clip = hp["clip_range"]
        stats = {"policy_loss": [], "value_loss": [], "entropy": []}
        for idx in self.minibatches(b["obs"].shape[0]):
            d = self.dist(b["obs"][idx], b["avail"][idx])
            ratio = (d.log_prob(b["actions"][idx]) - b["logp"][idx]).exp()
            adv = b["advantages"][idx]
            policy_loss = -torch.min(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv).mean()
            entropy = d.entropy().mean()
            self.clip_and_step(self.actor_opt, policy_loss - hp["ent_coef"] * entropy, self.actor.parameters(),
                               hp["max_grad_norm"])
            stats["value_loss"].append(self.critic_update(b, idx))
            stats["policy_loss"].append(policy_loss.item())
            stats["entropy"].append(entropy.item())
        return {k: float(np.mean(v)) for k, v in stats.items()}


class IPPO(MAPPO):
    """Independent PPO (de Witt et al., 2020): shared actor, decentralized critic V(o_i)."""

    name = "ippo"
    critic_type = "local"


class HAPPO(MAPPO):
    """Heterogeneous-Agent PPO (Kuba et al., 2022).

    Agents have separate policies and are updated one after another in a random order; agent i's surrogate is
    weighted by M = prod_{j updated before i} pi_j_new / pi_j_old, which yields a monotonic-improvement
    guarantee for the joint policy.
    """

    name = "happo"
    actor_type = "separate"

    def build_actor(self):
        return PerAgentActor(self.n, self.obs_dim, self.A, self.hp["hidden_sizes"], self.hp["activation"])

    def setup_actor_optimizer(self):
        self.actor_opts = [torch.optim.Adam(net.parameters(), lr=self.hp["lr"], eps=1e-5) for net in self.actor.nets]

    def agent_dist(self, i, x, avail):
        return Categorical(logits=masked_logits(self.actor.nets[i](x), avail))

    def update(self, data):
        hp = self.hp
        b = self.prepare(data)
        N, clip = b["obs"].shape[0], hp["clip_range"]
        x = self.agent_inputs(b["obs"])
        factor = torch.ones(N, device=self.device)
        stats = {"policy_loss": [], "value_loss": [], "entropy": []}
        for i in torch.randperm(self.n).tolist():
            for idx in self.minibatches(N):
                d = self.agent_dist(i, x[idx, i], b["avail"][idx, i])
                ratio = (d.log_prob(b["actions"][idx, i]) - b["logp"][idx, i]).exp()
                adv = factor[idx] * b["advantages"][idx, i]
                policy_loss = -torch.min(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv).mean()
                entropy = d.entropy().mean()
                self.clip_and_step(self.actor_opts[i], policy_loss - hp["ent_coef"] * entropy,
                                   self.actor.nets[i].parameters(), hp["max_grad_norm"])
                stats["policy_loss"].append(policy_loss.item())
                stats["entropy"].append(entropy.item())
            with torch.no_grad():
                new_logp = self.agent_dist(i, x[:, i], b["avail"][:, i]).log_prob(b["actions"][:, i])
                factor = factor * (new_logp - b["logp"][:, i]).exp()
        for idx in self.minibatches(N):
            stats["value_loss"].append(self.critic_update(b, idx))
        return {k: float(np.mean(v)) for k, v in stats.items()}


class CommNet(MAPPO):
    """CommNet communication actor (Sukhbaatar et al., 2016), trained with PPO + centralized critic."""

    name = "commnet"
    actor_type = "commnet"
    default_hparams = dict(comm_hidden=64, comm_steps=2, comm_activation="tanh")

    def build_actor(self):
        hp = self.hp
        return CommNetActor(self.obs_dim, self.A, hp["comm_hidden"], hp["comm_steps"], hp["comm_activation"])


class G2ANet(MAPPO):
    """G2ANet two-stage (hard + soft) attention actor (Liu et al., 2020), trained with PPO + centralized critic."""

    name = "g2anet"
    actor_type = "g2anet"
    default_hparams = dict(attention_hidden=64)

    def build_actor(self):
        return G2ANetActor(self.obs_dim, self.A, self.hp["attention_hidden"], self.hp["activation"])

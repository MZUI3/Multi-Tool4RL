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

    def __init__(self, spec, cfg, device, logger=None, accel=None):
        super().__init__(spec, cfg, device, logger, accel)
        if spec.discrete:
            raise TypeError(f"{self.name} requires a continuous (Box) action space")
        hp = self.hp
        self.low = torch.as_tensor(spec.act_low, device=device)
        self.high = torch.as_tensor(spec.act_high, device=device)
        self.scale = (self.high - self.low) / 2
        self.critics = nn.ModuleList([self.build_critic() for _ in range(int(hp["n_critics"]))]).to(device)
        self.critics_target = copy.deepcopy(self.critics)
        self.critic_opt = torch.optim.Adam(self.critics.parameters(), lr=hp["critic_lr"])
        self.buffer = self.make_buffer()
        self.n_updates = 0

    def build_critic(self):
        hp = self.hp
        return QCritic(self.spec.obs_dim, self.spec.act_dim, hp["hidden_sizes"], hp["activation"], encoder=self.make_encoder())

    def critic_step(self, b, target):
        loss = sum(F.mse_loss(c(b["obs"], b["action"]), target) for c in self.critics)
        self.clip_and_step(self.critic_opt, loss, self.critics.parameters(), self.hp["max_grad_norm"])
        return loss.item()


class DDPG(_ContinuousBase):
    """Lillicrap et al., 2016. TD3 reuses this with twin critics, delayed actor and target smoothing."""

    name = "ddpg"
    default_hparams = dict(exploration_noise=0.1, policy_delay=1, target_policy_noise=0.0, target_noise_clip=0.5)

    def __init__(self, spec, cfg, device, logger=None, accel=None):
        super().__init__(spec, cfg, device, logger, accel)
        hp = self.hp
        self.actor = DeterministicActor(spec.obs_dim, spec.act_dim, hp["hidden_sizes"], hp["activation"],
                                        spec.act_low, spec.act_high, encoder=self.make_encoder()).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=hp["actor_lr"])
        self._actor_loss = 0.0

    def networks(self):
        return {"actor": self.actor, "critics": self.critics, "actor_target": self.actor_target,
                "critics_target": self.critics_target}

    @torch.no_grad()
    def explain(self, obs):
        obs_t = self.t(np.asarray(obs)[None])
        a = self.actor(obs_t)
        return {"mean": a[0].cpu().numpy(), "q": float(self.critics[0](obs_t, a)[0])}

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
    """Soft Actor-Critic with automatic entropy tuning (Haarnoja et al., 2018).

    `target_subset` / `actor_q` / `policy_delay` / critic regularisation options make REDQ and DroQ plain
    hyper-parameter variants of this class.
    """

    name = "sac"
    default_hparams = dict(actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4, n_critics=2, alpha=0.2, autotune=True,
                           target_entropy=None,
                           target_subset=0,  # REDQ: min over a random subset of M target critics (0 = all)
                           actor_q="min",  # min | mean over critics in the actor objective
                           policy_delay=1,  # actor / alpha updates every N critic updates
                           critic_dropout=0.0, critic_layer_norm=False)  # DroQ regularisation

    def __init__(self, spec, cfg, device, logger=None, accel=None):
        super().__init__(spec, cfg, device, logger, accel)
        hp = self.hp
        self.actor = SquashedGaussianActor(spec.obs_dim, spec.act_dim, hp["hidden_sizes"], hp["activation"],
                                           spec.act_low, spec.act_high, encoder=self.make_encoder()).to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=hp["actor_lr"])
        self.target_entropy = hp["target_entropy"] if hp["target_entropy"] is not None else -float(spec.act_dim)
        self.log_alpha = torch.tensor(np.log(hp["alpha"]), dtype=torch.float32, device=device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=hp["alpha_lr"])
        self._actor_loss = 0.0

    def build_critic(self):
        hp = self.hp
        return QCritic(self.spec.obs_dim, self.spec.act_dim, hp["hidden_sizes"], hp["activation"],
                       encoder=self.make_encoder(), dropout=hp["critic_dropout"], layer_norm=hp["critic_layer_norm"])

    @property
    def alpha(self):
        return self.log_alpha.exp().detach()

    def target_critics_subset(self):
        m = int(self.hp["target_subset"])
        if m <= 0 or m >= len(self.critics_target):
            return list(self.critics_target)
        idx = np.random.choice(len(self.critics_target), m, replace=False)
        return [self.critics_target[i] for i in idx]

    def actor_q_value(self, obs, a):
        qs = torch.stack([c(obs, a) for c in self.critics])
        return qs.mean(0) if self.hp["actor_q"] == "mean" else qs.min(0).values

    def networks(self):
        return {"actor": self.actor, "critics": self.critics, "critics_target": self.critics_target}

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        a, _ = self.actor(self.t(obs), deterministic=deterministic)
        return a.cpu().numpy()

    @torch.no_grad()
    def explain(self, obs):
        obs_t = self.t(np.asarray(obs)[None])
        a, _ = self.actor(obs_t, deterministic=True)
        q = torch.min(torch.stack([c(obs_t, a) for c in self.critics]), 0).values
        return {"mean": a[0].cpu().numpy(), "q": float(q[0])}

    def sample_batch(self):
        _, batch = self.buffer.sample(int(self.hp["batch_size"]))
        return self.to_torch(batch)

    def actor_update(self, b):
        hp = self.hp
        a, logp = self.actor(b["obs"])
        actor_loss = (self.alpha * logp - self.actor_q_value(b["obs"], a)).mean()
        self.clip_and_step(self.actor_opt, actor_loss, self.actor.parameters(), hp["max_grad_norm"])
        if hp["autotune"]:
            alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
            self.clip_and_step(self.alpha_opt, alpha_loss, [self.log_alpha])
        self._actor_loss = actor_loss.item()

    def update(self):
        hp = self.hp
        b = self.sample_batch()
        with torch.no_grad():
            next_a, next_logp = self.actor(b["next_obs"])
            next_q = torch.min(torch.stack([c(b["next_obs"], next_a) for c in self.target_critics_subset()]), 0).values
            target = b["reward"] + b["discount"] * (1 - b["done"]) * (next_q - self.alpha * next_logp)
        critic_loss = self.critic_step(b, target)
        self.n_updates += 1
        if self.n_updates % int(hp["policy_delay"]) == 0:
            self.actor_update(b)
        soft_update(self.critics_target, self.critics, hp["tau"])
        return {"critic_loss": critic_loss, "actor_loss": self._actor_loss, "alpha": self.alpha.item()}


class REDQ(SAC):
    """Randomized Ensembled Double Q-learning (Chen et al., 2021): 10 critics, targets from the min over a random
    pair, actor objective on the ensemble mean, 20 critic updates per environment step."""

    name = "redq"
    default_hparams = dict(n_critics=10, target_subset=2, actor_q="mean", gradient_steps=20, policy_delay=20)


class DroQ(SAC):
    """DroQ (Hiraoka et al., 2022): REDQ's update-to-data ratio with only two critics regularised by dropout and
    LayerNorm instead of a large ensemble."""

    name = "droq"
    default_hparams = dict(n_critics=2, actor_q="min", critic_dropout=0.01, critic_layer_norm=True, gradient_steps=20,
                           policy_delay=20)


class TQC(SAC):
    """Truncated Quantile Critics (Kuznetsov et al., 2020): each critic outputs N quantiles; the target pools the
    quantiles of all critics, sorts them and drops the top `top_quantiles_to_drop` per critic to control
    overestimation."""

    name = "tqc"
    default_hparams = dict(n_critics=5, n_quantiles=25, top_quantiles_to_drop=2, kappa=1.0)

    def build_critic(self):
        hp = self.hp
        return QCritic(self.spec.obs_dim, self.spec.act_dim, hp["hidden_sizes"], hp["activation"],
                       encoder=self.make_encoder(), n_quantiles=int(hp["n_quantiles"]))

    def actor_q_value(self, obs, a):
        return torch.stack([c(obs, a).mean(-1) for c in self.critics]).mean(0)

    def update(self):
        hp = self.hp
        b = self.sample_batch()
        N, n_c = int(hp["n_quantiles"]), len(self.critics)
        keep = n_c * N - int(hp["top_quantiles_to_drop"]) * n_c
        with torch.no_grad():
            next_a, next_logp = self.actor(b["next_obs"])
            z = torch.cat([c(b["next_obs"], next_a) for c in self.critics_target], -1)  # (B, n_c * N)
            z = z.sort(-1).values[:, :keep]
            target = b["reward"][:, None] + (b["discount"] * (1 - b["done"]))[:, None] * (z - self.alpha * next_logp[:, None])
        tau = (torch.arange(N, device=self.device, dtype=torch.float32) + 0.5) / N
        losses = []
        for c in self.critics:
            q = c(b["obs"], b["action"])  # (B, N)
            td = target[:, None, :] - q[:, :, None]  # (B, N, keep)
            huber = torch.where(td.abs() <= hp["kappa"], 0.5 * td.pow(2), hp["kappa"] * (td.abs() - 0.5 * hp["kappa"]))
            losses.append(((tau[None, :, None] - (td.detach() < 0).float()).abs() * huber / hp["kappa"]).mean(2).sum(1).mean())
        critic_loss = sum(losses)
        self.clip_and_step(self.critic_opt, critic_loss, self.critics.parameters(), hp["max_grad_norm"])
        self.n_updates += 1
        if self.n_updates % int(hp["policy_delay"]) == 0:
            self.actor_update(b)
        soft_update(self.critics_target, self.critics, hp["tau"])
        return {"critic_loss": critic_loss.item(), "actor_loss": self._actor_loss, "alpha": self.alpha.item()}

    @torch.no_grad()
    def explain(self, obs):
        obs_t = self.t(np.asarray(obs)[None])
        a, _ = self.actor(obs_t, deterministic=True)
        return {"mean": a[0].cpu().numpy(), "q": float(self.actor_q_value(obs_t, a)[0])}


class CrossQ(SAC):
    """CrossQ (Bhatt et al., 2024): no target networks. The critic uses Batch Renormalization and is evaluated on
    the joint batch [(s, a); (s', a')] so both halves share the same normalisation statistics; the actor loss
    runs the critics in eval mode."""

    name = "crossq"
    default_hparams = dict(n_critics=2, policy_delay=3, critic_lr=1e-3, actor_lr=1e-3, critic_beta1=0.5)

    def __init__(self, spec, cfg, device, logger=None, accel=None):
        super().__init__(spec, cfg, device, logger, accel)
        self.critic_opt = torch.optim.Adam(self.critics.parameters(), lr=self.hp["critic_lr"], betas=(self.hp["critic_beta1"], 0.999))
        self.critics_target = self.critics  # alias: CrossQ has no target network

    def build_critic(self):
        hp = self.hp
        return QCritic(self.spec.obs_dim, self.spec.act_dim, hp["hidden_sizes"], hp["activation"],
                       encoder=self.make_encoder(), batch_renorm=True)

    def networks(self):
        return {"actor": self.actor, "critics": self.critics}

    def update(self):
        hp = self.hp
        b = self.sample_batch()
        B = b["obs"].shape[0]
        with torch.no_grad():
            next_a, next_logp = self.actor(b["next_obs"])
        obs_cat, act_cat = torch.cat([b["obs"], b["next_obs"]]), torch.cat([b["action"], next_a])
        self.critics.train()
        q_cat = torch.stack([c(obs_cat, act_cat) for c in self.critics])  # (n_c, 2B)
        q_cur, q_next = q_cat[:, :B], q_cat[:, B:]
        with torch.no_grad():
            target = b["reward"] + b["discount"] * (1 - b["done"]) * (q_next.min(0).values - self.alpha * next_logp)
        critic_loss = sum(F.mse_loss(q_cur[i], target) for i in range(len(self.critics)))
        self.clip_and_step(self.critic_opt, critic_loss, self.critics.parameters(), hp["max_grad_norm"])
        self.n_updates += 1
        if self.n_updates % int(hp["policy_delay"]) == 0:
            self.critics.eval()
            self.actor_update(b)
            self.critics.train()
        return {"critic_loss": critic_loss.item(), "actor_loss": self._actor_loss, "alpha": self.alpha.item()}

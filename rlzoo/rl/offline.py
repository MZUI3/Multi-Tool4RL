"""Offline RL from a fixed dataset (no environment interaction during training).

BC      behaviour cloning
TD3+BC  Fujimoto & Gu, 2021 (continuous)
CQL     Kumar et al., 2020 (discrete: CQL(H) on DQN; continuous: CQL on SAC)
IQL     Kostrikov et al., 2022 (expectile value, advantage-weighted policy extraction)
AWAC    Nair et al., 2020
DT      Decision Transformer, Chen et al., 2021 (return-conditioned sequence model)

The config names the data with `dataset: auto | path.npz | runs/<run_dir>` (see rlzoo/common/datasets.py);
`train.total_steps` counts gradient steps. Evaluation still runs in the real environment.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Independent, Normal

from rlzoo.common.datasets import episode_slices, resolve_dataset
from rlzoo.common.nets import CategoricalActor, DeterministicActor, DiscreteQCritic, QCritic, VNetwork, mlp
from rlzoo.common.utils import soft_update
from rlzoo.rl.base import RLAlgorithm


class GaussianActor(nn.Module):
    """Gaussian policy with a tanh-bounded mean and a state-independent log-std (IQL / AWAC style)."""

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, low, high, encoder=None, log_std_init=-1.0):
        super().__init__()
        self.mean_net = DeterministicActor(obs_dim, act_dim, hidden_sizes, activation, low, high, encoder)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std_init)))

    def dist(self, obs):
        mu = self.mean_net(obs)
        return Independent(Normal(mu, self.log_std.clamp(-5, 2).exp().expand_as(mu)), 1)

    def forward(self, obs):
        return self.mean_net(obs)


class OfflineAlgo(RLAlgorithm):
    default_hparams = dict(batch_size=256, lr=3e-4, hidden_sizes=[256, 256], normalize_obs=True)

    def __init__(self, spec, cfg, device, logger=None, accel=None):
        super().__init__(spec, cfg, device, logger, accel)
        self.data = resolve_dataset(cfg, verbose=self.accel.is_main)
        obs = self.data["obs"].astype(np.float32)
        if self.hp["normalize_obs"]:
            self.obs_mean, self.obs_std = obs.mean(0), obs.std(0) + 1e-3
        else:
            self.obs_mean, self.obs_std = np.zeros(obs.shape[1], np.float32), np.ones(obs.shape[1], np.float32)
        self._obs_mean_t, self._obs_std_t = self.t(self.obs_mean), self.t(self.obs_std)
        self.tensors = self.to_torch({
            "obs": (obs - self.obs_mean) / self.obs_std,
            "action": self.data["actions"],
            "reward": self.data["rewards"].astype(np.float32),
            "next_obs": (self.data["next_obs"].astype(np.float32) - self.obs_mean) / self.obs_std,
            "done": self.data["terminals"].astype(np.float32),
        })
        self.n = obs.shape[0]

    def norm(self, obs):
        return (self.t(obs) - self._obs_mean_t) / self._obs_std_t

    def sample(self):
        idx = torch.randint(0, self.n, (int(self.hp["batch_size"]),), device=self.device)
        return {k: v[idx] for k, v in self.tensors.items()}

    def update(self, b):
        raise NotImplementedError

    def train(self, venv, total_steps):
        while self.global_step < total_steps:
            self.train_metrics = self.update(self.sample())
            self.global_step += self.step_scale
            self.maybe_log()

    # policy helpers shared by the actor-based methods
    def build_policy(self):
        hp, spec = self.hp, self.spec
        if spec.discrete:
            return CategoricalActor(spec.obs_dim, spec.n_actions, hp["hidden_sizes"], hp["activation"], self.make_encoder())
        return GaussianActor(spec.obs_dim, spec.act_dim, hp["hidden_sizes"], hp["activation"], spec.act_low, spec.act_high,
                             self.make_encoder())

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        d = self.actor.dist(self.norm(obs))
        if self.spec.discrete:
            a = d.probs.argmax(-1) if deterministic else d.sample()
        else:
            a = d.mean if deterministic else d.sample()
        return a.cpu().numpy()

    @torch.no_grad()
    def explain(self, obs):
        d = self.actor.dist(self.norm(np.asarray(obs)[None]))
        return {"probs": d.probs[0].cpu().numpy()} if self.spec.discrete else {"mean": d.mean[0].cpu().numpy()}


class BC(OfflineAlgo):
    name = "bc"

    def __init__(self, spec, cfg, device, logger=None, accel=None):
        super().__init__(spec, cfg, device, logger, accel)
        self.actor = self.build_policy().to(device)
        self.opt = torch.optim.Adam(self.actor.parameters(), lr=self.hp["lr"])

    def networks(self):
        return {"actor": self.actor}

    def update(self, b):
        loss = -self.actor.dist(b["obs"]).log_prob(b["action"]).mean()
        self.clip_and_step(self.opt, loss, self.actor.parameters())
        return {"nll": loss.item()}


class TD3BC(OfflineAlgo):
    """TD3+BC: TD3 whose actor loss is -lambda Q / mean|Q| + (pi(s) - a)^2."""

    name = "td3_bc"
    default_hparams = dict(alpha=2.5, tau=0.005, gamma=0.99, policy_delay=2, target_policy_noise=0.2, target_noise_clip=0.5)

    def __init__(self, spec, cfg, device, logger=None, accel=None):
        super().__init__(spec, cfg, device, logger, accel)
        if spec.discrete:
            raise TypeError("td3_bc needs continuous actions (use cql / iql / awac / bc for discrete)")
        hp = self.hp
        self.actor = DeterministicActor(spec.obs_dim, spec.act_dim, hp["hidden_sizes"], hp["activation"], spec.act_low,
                                        spec.act_high, self.make_encoder()).to(device)
        self.critics = nn.ModuleList([QCritic(spec.obs_dim, spec.act_dim, hp["hidden_sizes"], hp["activation"],
                                              self.make_encoder()) for _ in range(2)]).to(device)
        self.actor_target, self.critics_target = copy.deepcopy(self.actor), copy.deepcopy(self.critics)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=hp["lr"])
        self.critic_opt = torch.optim.Adam(self.critics.parameters(), lr=hp["lr"])
        self.low, self.high = self.t(spec.act_low), self.t(spec.act_high)
        self._actor_loss = 0.0

    def networks(self):
        return {"actor": self.actor, "critics": self.critics}

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        return self.actor(self.norm(obs)).cpu().numpy()

    @torch.no_grad()
    def explain(self, obs):
        x = self.norm(np.asarray(obs)[None])
        a = self.actor(x)
        return {"mean": a[0].cpu().numpy(), "q": float(self.critics[0](x, a)[0])}

    def update(self, b):
        hp = self.hp
        with torch.no_grad():
            noise = (torch.randn_like(b["action"]) * hp["target_policy_noise"]).clamp(-hp["target_noise_clip"], hp["target_noise_clip"])
            next_a = torch.max(torch.min(self.actor_target(b["next_obs"]) + noise * (self.high - self.low) / 2, self.high), self.low)
            next_q = torch.min(torch.stack([c(b["next_obs"], next_a) for c in self.critics_target]), 0).values
            target = b["reward"] + hp["gamma"] * (1 - b["done"]) * next_q
        critic_loss = sum(F.mse_loss(c(b["obs"], b["action"]), target) for c in self.critics)
        self.clip_and_step(self.critic_opt, critic_loss, self.critics.parameters())
        self.global_step_updates = getattr(self, "global_step_updates", 0) + 1
        if self.global_step_updates % int(hp["policy_delay"]) == 0:
            pi = self.actor(b["obs"])
            q = self.critics[0](b["obs"], pi)
            lam = hp["alpha"] / q.abs().mean().detach()
            actor_loss = -lam * q.mean() + F.mse_loss(pi, b["action"])
            self.clip_and_step(self.actor_opt, actor_loss, self.actor.parameters())
            self._actor_loss = actor_loss.item()
            soft_update(self.actor_target, self.actor, hp["tau"])
            soft_update(self.critics_target, self.critics, hp["tau"])
        return {"critic_loss": critic_loss.item(), "actor_loss": self._actor_loss}


class CQL(OfflineAlgo):
    """Conservative Q-Learning. Discrete: DQN target + alpha (logsumexp_a Q(s,a) - Q(s,a_data)).
    Continuous: SAC-style actor with the CQL(H) penalty estimated from random + policy actions."""

    name = "cql"
    default_hparams = dict(cql_alpha=1.0, gamma=0.99, tau=0.005, n_random_actions=10, sac_alpha=0.2, target_update_interval=1)

    def __init__(self, spec, cfg, device, logger=None, accel=None):
        super().__init__(spec, cfg, device, logger, accel)
        hp = self.hp
        if spec.discrete:
            self.q = DiscreteQCritic(spec.obs_dim, spec.n_actions, hp["hidden_sizes"], hp["activation"], self.make_encoder()).to(device)
            self.q_target = copy.deepcopy(self.q)
            self.critic_opt = torch.optim.Adam(self.q.parameters(), lr=hp["lr"])
        else:
            from rlzoo.common.nets import SquashedGaussianActor

            self.actor = SquashedGaussianActor(spec.obs_dim, spec.act_dim, hp["hidden_sizes"], hp["activation"], spec.act_low,
                                               spec.act_high, self.make_encoder()).to(device)
            self.critics = nn.ModuleList([QCritic(spec.obs_dim, spec.act_dim, hp["hidden_sizes"], hp["activation"],
                                                  self.make_encoder()) for _ in range(2)]).to(device)
            self.critics_target = copy.deepcopy(self.critics)
            self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=hp["lr"])
            self.critic_opt = torch.optim.Adam(self.critics.parameters(), lr=hp["lr"])
            self.low, self.high = self.t(spec.act_low), self.t(spec.act_high)

    def networks(self):
        if self.spec.discrete:
            return {"q": self.q}
        return {"actor": self.actor, "critics": self.critics}

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        x = self.norm(obs)
        if self.spec.discrete:
            return self.q(x).argmax(-1).cpu().numpy()
        a, _ = self.actor(x, deterministic=deterministic)
        return a.cpu().numpy()

    @torch.no_grad()
    def explain(self, obs):
        x = self.norm(np.asarray(obs)[None])
        if self.spec.discrete:
            return {"q": self.q(x)[0].cpu().numpy()}
        a, _ = self.actor(x, deterministic=True)
        return {"mean": a[0].cpu().numpy(), "q": float(self.critics[0](x, a)[0])}

    def update(self, b):
        hp = self.hp
        if self.spec.discrete:
            q_all = self.q(b["obs"])
            q = q_all.gather(1, b["action"][:, None]).squeeze(1)
            with torch.no_grad():
                next_a = self.q(b["next_obs"]).argmax(1, keepdim=True)
                target = b["reward"] + hp["gamma"] * (1 - b["done"]) * self.q_target(b["next_obs"]).gather(1, next_a).squeeze(1)
            td_loss = F.smooth_l1_loss(q, target)
            cql_loss = (torch.logsumexp(q_all, -1) - q).mean()
            loss = td_loss + hp["cql_alpha"] * cql_loss
            self.clip_and_step(self.critic_opt, loss, self.q.parameters())
            soft_update(self.q_target, self.q, hp["tau"])
            return {"td_loss": td_loss.item(), "cql_loss": cql_loss.item()}
        # continuous
        B, K, alpha = b["obs"].shape[0], int(hp["n_random_actions"]), hp["sac_alpha"]
        with torch.no_grad():
            next_a, next_logp = self.actor(b["next_obs"])
            next_q = torch.min(torch.stack([c(b["next_obs"], next_a) for c in self.critics_target]), 0).values
            target = b["reward"] + hp["gamma"] * (1 - b["done"]) * (next_q - alpha * next_logp)
            rand_a = torch.rand(B * K, self.spec.act_dim, device=self.device) * (self.high - self.low) + self.low
            obs_rep = b["obs"].repeat_interleave(K, 0)
            pi_a, pi_logp = self.actor(obs_rep)
            log_unif = -math.log(float(torch.prod(self.high - self.low)))
        td_loss, cql_loss = 0.0, 0.0
        for c in self.critics:
            q = c(b["obs"], b["action"])
            td_loss = td_loss + F.mse_loss(q, target)
            q_rand = c(obs_rep, rand_a).view(B, K) - log_unif
            q_pi = c(obs_rep, pi_a).view(B, K) - pi_logp.view(B, K)
            cql_loss = cql_loss + (torch.logsumexp(torch.cat([q_rand, q_pi], 1), 1) - math.log(2 * K) - q).mean()
        loss = td_loss + hp["cql_alpha"] * cql_loss
        self.clip_and_step(self.critic_opt, loss, self.critics.parameters())
        a, logp = self.actor(b["obs"])
        actor_loss = (alpha * logp - torch.min(torch.stack([c(b["obs"], a) for c in self.critics]), 0).values).mean()
        self.clip_and_step(self.actor_opt, actor_loss, self.actor.parameters())
        soft_update(self.critics_target, self.critics, hp["tau"])
        return {"td_loss": float(td_loss), "cql_loss": float(cql_loss), "actor_loss": actor_loss.item()}


class IQLOffline(OfflineAlgo):
    """Implicit Q-Learning: V by expectile regression on Q, Q by TD on V, policy by advantage-weighted regression."""

    name = "iql_offline"
    default_hparams = dict(expectile=0.7, beta=3.0, gamma=0.99, tau=0.005, max_weight=100.0)

    def __init__(self, spec, cfg, device, logger=None, accel=None):
        super().__init__(spec, cfg, device, logger, accel)
        hp = self.hp
        self.actor = self.build_policy().to(device)
        self.v = VNetwork(spec.obs_dim, hp["hidden_sizes"], hp["activation"], self.make_encoder()).to(device)
        if spec.discrete:
            self.critics = nn.ModuleList([DiscreteQCritic(spec.obs_dim, spec.n_actions, hp["hidden_sizes"], hp["activation"],
                                                          self.make_encoder()) for _ in range(2)]).to(device)
        else:
            self.critics = nn.ModuleList([QCritic(spec.obs_dim, spec.act_dim, hp["hidden_sizes"], hp["activation"],
                                                  self.make_encoder()) for _ in range(2)]).to(device)
        self.critics_target = copy.deepcopy(self.critics)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=hp["lr"])
        self.v_opt = torch.optim.Adam(self.v.parameters(), lr=hp["lr"])
        self.critic_opt = torch.optim.Adam(self.critics.parameters(), lr=hp["lr"])

    def networks(self):
        return {"actor": self.actor, "v": self.v, "critics": self.critics}

    def q_of(self, critics, obs, action):
        if self.spec.discrete:
            return torch.min(torch.stack([c(obs).gather(1, action[:, None]).squeeze(1) for c in critics]), 0).values
        return torch.min(torch.stack([c(obs, action) for c in critics]), 0).values

    def update(self, b):
        hp = self.hp
        with torch.no_grad():
            q_t = self.q_of(self.critics_target, b["obs"], b["action"])
        v = self.v(b["obs"])
        diff = q_t - v
        v_loss = (torch.abs(hp["expectile"] - (diff < 0).float()) * diff ** 2).mean()
        self.clip_and_step(self.v_opt, v_loss, self.v.parameters())
        with torch.no_grad():
            target = b["reward"] + hp["gamma"] * (1 - b["done"]) * self.v(b["next_obs"])
        if self.spec.discrete:
            q_loss = sum(F.mse_loss(c(b["obs"]).gather(1, b["action"][:, None]).squeeze(1), target) for c in self.critics)
        else:
            q_loss = sum(F.mse_loss(c(b["obs"], b["action"]), target) for c in self.critics)
        self.clip_and_step(self.critic_opt, q_loss, self.critics.parameters())
        with torch.no_grad():
            w = torch.exp(hp["beta"] * diff).clamp(max=hp["max_weight"])
        logp = self.actor.dist(b["obs"]).log_prob(b["action"])
        actor_loss = -(w * logp).mean()
        self.clip_and_step(self.actor_opt, actor_loss, self.actor.parameters())
        soft_update(self.critics_target, self.critics, hp["tau"])
        return {"v_loss": v_loss.item(), "q_loss": q_loss.item(), "actor_loss": actor_loss.item()}


class AWAC(OfflineAlgo):
    """Advantage-Weighted Actor-Critic: SAC-like critics, policy trained by weighted regression with weights
    exp(A(s,a) / lambda), A = Q(s,a) - E_{a' ~ pi} Q(s,a')."""

    name = "awac"
    default_hparams = dict(lam=1.0, gamma=0.99, tau=0.005, n_action_samples=4, max_weight=20.0)

    def __init__(self, spec, cfg, device, logger=None, accel=None):
        super().__init__(spec, cfg, device, logger, accel)
        hp = self.hp
        self.actor = self.build_policy().to(device)
        if spec.discrete:
            self.critics = nn.ModuleList([DiscreteQCritic(spec.obs_dim, spec.n_actions, hp["hidden_sizes"], hp["activation"],
                                                          self.make_encoder()) for _ in range(2)]).to(device)
        else:
            self.critics = nn.ModuleList([QCritic(spec.obs_dim, spec.act_dim, hp["hidden_sizes"], hp["activation"],
                                                  self.make_encoder()) for _ in range(2)]).to(device)
        self.critics_target = copy.deepcopy(self.critics)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=hp["lr"])
        self.critic_opt = torch.optim.Adam(self.critics.parameters(), lr=hp["lr"])

    def networks(self):
        return {"actor": self.actor, "critics": self.critics}

    def _q(self, critics, obs, action):
        if self.spec.discrete:
            return torch.min(torch.stack([c(obs).gather(1, action[:, None]).squeeze(1) for c in critics]), 0).values
        return torch.min(torch.stack([c(obs, action) for c in critics]), 0).values

    def update(self, b):
        hp = self.hp
        with torch.no_grad():
            d_next = self.actor.dist(b["next_obs"])
            next_a = d_next.sample()
            target = b["reward"] + hp["gamma"] * (1 - b["done"]) * self._q(self.critics_target, b["next_obs"], next_a)
        q_loss = sum(F.mse_loss(self._q([c], b["obs"], b["action"]), target) for c in self.critics)
        self.clip_and_step(self.critic_opt, q_loss, self.critics.parameters())
        d = self.actor.dist(b["obs"])
        with torch.no_grad():
            if self.spec.discrete:
                q_all = torch.min(torch.stack([c(b["obs"]) for c in self.critics]), 0).values
                v = (d.probs * q_all).sum(-1)
            else:
                v = torch.stack([self._q(self.critics, b["obs"], d.sample()) for _ in range(int(hp["n_action_samples"]))]).mean(0)
            adv = self._q(self.critics, b["obs"], b["action"]) - v
            w = torch.exp(adv / hp["lam"]).clamp(max=hp["max_weight"])
        actor_loss = -(w * d.log_prob(b["action"])).mean()
        self.clip_and_step(self.actor_opt, actor_loss, self.actor.parameters())
        soft_update(self.critics_target, self.critics, hp["tau"])
        return {"q_loss": q_loss.item(), "actor_loss": actor_loss.item(), "adv": adv.mean().item()}


# ------------------------------------------------------------------------------- Decision Transformer


class _GPTBlock(nn.Module):
    def __init__(self, embed, heads, dropout):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(embed), nn.LayerNorm(embed)
        self.attn = nn.MultiheadAttention(embed, heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(embed, 4 * embed), nn.GELU(), nn.Linear(4 * embed, embed), nn.Dropout(dropout))

    def forward(self, x, mask):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class DecisionTransformerNet(nn.Module):
    def __init__(self, obs_dim, act_dim, discrete, embed=128, heads=1, layers=3, context=20, max_ep_len=1000, dropout=0.1):
        super().__init__()
        self.discrete, self.act_dim, self.context = discrete, act_dim, context
        self.embed_t = nn.Embedding(max_ep_len + context, embed)
        self.embed_r = nn.Linear(1, embed)
        self.embed_s = nn.Linear(obs_dim, embed)
        self.embed_a = nn.Embedding(act_dim + 1, embed) if discrete else nn.Linear(act_dim, embed)
        self.ln = nn.LayerNorm(embed)
        self.blocks = nn.ModuleList([_GPTBlock(embed, heads, dropout) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(embed)
        self.head = nn.Linear(embed, act_dim)
        self.register_buffer("causal", torch.triu(torch.ones(3 * context, 3 * context, dtype=torch.bool), 1))

    def forward(self, rtg, obs, act, timesteps):
        """rtg (B, K, 1), obs (B, K, d), act (B, K[, a]), timesteps (B, K) -> action predictions (B, K, act_dim)."""
        B, K = obs.shape[:2]
        te = self.embed_t(timesteps)
        r, s, a = self.embed_r(rtg) + te, self.embed_s(obs) + te, self.embed_a(act) + te
        x = torch.stack([r, s, a], 2).reshape(B, 3 * K, -1)  # (R_1, s_1, a_1, R_2, ...)
        x = self.ln(x)
        mask = self.causal[: 3 * K, : 3 * K]
        for blk in self.blocks:
            x = blk(x, mask)
        x = self.ln_f(x).view(B, K, 3, -1)
        return self.head(x[:, :, 1])  # predict a_t from the state token


class DecisionTransformer(OfflineAlgo):
    """Decision Transformer: a causal transformer over (return-to-go, state, action) tokens trained to predict the
    dataset actions; at test time it is conditioned on a target return (`target_return`, default: the best
    dataset return)."""

    name = "dt"
    default_hparams = dict(context=20, embed=128, n_heads=1, n_layers=3, dropout=0.1, lr=1e-4, weight_decay=1e-4,
                           batch_size=64, target_return=None, return_scale=None, max_ep_len=1000)

    def __init__(self, spec, cfg, device, logger=None, accel=None):
        super().__init__(spec, cfg, device, logger, accel)
        hp = self.hp
        self.K = int(hp["context"])
        self.act_dim = spec.n_actions if spec.discrete else spec.act_dim
        self.model = DecisionTransformerNet(spec.obs_dim, self.act_dim, spec.discrete, int(hp["embed"]), int(hp["n_heads"]),
                                            int(hp["n_layers"]), self.K, int(hp["max_ep_len"]), hp["dropout"]).to(device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
        # per-episode returns-to-go
        self.slices = episode_slices(self.data)
        rewards = self.data["rewards"].astype(np.float32)
        self.rtg = np.zeros_like(rewards)
        ep_returns = []
        for s, e in self.slices:
            self.rtg[s:e] = np.cumsum(rewards[s:e][::-1])[::-1]
            ep_returns.append(float(rewards[s:e].sum()))
        self.return_scale = float(hp["return_scale"] or max(1.0, np.abs(ep_returns).max()))
        self.target_return = float(hp["target_return"] if hp["target_return"] is not None else max(ep_returns))
        self.timesteps = np.concatenate([np.arange(e - s) for s, e in self.slices])
        self._ctx = {}

    def networks(self):
        return {"model": self.model}

    def sample(self):
        B, K = int(self.hp["batch_size"]), self.K
        obs = np.zeros((B, K, self.spec.obs_dim), np.float32)
        act = np.zeros((B, K, self.spec.act_dim), np.float32) if not self.spec.discrete else np.full((B, K), self.act_dim, np.int64)
        rtg = np.zeros((B, K, 1), np.float32)
        ts = np.zeros((B, K), np.int64)
        mask = np.zeros((B, K), np.float32)
        tgt = np.zeros((B, K), np.int64) if self.spec.discrete else np.zeros((B, K, self.spec.act_dim), np.float32)
        norm_obs = self.tensors["obs"].cpu().numpy()
        for i in range(B):
            s, e = self.slices[np.random.randint(len(self.slices))]
            start = np.random.randint(s, e)
            end = min(start + K, e)
            n = end - start
            obs[i, :n] = norm_obs[start:end]
            act[i, :n] = self.data["actions"][start:end]
            tgt[i, :n] = self.data["actions"][start:end]
            rtg[i, :n, 0] = self.rtg[start:end] / self.return_scale
            ts[i, :n] = np.minimum(self.timesteps[start:end], int(self.hp["max_ep_len"]) - 1)
            mask[i, :n] = 1.0
        batch = self.to_torch({"obs": obs, "act": act, "rtg": rtg, "ts": ts, "mask": mask, "tgt": tgt})
        return batch

    def update(self, b):
        self.model.train()
        pred = self.model(b["rtg"], b["obs"], b["act"], b["ts"])
        if self.spec.discrete:
            loss = (F.cross_entropy(pred.reshape(-1, self.act_dim), b["tgt"].reshape(-1), reduction="none") * b["mask"].reshape(-1)).sum() / b["mask"].sum()
        else:
            loss = (((pred - b["tgt"]) ** 2).mean(-1) * b["mask"]).sum() / b["mask"].sum()
        self.clip_and_step(self.opt, loss, self.model.parameters(), 0.25)
        return {"loss": loss.item()}

    # ---------------------------------------------------------------- acting with a rolling context
    def reset_hidden(self, n: int):
        self._ctx[n] = {"obs": [], "act": [], "rtg": [np.full(n, self.target_return, np.float32)], "t": 0}

    def observe(self, reward):
        """Feed the reward of the last step so the return-to-go can be decreased (called by the evaluator)."""
        for ctx in self._ctx.values():
            ctx["rtg"].append(ctx["rtg"][-1] - np.asarray(reward, np.float32))

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        n = obs.shape[0]
        if n not in self._ctx:
            self.reset_hidden(n)
        ctx = self._ctx[n]
        if len(ctx["rtg"]) <= len(ctx["obs"]):  # reward not observed: keep the previous return-to-go
            ctx["rtg"].append(ctx["rtg"][-1])
        ctx["obs"].append(((np.asarray(obs, np.float32) - self.obs_mean) / self.obs_std))
        K = self.K
        obs_seq = np.stack(ctx["obs"][-K:], 1)  # (n, k, d)
        k = obs_seq.shape[1]
        acts = ctx["act"][-(k - 1):] if k > 1 else []
        if self.spec.discrete:
            act_seq = np.full((n, k), self.act_dim, np.int64)
            if acts:
                act_seq[:, :k - 1] = np.stack(acts, 1)
        else:
            act_seq = np.zeros((n, k, self.spec.act_dim), np.float32)
            if acts:
                act_seq[:, :k - 1] = np.stack(acts, 1)
        rtg_seq = np.stack(ctx["rtg"][-k:], 1)[..., None] / self.return_scale
        ts = np.minimum(np.arange(ctx["t"] - k + 1, ctx["t"] + 1), int(self.hp["max_ep_len"]) - 1)[None].repeat(n, 0)
        self.model.eval()
        pred = self.model(self.t(rtg_seq), self.t(obs_seq), self.t(act_seq, torch.long if self.spec.discrete else torch.float32),
                          self.t(ts, torch.long))[:, -1]
        if self.spec.discrete:
            a = pred.argmax(-1) if deterministic else Categorical(logits=pred).sample()
            a_np = a.cpu().numpy()
        else:
            a_np = np.clip(pred.cpu().numpy(), self.spec.act_low, self.spec.act_high)
        ctx["act"].append(a_np)
        ctx["t"] += 1
        return a_np

    @torch.no_grad()
    def explain(self, obs):
        return {"target_return": float(self._ctx[1]["rtg"][-1][0]) if 1 in self._ctx else self.target_return}

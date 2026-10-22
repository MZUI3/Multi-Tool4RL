"""DQN family: DQN, Double DQN, Dueling DQN, PER-DQN, C51, QR-DQN and Rainbow.

All variants share one implementation; each class just changes default hparams (double / dueling / noisy /
prioritized / n_step) or the distributional loss, so every ingredient can also be toggled from the config.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from rlzoo.common.buffers import NStepAccumulator, PrioritizedReplayBuffer, ReplayBuffer
from rlzoo.common.nets import QNetwork
from rlzoo.common.utils import linear_schedule, update_target
from rlzoo.rl.base import OffPolicyAlgo


class DQN(OffPolicyAlgo):
    """Mnih et al., 2015."""

    name = "dqn"
    default_hparams = dict(
        lr=1e-3,
        buffer_size=50_000,
        target_update_interval=500,  # gradient updates between hard target copies (if tau == 1)
        tau=1.0,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_steps=10_000,
        double=False,
        dueling=False,
        noisy=False,
        noisy_sigma=0.5,
        prioritized=False,
        per_alpha=0.6,
        per_beta_start=0.4,
        per_beta_steps=0,  # 0 -> anneal beta to 1 over the whole run
        n_step=1,
        max_grad_norm=10.0,
        loss="huber",  # huber | mse (ignored by distributional variants)
    )

    def __init__(self, spec, cfg, device, logger=None):
        super().__init__(spec, cfg, device, logger)
        if not spec.discrete:
            raise TypeError(f"{self.name} requires a discrete action space")
        hp = self.hp
        self.total_steps = int(cfg["train"]["total_steps"])
        self.q = self.build_network().to(device)
        self.q_target = copy.deepcopy(self.q)
        self.opt = torch.optim.Adam(self.q.parameters(), lr=hp["lr"])
        if hp["prioritized"]:
            self.buffer = PrioritizedReplayBuffer(int(hp["buffer_size"]), self.buffer_fields(), hp["per_alpha"])
        else:
            self.buffer = ReplayBuffer(int(hp["buffer_size"]), self.buffer_fields())
        self.nstep = [NStepAccumulator(hp["n_step"], hp["gamma"]) for _ in range(spec.num_envs)]
        self.n_updates = 0

    # ---------------------------------------------------------------- network
    @property
    def n_atoms(self) -> int:
        return 1

    def build_network(self):
        hp = self.hp
        return QNetwork(self.spec.obs_dim, self.spec.n_actions, hp["hidden_sizes"], hp["activation"],
                        dueling=hp["dueling"], noisy=hp["noisy"], n_atoms=self.n_atoms, sigma0=hp["noisy_sigma"])

    def q_values(self, net, obs):
        return net(obs)

    def networks(self):
        return {"q": self.q, "q_target": self.q_target}

    # ---------------------------------------------------------------- acting
    def epsilon(self) -> float:
        hp = self.hp
        return linear_schedule(hp["eps_start"], hp["eps_end"], hp["eps_decay_steps"], self.global_step)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        if deterministic:
            self.q.eval()  # NoisyLinear uses mean weights in eval mode
        greedy = self.q_values(self.q, self.t(obs)).argmax(-1).cpu().numpy()
        self.q.train()
        if deterministic:
            return greedy
        random = np.random.randint(self.spec.n_actions, size=greedy.shape)
        explore = np.random.rand(*greedy.shape) < self.epsilon()
        return np.where(explore, random, greedy)

    def store(self, env_idx, obs, action, reward, next_obs, terminated, done):
        for tr in self.nstep[env_idx].push(obs, action, reward, next_obs, terminated, done):
            self.buffer.add(**tr)

    # ---------------------------------------------------------------- learning
    def compute_loss(self, b):
        """Returns per-sample loss (B,) and new priorities (B,)."""
        q = self.q_values(self.q, b["obs"]).gather(1, b["action"][:, None]).squeeze(1)
        with torch.no_grad():
            next_target = self.q_values(self.q_target, b["next_obs"])
            if self.hp["double"]:
                next_a = self.q_values(self.q, b["next_obs"]).argmax(1, keepdim=True)
                next_q = next_target.gather(1, next_a).squeeze(1)
            else:
                next_q = next_target.max(1).values
            target = b["reward"] + b["discount"] * (1 - b["done"]) * next_q
        if self.hp["loss"] == "mse":
            loss = F.mse_loss(q, target, reduction="none")
        else:
            loss = F.smooth_l1_loss(q, target, reduction="none")
        return loss, (target - q).abs().detach()

    def update(self):
        hp = self.hp
        if hp["prioritized"]:
            beta_steps = hp["per_beta_steps"] or self.total_steps
            beta = linear_schedule(hp["per_beta_start"], 1.0, beta_steps, self.global_step)
            idx, batch, weights = self.buffer.sample(int(hp["batch_size"]), beta)
            weights = self.t(weights)
        else:
            idx, batch = self.buffer.sample(int(hp["batch_size"]))
            weights = None
        b = self.to_torch(batch)
        loss_each, prio = self.compute_loss(b)
        loss = (loss_each * weights).mean() if weights is not None else loss_each.mean()
        self.clip_and_step(self.opt, loss, self.q.parameters(), hp["max_grad_norm"])
        if hp["prioritized"]:
            self.buffer.update_priorities(idx, prio.cpu().numpy())
        self.n_updates += 1
        update_target(self.q_target, self.q, hp["tau"], self.n_updates, hp["target_update_interval"])
        if hp["noisy"]:
            self.q.reset_noise()
            self.q_target.reset_noise()
        return {"loss": loss.item(), "epsilon": self.epsilon()}


class DoubleDQN(DQN):
    """van Hasselt et al., 2016."""

    name = "double_dqn"
    default_hparams = dict(double=True)


class DuelingDQN(DQN):
    """Wang et al., 2016 (dueling architecture on top of Double DQN, as in the paper)."""

    name = "dueling_dqn"
    default_hparams = dict(double=True, dueling=True)


class PERDQN(DQN):
    """Schaul et al., 2016 (prioritized replay on top of Double DQN, as in the paper)."""

    name = "per_dqn"
    default_hparams = dict(double=True, prioritized=True)


class C51(DQN):
    """Categorical DQN, Bellemare et al., 2017."""

    name = "c51"
    default_hparams = dict(atoms=51, v_min=-10.0, v_max=10.0)

    def __init__(self, spec, cfg, device, logger=None):
        hp = self.resolve_hparams(cfg.get("hparams") or {})
        self.support = torch.linspace(hp["v_min"], hp["v_max"], int(hp["atoms"]), device=device)
        self.delta_z = (hp["v_max"] - hp["v_min"]) / (int(hp["atoms"]) - 1)
        super().__init__(spec, cfg, device, logger)

    @property
    def n_atoms(self) -> int:
        return int(self.hp["atoms"])

    def q_values(self, net, obs):
        return (F.softmax(net(obs), -1) * self.support).sum(-1)

    def compute_loss(self, b):
        B, N = b["obs"].shape[0], self.n_atoms
        rng = torch.arange(B, device=self.device)
        log_p = F.log_softmax(self.q(b["obs"])[rng, b["action"]], -1)
        with torch.no_grad():
            next_probs = F.softmax(self.q_target(b["next_obs"]), -1)  # (B, A, N)
            if self.hp["double"]:
                next_a = self.q_values(self.q, b["next_obs"]).argmax(1)
            else:
                next_a = (next_probs * self.support).sum(-1).argmax(1)
            next_dist = next_probs[rng, next_a]
            tz = b["reward"][:, None] + (b["discount"] * (1 - b["done"]))[:, None] * self.support[None]
            tz = tz.clamp(self.hp["v_min"], self.hp["v_max"])
            pos = (tz - self.hp["v_min"]) / self.delta_z
            lo, hi = pos.floor().long(), pos.ceil().long()
            lo[(hi > 0) & (lo == hi)] -= 1
            hi[(lo < N - 1) & (lo == hi)] += 1
            m = torch.zeros(B, N, device=self.device)
            offset = (rng * N)[:, None]
            m.view(-1).index_add_(0, (lo + offset).view(-1), (next_dist * (hi.float() - pos)).view(-1))
            m.view(-1).index_add_(0, (hi + offset).view(-1), (next_dist * (pos - lo.float())).view(-1))
        loss = -(m * log_p).sum(-1)
        return loss, loss.detach()


class QRDQN(DQN):
    """Quantile Regression DQN, Dabney et al., 2018."""

    name = "qr_dqn"
    default_hparams = dict(quantiles=50, kappa=1.0)

    @property
    def n_atoms(self) -> int:
        return int(self.hp["quantiles"])

    def q_values(self, net, obs):
        return net(obs).mean(-1)

    def compute_loss(self, b):
        B, N, kappa = b["obs"].shape[0], self.n_atoms, self.hp["kappa"]
        rng = torch.arange(B, device=self.device)
        quantiles = self.q(b["obs"])[rng, b["action"]]  # (B, N)
        with torch.no_grad():
            next_q = self.q_target(b["next_obs"])
            if self.hp["double"]:
                next_a = self.q_values(self.q, b["next_obs"]).argmax(1)
            else:
                next_a = next_q.mean(-1).argmax(1)
            target = b["reward"][:, None] + (b["discount"] * (1 - b["done"]))[:, None] * next_q[rng, next_a]
        td = target[:, None, :] - quantiles[:, :, None]  # (B, N_pred, N_target)
        huber = torch.where(td.abs() <= kappa, 0.5 * td.pow(2), kappa * (td.abs() - 0.5 * kappa))
        tau = (torch.arange(N, device=self.device, dtype=torch.float32) + 0.5) / N
        weight = (tau[None, :, None] - (td.detach() < 0).float()).abs()
        loss = (weight * huber / kappa).mean(2).sum(1)
        return loss, loss.detach()


class Rainbow(C51):
    """Hessel et al., 2018: C51 + Double + Dueling + Noisy nets + PER + n-step returns."""

    name = "rainbow"
    default_hparams = dict(double=True, dueling=True, noisy=True, prioritized=True, n_step=3, per_alpha=0.5,
                           eps_start=0.0, eps_end=0.0)

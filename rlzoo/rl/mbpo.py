"""Model-Based Policy Optimization (Janner et al., 2019): SAC trained on a mix of real transitions and short
rollouts of a probabilistic ensemble dynamics model branched from real states."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rlzoo.common.buffers import ReplayBuffer
from rlzoo.rl.continuous import SAC


class EnsembleLinear(nn.Module):
    def __init__(self, n, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n, in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(n, 1, out_dim))
        for w in self.weight:
            nn.init.trunc_normal_(w, std=1.0 / (2 * np.sqrt(in_dim)))

    def forward(self, x):  # x (n, B, in)
        return torch.baddbmm(self.bias, x, self.weight)


class DynamicsEnsemble(nn.Module):
    """Ensemble of Gaussian MLPs predicting (delta_obs, reward) with learned, bounded log-variance."""

    def __init__(self, obs_dim, act_dim, n_models=5, hidden=(200, 200, 200)):
        super().__init__()
        self.n, self.out_dim = n_models, obs_dim + 1
        layers, last = [], obs_dim + act_dim
        for h in hidden:
            layers += [EnsembleLinear(n_models, last, h), nn.SiLU()]
            last = h
        self.body = nn.Sequential(*layers)
        self.head = EnsembleLinear(n_models, last, 2 * self.out_dim)
        self.max_logvar = nn.Parameter(torch.full((1, 1, self.out_dim), 0.5))
        self.min_logvar = nn.Parameter(torch.full((1, 1, self.out_dim), -10.0))
        self.register_buffer("in_mean", torch.zeros(obs_dim + act_dim))
        self.register_buffer("in_std", torch.ones(obs_dim + act_dim))

    def forward(self, obs, act):  # (n, B, ...) or broadcast (B, ...)
        x = torch.cat([obs, act], -1)
        if x.dim() == 2:
            x = x.unsqueeze(0).expand(self.n, -1, -1)
        x = (x - self.in_mean) / self.in_std
        out = self.head(self.body(x))
        mean, logvar = out[..., : self.out_dim], out[..., self.out_dim:]
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mean, logvar

    def loss(self, obs, act, target):
        mean, logvar = self(obs, act)
        inv_var = torch.exp(-logvar)
        nll = (((mean - target) ** 2) * inv_var + logvar).mean()
        return nll + 0.01 * (self.max_logvar.sum() - self.min_logvar.sum()), ((mean - target) ** 2).mean().item()


class MBPO(SAC):
    name = "mbpo"
    default_hparams = dict(n_models=5, model_hidden=[200, 200, 200], model_lr=1e-3, model_train_freq=250,
                           model_epochs=5, model_batch_size=256, rollout_length=1, rollout_batch=400,
                           model_buffer_size=100_000, real_ratio=0.05, gradient_steps=10, learning_starts=1_000,
                           random_steps=1_000, batch_size=256, max_model_logvar=0.5)

    def __init__(self, spec, cfg, device, logger=None, accel=None):
        super().__init__(spec, cfg, device, logger, accel)
        hp = self.hp
        self.model = DynamicsEnsemble(spec.obs_dim, spec.act_dim, int(hp["n_models"]), hp["model_hidden"]).to(device)
        self.model_opt = torch.optim.Adam(self.model.parameters(), lr=hp["model_lr"], weight_decay=1e-5)
        self.model_buffer = ReplayBuffer(int(hp["model_buffer_size"]), self.buffer_fields())
        self._last_model_train = -1
        self._model_mse = float("nan")

    def networks(self):
        return {**super().networks(), "model": self.model}

    # ---------------------------------------------------------------- model
    def train_model(self):
        hp = self.hp
        n = len(self.buffer)
        data = self.buffer.get(np.arange(n))
        obs, act = self.t(data["obs"]), self.t(data["action"])
        target = torch.cat([self.t(data["next_obs"]) - obs, self.t(data["reward"])[:, None]], -1)
        with torch.no_grad():
            x = torch.cat([obs, act], -1)
            self.model.in_mean.copy_(x.mean(0))
            self.model.in_std.copy_(x.std(0) + 1e-6)
        mb = int(hp["model_batch_size"])
        mses = []
        for _ in range(int(hp["model_epochs"])):
            for _ in range(max(1, n // mb)):
                idx = torch.stack([torch.randint(0, n, (mb,), device=self.device) for _ in range(self.model.n)])  # bootstrap per model
                loss, mse = self.model.loss(obs[idx], act[idx], target[idx])
                self.clip_and_step(self.model_opt, loss, self.model.parameters())
                mses.append(mse)
        self._model_mse = float(np.mean(mses))

    @torch.no_grad()
    def rollout_model(self):
        hp = self.hp
        B = min(int(hp["rollout_batch"]), len(self.buffer))
        _, batch = self.buffer.sample(B)
        obs = self.t(batch["obs"])
        for _ in range(int(hp["rollout_length"])):
            act, _ = self.actor(obs)
            mean, logvar = self.model(obs, act)  # (n, B, d+1)
            sample = mean + torch.randn_like(mean) * (0.5 * logvar).exp()
            pick = torch.randint(0, self.model.n, (obs.shape[0],), device=self.device)
            pred = sample[pick, torch.arange(obs.shape[0], device=self.device)]
            next_obs, reward = obs + pred[:, :-1], pred[:, -1]
            for o, a, r, o2 in zip(obs.cpu().numpy(), act.cpu().numpy(), reward.cpu().numpy(), next_obs.cpu().numpy()):
                self.model_buffer.add(obs=o, action=a, reward=r, next_obs=o2, done=0.0, discount=hp["gamma"])
            obs = next_obs

    def sample_batch(self):
        hp = self.hp
        B = int(hp["batch_size"])
        n_real = max(1, int(round(B * hp["real_ratio"]))) if len(self.model_buffer) >= B else B
        _, real = self.buffer.sample(n_real)
        if n_real < B:
            _, fake = self.model_buffer.sample(B - n_real)
            real = {k: np.concatenate([real[k], fake[k]]) for k in real}
        return self.to_torch(real)

    def update(self):
        hp = self.hp
        if self._last_model_train < 0 or self.global_step - self._last_model_train >= int(hp["model_train_freq"]) * self.step_scale:
            self.train_model()
            self.rollout_model()
            self._last_model_train = self.global_step
        metrics = super().update()
        metrics["model_mse"] = self._model_mse
        metrics["model_buffer"] = float(len(self.model_buffer))
        return metrics

"""Intrinsic-motivation exploration on top of PPO: RND (Burda et al., 2019) and ICM (Pathak et al., 2017).

Both add an intrinsic bonus to the environment reward inside the rollout (`reward_hook`) and train their
auxiliary networks after every PPO update. A single value head estimates the combined return, which is
the simplified ("single-head") form of the papers.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rlzoo.common.nets import mlp
from rlzoo.rl.ppo import PPO


class VecRunningMeanStd:
    def __init__(self, shape=(), eps=1e-4):
        self.mean, self.var, self.count = np.zeros(shape, np.float64), np.ones(shape, np.float64), eps

    def update(self, x):
        x = np.asarray(x, np.float64).reshape(-1, *self.mean.shape)
        b_mean, b_var, n = x.mean(0), x.var(0), x.shape[0]
        delta, total = b_mean - self.mean, self.count + n
        self.mean = self.mean + delta * n / total
        self.var = (self.var * self.count + b_var * n + delta ** 2 * self.count * n / total) / total
        self.count = total

    @property
    def std(self):
        return np.sqrt(self.var + 1e-8)


class _IntrinsicPPO(PPO):
    default_hparams = dict(int_coef=1.0, ext_coef=1.0, aux_lr=1e-4, aux_epochs=1, aux_minibatches=4, int_norm=True)

    def __init__(self, spec, cfg, device, logger=None, accel=None):
        super().__init__(spec, cfg, device, logger, accel)
        self._pending = []
        self.int_rms = VecRunningMeanStd(())
        self._last_int = 0.0

    def intrinsic(self, obs, actions, next_obs) -> np.ndarray:
        raise NotImplementedError

    def aux_loss(self, b) -> torch.Tensor:
        raise NotImplementedError

    def reward_hook(self, obs, actions, next_obs, rewards):
        hp = self.hp
        r_int = self.intrinsic(obs, actions, next_obs)
        self._pending.append((obs, actions, next_obs))
        if hp["int_norm"]:
            self.int_rms.update(r_int)
            r_int = r_int / self.int_rms.std
        self._last_int = float(np.mean(r_int))
        return hp["ext_coef"] * rewards + hp["int_coef"] * r_int.astype(np.float32)

    def update(self, b):
        metrics = super().update(b)
        obs = np.concatenate([p[0] for p in self._pending])
        acts = np.concatenate([p[1] for p in self._pending])
        nxt = np.concatenate([p[2] for p in self._pending])
        self._pending = []
        batch = self.to_torch({"obs": obs, "actions": acts, "next_obs": nxt})
        N = obs.shape[0]
        mb = max(1, N // int(self.hp["aux_minibatches"]))
        losses = []
        for _ in range(int(self.hp["aux_epochs"])):
            perm = torch.randperm(N, device=self.device)
            for s in range(0, N, mb):
                idx = perm[s:s + mb]
                loss = self.aux_loss({k: v[idx] for k, v in batch.items()})
                self.clip_and_step(self.aux_opt, loss, self.aux_params, 0.0)
                losses.append(loss.item())
        metrics.update({"aux_loss": float(np.mean(losses)), "int_reward": self._last_int})
        return metrics


class RND(_IntrinsicPPO):
    """Random Network Distillation: the bonus is the prediction error of a trained network imitating a fixed,
    randomly initialised target network on the next observation."""

    name = "rnd"
    default_hparams = dict(rnd_hidden=[64, 64], rnd_dim=64, obs_norm=True, int_coef=1.0)

    def __init__(self, spec, cfg, device, logger=None, accel=None):
        super().__init__(spec, cfg, device, logger, accel)
        hp = self.hp
        self.target = mlp(spec.obs_dim, int(hp["rnd_dim"]), hp["rnd_hidden"], "relu").to(device)
        self.predictor = mlp(spec.obs_dim, int(hp["rnd_dim"]), hp["rnd_hidden"], "relu").to(device)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.aux_params = list(self.predictor.parameters())
        self.aux_opt = torch.optim.Adam(self.aux_params, lr=hp["aux_lr"])
        self.obs_rms = VecRunningMeanStd((spec.obs_dim,))

    def networks(self):
        return {"ac": self.ac, "target": self.target, "predictor": self.predictor}

    def _norm(self, obs_np):
        if self.hp["obs_norm"]:
            self.obs_rms.update(obs_np)
            obs_np = np.clip((obs_np - self.obs_rms.mean) / self.obs_rms.std, -5, 5)
        return self.t(obs_np)

    @torch.no_grad()
    def intrinsic(self, obs, actions, next_obs):
        x = self._norm(next_obs)
        return ((self.target(x) - self.predictor(x)) ** 2).mean(-1).cpu().numpy()

    def aux_loss(self, b):
        x = b["next_obs"]
        if self.hp["obs_norm"]:
            x = ((x - self.t(self.obs_rms.mean)) / self.t(self.obs_rms.std)).clamp(-5, 5)
        return F.mse_loss(self.predictor(x), self.target(x).detach())


class ICM(_IntrinsicPPO):
    """Intrinsic Curiosity Module: a feature encoder trained through an inverse model (predict the action from
    phi(s), phi(s')) and a forward model whose prediction error of phi(s') is the curiosity bonus."""

    name = "icm"
    default_hparams = dict(icm_hidden=[64, 64], feat_dim=32, beta=0.2, int_coef=0.01, aux_lr=1e-3)

    def __init__(self, spec, cfg, device, logger=None, accel=None):
        super().__init__(spec, cfg, device, logger, accel)
        hp = self.hp
        d = int(hp["feat_dim"])
        self.a_dim = spec.n_actions if spec.discrete else spec.act_dim
        self.phi = mlp(spec.obs_dim, d, hp["icm_hidden"], "relu").to(device)
        self.inverse = mlp(2 * d, self.a_dim, hp["icm_hidden"], "relu").to(device)
        self.forward_model = mlp(d + self.a_dim, d, hp["icm_hidden"], "relu").to(device)
        self.aux_params = list(self.phi.parameters()) + list(self.inverse.parameters()) + list(self.forward_model.parameters())
        self.aux_opt = torch.optim.Adam(self.aux_params, lr=hp["aux_lr"])

    def networks(self):
        return {"ac": self.ac, "phi": self.phi, "inverse": self.inverse, "forward_model": self.forward_model}

    def _act_vec(self, a):
        return F.one_hot(a.long(), self.a_dim).float() if self.spec.discrete else a.float()

    @torch.no_grad()
    def intrinsic(self, obs, actions, next_obs):
        f, f2 = self.phi(self.t(obs)), self.phi(self.t(next_obs))
        pred = self.forward_model(torch.cat([f, self._act_vec(self.t(actions))], -1))
        return 0.5 * ((pred - f2) ** 2).sum(-1).cpu().numpy()

    def aux_loss(self, b):
        f, f2 = self.phi(b["obs"]), self.phi(b["next_obs"])
        a_vec = self._act_vec(b["actions"])
        inv = self.inverse(torch.cat([f, f2], -1))
        inv_loss = F.cross_entropy(inv, b["actions"].long()) if self.spec.discrete else F.mse_loss(inv, a_vec)
        fwd_loss = 0.5 * ((self.forward_model(torch.cat([f, a_vec], -1)) - f2.detach()) ** 2).sum(-1).mean()
        return (1 - self.hp["beta"]) * inv_loss + self.hp["beta"] * fwd_loss

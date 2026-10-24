"""Generative Adversarial Imitation Learning (Ho & Ermon, 2016): PPO whose reward is -log(1 - D(s, a)) from a
discriminator trained to tell expert transitions (an offline dataset) from the policy's own."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from rlzoo.common.datasets import resolve_dataset
from rlzoo.common.nets import mlp
from rlzoo.rl.ppo import PPO


class GAIL(PPO):
    name = "gail"
    default_hparams = dict(disc_lr=3e-4, disc_hidden=[64, 64], disc_epochs=2, disc_minibatches=4, env_reward_coef=0.0,
                           entropy_reg=1e-3, ent_coef=0.0)

    def __init__(self, spec, cfg, device, logger=None, accel=None):
        super().__init__(spec, cfg, device, logger, accel)
        hp = self.hp
        self.a_dim = spec.n_actions if spec.discrete else spec.act_dim
        self.disc = mlp(spec.obs_dim + self.a_dim, 1, hp["disc_hidden"], "tanh").to(device)
        self.disc_opt = torch.optim.Adam(self.disc.parameters(), lr=hp["disc_lr"])
        data = resolve_dataset(cfg, verbose=self.accel.is_main)
        self.expert = self.to_torch({"obs": data["obs"].astype(np.float32), "actions": data["actions"]})
        self._pending = []
        self._disc_acc = 0.0

    def networks(self):
        return {"ac": self.ac, "disc": self.disc}

    def _inputs(self, obs, actions):
        a = F.one_hot(actions.long(), self.a_dim).float() if self.spec.discrete else actions.float()
        return torch.cat([obs, a], -1)

    def reward_hook(self, obs, actions, next_obs, rewards):
        self._pending.append((obs, actions))
        with torch.no_grad():
            logits = self.disc(self._inputs(self.t(obs), self.t(actions))).squeeze(-1)
            r_imit = -F.logsigmoid(-logits)  # -log(1 - D) with D = sigmoid(logits) = P(expert)
        return self.hp["env_reward_coef"] * rewards + r_imit.cpu().numpy().astype(np.float32)

    def update(self, b):
        hp = self.hp
        metrics = super().update(b)
        obs = np.concatenate([p[0] for p in self._pending])
        acts = np.concatenate([p[1] for p in self._pending])
        self._pending = []
        pol = self.to_torch({"obs": obs, "actions": acts})
        N = obs.shape[0]
        mb = max(1, N // int(hp["disc_minibatches"]))
        losses, accs = [], []
        for _ in range(int(hp["disc_epochs"])):
            perm = torch.randperm(N, device=self.device)
            for s in range(0, N, mb):
                idx = perm[s:s + mb]
                e_idx = torch.randint(0, self.expert["obs"].shape[0], (idx.numel(),), device=self.device)
                pol_logits = self.disc(self._inputs(pol["obs"][idx], pol["actions"][idx])).squeeze(-1)
                exp_logits = self.disc(self._inputs(self.expert["obs"][e_idx], self.expert["actions"][e_idx])).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(pol_logits, torch.zeros_like(pol_logits)) + \
                    F.binary_cross_entropy_with_logits(exp_logits, torch.ones_like(exp_logits))
                probs = torch.cat([torch.sigmoid(pol_logits), torch.sigmoid(exp_logits)])
                entropy = -(probs * probs.clamp_min(1e-8).log() + (1 - probs) * (1 - probs).clamp_min(1e-8).log()).mean()
                self.clip_and_step(self.disc_opt, loss - hp["entropy_reg"] * entropy, self.disc.parameters())
                losses.append(loss.item())
                accs.append(0.5 * ((pol_logits < 0).float().mean() + (exp_logits > 0).float().mean()).item())
        metrics.update({"disc_loss": float(np.mean(losses)), "disc_acc": float(np.mean(accs))})
        return metrics

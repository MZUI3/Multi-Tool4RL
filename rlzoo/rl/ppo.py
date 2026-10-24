"""Proximal Policy Optimization (Schulman et al., 2017), clipped objective."""
from __future__ import annotations

import numpy as np
import torch

from rlzoo.rl.base import OnPolicyAlgo


class PPO(OnPolicyAlgo):
    name = "ppo"
    default_hparams = dict(lr=3e-4, n_steps=128, n_epochs=10, n_minibatches=4, clip_range=0.2, clip_vloss=False,
                           target_kl=0.0, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5)

    def __init__(self, spec, cfg, device, logger=None, accel=None):
        super().__init__(spec, cfg, device, logger, accel)
        self.opt = torch.optim.Adam(self.ac.parameters(), lr=self.hp["lr"], eps=1e-5)

    def update(self, b):
        hp = self.hp
        N = b["obs"].shape[0]
        mb_size = max(1, N // int(hp["n_minibatches"]))
        clip = hp["clip_range"]
        stats = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clip_frac": []}
        for _ in range(int(hp["n_epochs"])):
            perm = torch.randperm(N, device=self.device)
            for start in range(0, N, mb_size):
                idx = perm[start:start + mb_size]
                dist = self.ac.dist(b["obs"][idx])
                logp = dist.log_prob(b["actions"][idx])
                entropy = dist.entropy().mean()
                log_ratio = logp - b["logp"][idx]
                ratio = log_ratio.exp()
                adv = b["advantages"][idx]
                if hp["normalize_advantage"] and adv.numel() > 1:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                policy_loss = -torch.min(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv).mean()
                v = self.ac.value(b["obs"][idx])
                if hp["clip_vloss"]:
                    v_clipped = b["values"][idx] + (v - b["values"][idx]).clamp(-clip, clip)
                    value_loss = 0.5 * torch.max((v - b["returns"][idx]) ** 2,
                                                 (v_clipped - b["returns"][idx]) ** 2).mean()
                else:
                    value_loss = 0.5 * ((v - b["returns"][idx]) ** 2).mean()
                loss = policy_loss + hp["vf_coef"] * value_loss - hp["ent_coef"] * entropy
                self.clip_and_step(self.opt, loss, self.ac.parameters(), hp["max_grad_norm"])
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                stats["policy_loss"].append(policy_loss.item())
                stats["value_loss"].append(value_loss.item())
                stats["entropy"].append(entropy.item())
                stats["approx_kl"].append(approx_kl)
                stats["clip_frac"].append(((ratio - 1).abs() > clip).float().mean().item())
            if hp["target_kl"]:
                kl = self.reduce_mean(float(np.mean(stats["approx_kl"][-int(hp["n_minibatches"]):])))
                if kl > hp["target_kl"]:
                    break
        return {k: float(np.mean(v)) for k, v in stats.items()}

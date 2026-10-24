"""Phasic Policy Gradient (Cobbe et al., 2020)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical, Independent, Normal, kl_divergence

from rlzoo.common.nets import PPGNetwork
from rlzoo.rl.ppo import PPO


class PPG(PPO):
    """PPO with separate policy / value networks plus an auxiliary phase every `n_policy_iters` rollouts:
    the policy trunk is trained to predict values through an auxiliary head (L_aux = 0.5 (V_aux - R)^2)
    while a KL term keeps the policy close to what it was before the phase, and the value network is
    trained further on the stored returns."""

    name = "ppg"
    default_hparams = dict(n_policy_iters=16, aux_epochs=6, aux_minibatches=8, beta_clone=1.0, n_epochs=1,
                           ent_coef=0.0)

    def build_actor_critic(self):
        hp = self.hp
        return PPGNetwork(self.spec.obs_dim, self.spec, hp["hidden_sizes"], hp["activation"], hp["log_std_init"],
                          hp["ortho_init"], encoder_fn=self.make_encoder)

    def _old_dist(self, obs):
        d = self.ac.dist(obs)
        if self.ac.discrete:
            return Categorical(logits=d.logits.detach())
        return Independent(Normal(d.base_dist.loc.detach(), d.base_dist.scale.detach()), 1)

    def aux_phase(self, obs, returns):
        hp = self.hp
        N = obs.shape[0]
        mb = max(1, N // int(hp["aux_minibatches"]))
        with torch.no_grad():
            old = self._old_dist(obs)
        stats = {"aux_value_loss": [], "aux_kl": [], "aux_vf_loss": []}
        for _ in range(int(hp["aux_epochs"])):
            perm = torch.randperm(N, device=self.device)
            for s in range(0, N, mb):
                idx = perm[s:s + mb]
                dist, aux_v = self.ac.dist_and_aux(obs[idx])
                if self.ac.discrete:
                    old_mb = Categorical(logits=old.logits[idx])
                else:
                    old_mb = Independent(Normal(old.base_dist.loc[idx], old.base_dist.scale[idx]), 1)
                kl = kl_divergence(old_mb, dist).mean()
                aux_loss = 0.5 * F.mse_loss(aux_v, returns[idx]) + hp["beta_clone"] * kl
                vf_loss = 0.5 * F.mse_loss(self.ac.value(obs[idx]), returns[idx])
                self.clip_and_step(self.opt, aux_loss + vf_loss, self.ac.parameters(), hp["max_grad_norm"])
                stats["aux_value_loss"].append(aux_loss.item())
                stats["aux_kl"].append(kl.item())
                stats["aux_vf_loss"].append(vf_loss.item())
        return {k: float(np.mean(v)) for k, v in stats.items()}

    def train(self, venv, total_steps):
        hp = self.hp
        obs = venv.reset()
        aux_obs, aux_ret = [], []
        iters = 0
        while self.global_step < total_steps:
            batch, obs, episodes = self.collect_rollout(self.ac, venv, obs, hp["n_steps"])
            self.global_step += int(hp["n_steps"]) * venv.num_envs * self.step_scale
            self.record_episodes(episodes)
            b = self.to_torch(batch)
            self.train_metrics = self.update(b)
            aux_obs.append(b["obs"])
            aux_ret.append(b["returns"])
            iters += 1
            if iters % int(hp["n_policy_iters"]) == 0:
                self.train_metrics.update(self.aux_phase(torch.cat(aux_obs), torch.cat(aux_ret)))
                aux_obs, aux_ret = [], []
            self.maybe_log()

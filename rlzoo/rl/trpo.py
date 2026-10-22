"""Trust Region Policy Optimization (Schulman et al., 2015)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.distributions import Categorical, Independent, Normal, kl_divergence

from rlzoo.rl.base import OnPolicyAlgo


def _flat_grad(y, params, **kwargs):
    grads = torch.autograd.grad(y, params, **kwargs)
    return torch.cat([g.reshape(-1) for g in grads])


def _flat_params(params):
    return torch.cat([p.data.reshape(-1) for p in params])


def _set_flat_params(params, flat):
    i = 0
    for p in params:
        n = p.numel()
        p.data.copy_(flat[i:i + n].view_as(p))
        i += n


def _conjugate_gradient(fvp, b, iters, tol=1e-10):
    x = torch.zeros_like(b)
    r, p = b.clone(), b.clone()
    rr = r @ r
    for _ in range(iters):
        Ap = fvp(p)
        alpha = rr / (p @ Ap + 1e-8)
        x += alpha * p
        r -= alpha * Ap
        new_rr = r @ r
        if new_rr < tol:
            break
        p = r + (new_rr / rr) * p
        rr = new_rr
    return x


class TRPO(OnPolicyAlgo):
    name = "trpo"
    default_hparams = dict(n_steps=512, max_kl=0.01, damping=0.1, cg_iters=10, backtrack_iters=10,
                           backtrack_coef=0.5, vf_lr=1e-3, vf_iters=5, vf_minibatches=4)

    def __init__(self, spec, cfg, device, logger=None):
        super().__init__(spec, cfg, device, logger)
        self.vf_opt = torch.optim.Adam(self.ac.v.parameters(), lr=self.hp["vf_lr"])

    def _detached_dist(self, obs):
        with torch.no_grad():
            d = self.ac.dist(obs)
        if self.ac.discrete:
            return Categorical(logits=d.logits.detach())
        return Independent(Normal(d.base_dist.loc.detach(), d.base_dist.scale.detach()), 1)

    def update(self, b):
        hp = self.hp
        params = self.ac.policy_parameters()
        obs, actions = b["obs"], b["actions"]
        adv = b["advantages"]
        if hp["normalize_advantage"]:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        old_dist = self._detached_dist(obs)

        def surrogate():
            return ((self.ac.dist(obs).log_prob(actions) - b["logp"]).exp() * adv).mean()

        def mean_kl():
            return kl_divergence(old_dist, self.ac.dist(obs)).mean()

        def fvp(v):
            g = _flat_grad(mean_kl(), params, create_graph=True)
            return _flat_grad(g @ v, params) + hp["damping"] * v

        loss = surrogate()
        g = _flat_grad(loss, params)
        step_dir = _conjugate_gradient(fvp, g, int(hp["cg_iters"]))
        shs = 0.5 * (step_dir @ fvp(step_dir))
        full_step = step_dir / torch.sqrt(shs / hp["max_kl"] + 1e-10)
        old_params = _flat_params(params)
        accepted, kl = False, torch.zeros(())
        with torch.no_grad():
            for i in range(int(hp["backtrack_iters"])):
                _set_flat_params(params, old_params + (hp["backtrack_coef"] ** i) * full_step)
                new_loss, kl = surrogate(), mean_kl()
                if kl <= 1.5 * hp["max_kl"] and new_loss > loss:
                    accepted = True
                    break
            if not accepted:
                _set_flat_params(params, old_params)

        N = obs.shape[0]
        mb = max(1, N // int(hp["vf_minibatches"]))
        for _ in range(int(hp["vf_iters"])):
            perm = torch.randperm(N, device=self.device)
            for s in range(0, N, mb):
                idx = perm[s:s + mb]
                value_loss = F.mse_loss(self.ac.value(obs[idx]), b["returns"][idx])
                self.clip_and_step(self.vf_opt, value_loss, self.ac.v.parameters(), hp["max_grad_norm"])
        return {"surrogate": loss.item(), "kl": float(kl), "step_accepted": float(accepted),
                "value_loss": value_loss.item()}

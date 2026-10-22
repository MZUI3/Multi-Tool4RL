"""MAT: Multi-Agent Transformer (Wen et al., 2022), trained with PPO as a sequence model."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from rlzoo.common.buffers import compute_gae
from rlzoo.common.utils import RunningMeanStd, masked_logits
from rlzoo.marl.base import MAOnPolicyAlgo, flatten_time
from rlzoo.marl.networks import MATNetwork


class MAT(MAOnPolicyAlgo):
    name = "mat"
    default_hparams = dict(lr=5e-4, n_steps=100, gae_lambda=0.95, clip_range=0.2, n_epochs=5, n_minibatches=2,
                           ent_coef=0.01, vf_coef=1.0, max_grad_norm=10.0, normalize_advantage=True, embed_dim=64,
                           n_heads=2, n_blocks=1, value_norm=True)

    def __init__(self, spec, cfg, device, logger=None):
        super().__init__(spec, cfg, device, logger)
        hp = self.hp
        self.model = MATNetwork(self.obs_dim, self.A, self.n, hp["embed_dim"], hp["n_heads"], hp["n_blocks"]).to(device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=hp["lr"], eps=1e-5)
        self.vnorm = RunningMeanStd() if hp["value_norm"] else None

    def denorm(self, values):
        return self.vnorm.denormalize(values) if self.vnorm is not None else values

    def networks(self):
        return {"model": self.model}

    def sample(self, obs, avail, deterministic=False):
        """Autoregressive action generation, agent 0 .. n-1."""
        rep, values = self.model.encode(self.agent_inputs(obs))
        B = obs.shape[0]
        shifted = torch.zeros(B, self.n, self.A + 1, device=self.device)
        shifted[:, 0, 0] = 1
        actions = torch.zeros(B, self.n, dtype=torch.long, device=self.device)
        logps = torch.zeros(B, self.n, device=self.device)
        for i in range(self.n):
            logits = masked_logits(self.model.decode(rep, shifted[:, : i + 1])[:, i], avail[:, i])
            d = Categorical(logits=logits)
            a = d.probs.argmax(-1) if deterministic else d.sample()
            actions[:, i], logps[:, i] = a, d.log_prob(a)
            if i + 1 < self.n:
                shifted[:, i + 1, 1:] = F.one_hot(a, self.A).float()
        return actions, logps, values

    def evaluate_actions(self, obs, avail, actions):
        rep, values = self.model.encode(self.agent_inputs(obs))
        logits = self.model.decode(rep, self.model.shift(F.one_hot(actions, self.A).float()))
        d = Categorical(logits=masked_logits(logits, avail))
        return d.log_prob(actions), d.entropy(), values

    @torch.no_grad()
    def act(self, obs, state, avail, deterministic=False):
        actions, _, _ = self.sample(self.t(obs), self.t(avail), deterministic)
        return actions.cpu().numpy()

    def policy_step(self, obs, state, avail):
        actions, logp, values = self.sample(obs, avail)
        return actions, logp, self.denorm(values)

    def bootstrap_value(self, obs, state, avail):
        return self.denorm(self.model.encode(self.agent_inputs(obs))[1])

    def update(self, data):
        hp = self.hp
        dones = np.repeat(data["dones"][..., None], self.n, axis=-1)
        adv, ret = compute_gae(data["rewards"], data["values"], dones, data["last_values"], hp["gamma"],
                               hp["gae_lambda"])
        b = self.to_torch(flatten_time(dict(data, advantages=adv, returns=ret),
                                       ("obs", "avail", "actions", "logp", "advantages", "returns")))
        if hp["normalize_advantage"]:
            b["advantages"] = (b["advantages"] - b["advantages"].mean()) / (b["advantages"].std() + 1e-8)
        if self.vnorm is not None:
            self.vnorm.update(ret)
            b["returns"] = self.vnorm.normalize(b["returns"])
        N, clip = b["obs"].shape[0], hp["clip_range"]
        mb = max(1, N // int(hp["n_minibatches"]))
        stats = {"policy_loss": [], "value_loss": [], "entropy": []}
        for _ in range(int(hp["n_epochs"])):
            perm = torch.randperm(N, device=self.device)
            for s in range(0, N, mb):
                idx = perm[s:s + mb]
                logp, entropy, values = self.evaluate_actions(b["obs"][idx], b["avail"][idx], b["actions"][idx])
                ratio = (logp - b["logp"][idx]).exp()
                a = b["advantages"][idx]
                policy_loss = -torch.min(ratio * a, ratio.clamp(1 - clip, 1 + clip) * a).mean()
                value_loss = 0.5 * ((values - b["returns"][idx]) ** 2).mean()
                entropy = entropy.mean()
                loss = policy_loss + hp["vf_coef"] * value_loss - hp["ent_coef"] * entropy
                self.clip_and_step(self.opt, loss, self.model.parameters(), hp["max_grad_norm"])
                stats["policy_loss"].append(policy_loss.item())
                stats["value_loss"].append(value_loss.item())
                stats["entropy"].append(entropy.item())
        return {k: float(np.mean(v)) for k, v in stats.items()}

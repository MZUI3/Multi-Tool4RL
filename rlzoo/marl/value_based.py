"""Value-based MARL: IQL, VDN, QMIX, QTRAN, QPLEX and Weighted QMIX.

Agents share one Q-network (input: observation + one-hot id). Training samples transitions from a replay
buffer; the mixer turns per-agent Q-values into Q_tot trained on the team reward (IQL trains each agent's
Q on the team reward directly).
"""
from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from rlzoo.common.nets import mlp
from rlzoo.common.utils import linear_schedule, masked_logits, sample_available, update_target
from rlzoo.marl.base import MAOffPolicyAlgo
from rlzoo.marl.mixers import QMixer, QPLEXMixer, QattenMixer, VDNMixer


class IQL(MAOffPolicyAlgo):
    """Independent Q-learning (Tan, 1993) with parameter sharing."""

    name = "iql"
    default_hparams = dict(lr=5e-4, target_update_interval=200, tau=1.0, eps_start=1.0, eps_end=0.05,
                           eps_decay_steps=20_000, double_q=True, max_grad_norm=10.0,
                           reward_scale=1.0)  # multiply team rewards (keeps Q magnitudes moderate)

    def __init__(self, spec, cfg, device, logger=None, accel=None):
        super().__init__(spec, cfg, device, logger, accel)
        hp = self.hp
        self.agent = mlp(self.obs_dim, self.A, hp["hidden_sizes"], hp["activation"]).to(device)
        self.mixer = self.build_mixer()
        if self.mixer is not None:
            self.mixer.to(device)
        self.target_agent = copy.deepcopy(self.agent)
        self.target_mixer = copy.deepcopy(self.mixer)
        self.buffer = self.make_buffer()
        self.n_updates = 0
        self.setup_optimizer()

    def build_mixer(self):
        return None

    def trainable(self):
        """(online, target) module pairs; their online params are optimized and targets synced."""
        pairs = [(self.agent, self.target_agent)]
        if self.mixer is not None:
            pairs.append((self.mixer, self.target_mixer))
        return pairs

    def setup_optimizer(self):
        self.params = [p for online, _ in self.trainable() for p in online.parameters()]
        self.opt = torch.optim.Adam(self.params, lr=self.hp["lr"])

    def networks(self):
        nets = {"agent": self.agent}
        if self.mixer is not None:
            nets["mixer"] = self.mixer
        return nets

    # ---------------------------------------------------------------- acting
    def q_all(self, net, obs):
        return net(self.agent_inputs(obs))  # (B, n, A)

    def epsilon(self):
        hp = self.hp
        return linear_schedule(hp["eps_start"], hp["eps_end"], hp["eps_decay_steps"], self.global_step)

    @torch.no_grad()
    def act(self, obs, state, avail, deterministic=False):
        avail_t = self.t(avail)
        greedy = masked_logits(self.q_all(self.agent, self.t(obs)), avail_t).argmax(-1)
        if not deterministic:
            explore = torch.rand(greedy.shape, device=self.device) < self.epsilon()
            greedy = torch.where(explore, sample_available(avail_t), greedy)
        return greedy.cpu().numpy()

    @torch.no_grad()
    def explain(self, obs, state, avail):
        q = masked_logits(self.q_all(self.agent, self.t(obs[None])), self.t(avail[None]))[0]
        return {"q": q.cpu().numpy()}

    # ---------------------------------------------------------------- learning
    def mix(self, mixer, qs, state, actions, q_all_masked, obs=None):
        if mixer is None:
            return qs
        return mixer(qs, state, actions_onehot=F.one_hot(actions, self.A).float(),
                     max_qs=q_all_masked.max(-1).values, obs=obs)

    def next_greedy(self, b):
        """Greedy next actions (double-Q: from the online agent) and target Q-values (B, n, A)."""
        with torch.no_grad():
            target_q = masked_logits(self.q_all(self.target_agent, b["next_obs"]), b["next_avail"])
            if self.hp["double_q"]:
                next_a = masked_logits(self.q_all(self.agent, b["next_obs"]), b["next_avail"]).argmax(-1)
            else:
                next_a = target_q.argmax(-1)
        return next_a, target_q

    def td_target(self, b, value):
        r, d = b["reward"] * self.hp["reward_scale"], b["done"]
        if value.dim() == 2:
            r, d = r[:, None], d[:, None]
        return r + self.hp["gamma"] * (1 - d) * value

    def sample(self):
        _, batch = self.buffer.sample(int(self.hp["batch_size"]))
        return self.to_torch(batch)

    def compute_loss(self, b):
        q = self.q_all(self.agent, b["obs"])
        chosen = q.gather(-1, b["actions"][..., None]).squeeze(-1)
        next_a, target_q = self.next_greedy(b)
        with torch.no_grad():
            target_chosen = target_q.gather(-1, next_a[..., None]).squeeze(-1)
            y = self.td_target(b, self.mix(self.target_mixer, target_chosen, b["next_state"], next_a, target_q, b["next_obs"]))
        q_tot = self.mix(self.mixer, chosen, b["state"], b["actions"], masked_logits(q, b["avail"]), b["obs"])
        loss = F.mse_loss(q_tot, y)
        return loss, {"loss": loss.item(), "q_taken": q_tot.mean().item()}

    def update(self):
        loss, metrics = self.compute_loss(self.sample())
        self.clip_and_step(self.opt, loss, self.params, self.hp["max_grad_norm"])
        self.n_updates += 1
        for online, target in self.trainable():
            if target is not None:
                update_target(target, online, self.hp["tau"], self.n_updates, self.hp["target_update_interval"])
        metrics["epsilon"] = self.epsilon()
        return metrics


class VDN(IQL):
    """Value Decomposition Networks (Sunehag et al., 2018): Q_tot = sum_i Q_i."""

    name = "vdn"

    def build_mixer(self):
        return VDNMixer()


class QMIX(IQL):
    """QMIX (Rashid et al., 2018): monotonic state-conditioned mixing."""

    name = "qmix"
    # QMIX's hypernetwork mixer diverges at a high replay ratio (1 update / env step); these defaults mirror
    # pymarl's ~1 update per episode on a large batch.
    default_hparams = dict(mixer_embed_dim=32, hypernet_embed=64, batch_size=512, train_freq=25)

    def build_mixer(self):
        return QMixer(self.n, self.state_dim, self.hp["mixer_embed_dim"], self.hp["hypernet_embed"])


class QPLEX(IQL):
    """QPLEX (Wang et al., 2021): duplex dueling mixing with attention-based advantage weights."""

    name = "qplex"
    default_hparams = dict(hypernet_embed=64, attention_heads=4)

    def build_mixer(self):
        return QPLEXMixer(self.n, self.state_dim, self.A, self.hp["hypernet_embed"], self.hp["attention_heads"])


class Qatten(IQL):
    """Qatten (Yang et al., 2020): attention-weighted monotonic mixing (see QattenMixer)."""

    name = "qatten"
    default_hparams = dict(attention_heads=4, mixer_embed_dim=32, hypernet_embed=64, batch_size=256, train_freq=10)

    def build_mixer(self):
        hp = self.hp
        return QattenMixer(self.n, self.state_dim, self.spec.obs_dim, hp["attention_heads"], hp["mixer_embed_dim"],
                           hp["hypernet_embed"])


class QTRAN(IQL):
    """QTRAN-base (Son et al., 2019).

    A joint action-value Q_jt(s, u) and state value V_jt(s) are learned alongside the agents; the
    factorisation sum_i Q_i is tied to Q_jt with the L_opt / L_nopt constraints.
    """

    name = "qtran"
    default_hparams = dict(joint_hidden=[64, 64], opt_loss_coef=1.0, nopt_min_loss_coef=0.1)

    def __init__(self, spec, cfg, device, logger=None, accel=None):
        hp = self.resolve_hparams(cfg.get("hparams") or {})
        self.q_joint = mlp(spec.state_dim + spec.n_agents * spec.n_actions, 1, hp["joint_hidden"],
                           hp["activation"]).to(device)
        self.v_joint = mlp(spec.state_dim, 1, hp["joint_hidden"], hp["activation"]).to(device)
        self.target_q_joint = copy.deepcopy(self.q_joint)
        super().__init__(spec, cfg, device, logger, accel)

    def trainable(self):
        return [(self.agent, self.target_agent), (self.q_joint, self.target_q_joint), (self.v_joint, None)]

    def networks(self):
        return {"agent": self.agent, "q_joint": self.q_joint, "v_joint": self.v_joint}

    def joint_q(self, net, state, actions):
        onehot = F.one_hot(actions, self.A).float().flatten(1)
        return net(torch.cat([state, onehot], -1)).squeeze(-1)

    def compute_loss(self, b):
        hp = self.hp
        q = self.q_all(self.agent, b["obs"])
        q_masked = masked_logits(q, b["avail"])
        chosen = q.gather(-1, b["actions"][..., None]).squeeze(-1)
        next_a, _ = self.next_greedy(b)
        with torch.no_grad():
            y = self.td_target(b, self.joint_q(self.target_q_joint, b["next_state"], next_a))
        q_jt = self.joint_q(self.q_joint, b["state"], b["actions"])
        td_loss = F.mse_loss(q_jt, y)
        greedy = q_masked.argmax(-1)
        q_jt_greedy = self.joint_q(self.q_joint, b["state"], greedy).detach()
        v_jt = self.v_joint(b["state"]).squeeze(-1)
        opt_loss = ((q_masked.max(-1).values.sum(-1) - q_jt_greedy + v_jt) ** 2).mean()
        nopt_loss = ((chosen.sum(-1) - q_jt.detach() + v_jt).clamp(max=0) ** 2).mean()
        loss = td_loss + hp["opt_loss_coef"] * opt_loss + hp["nopt_min_loss_coef"] * nopt_loss
        return loss, {"loss": loss.item(), "td_loss": td_loss.item(), "opt_loss": opt_loss.item(),
                      "nopt_loss": nopt_loss.item()}


class WQMIX(QMIX):
    """Weighted QMIX (Rashid et al., 2020), OW (optimistic) or CW (centrally) weighting.

    An unrestricted central Q* (separate agent net + non-monotonic feed-forward mixer) provides the
    bootstrap target; the monotonic QMIX loss is down-weighted by `w_alpha` on non-preferred joint actions.
    """

    name = "wqmix"
    # the central Q* target keeps WQMIX stable at a high replay ratio, so it does not need QMIX's large batch
    default_hparams = dict(weighting="ow", w_alpha=0.1, central_mixer_hidden=[128, 128], central_loss_coef=1.0,
                           batch_size=64, train_freq=1)

    def __init__(self, spec, cfg, device, logger=None, accel=None):
        hp = self.resolve_hparams(cfg.get("hparams") or {})
        n_in = spec.obs_dim + (spec.n_agents if hp["agent_id"] else 0)
        self.central_agent = mlp(n_in, spec.n_actions, hp["hidden_sizes"], hp["activation"]).to(device)
        self.central_mixer = mlp(spec.n_agents + spec.state_dim, 1, hp["central_mixer_hidden"], "relu").to(device)
        self.target_central_agent = copy.deepcopy(self.central_agent)
        self.target_central_mixer = copy.deepcopy(self.central_mixer)
        super().__init__(spec, cfg, device, logger, accel)

    def trainable(self):
        return super().trainable() + [(self.central_agent, self.target_central_agent),
                                      (self.central_mixer, self.target_central_mixer)]

    def networks(self):
        return {**super().networks(), "central_agent": self.central_agent, "central_mixer": self.central_mixer}

    @staticmethod
    def central_tot(mixer, qs, state):
        return mixer(torch.cat([qs, state], -1)).squeeze(-1)

    def compute_loss(self, b):
        hp = self.hp
        q = self.q_all(self.agent, b["obs"])
        q_masked = masked_logits(q, b["avail"])
        chosen = q.gather(-1, b["actions"][..., None]).squeeze(-1)
        qc = self.q_all(self.central_agent, b["obs"])
        chosen_c = qc.gather(-1, b["actions"][..., None]).squeeze(-1)
        with torch.no_grad():
            next_a = masked_logits(self.q_all(self.agent, b["next_obs"]), b["next_avail"]).argmax(-1)
            target_qc = self.q_all(self.target_central_agent, b["next_obs"]).gather(-1, next_a[..., None]).squeeze(-1)
            y = self.td_target(b, self.central_tot(self.target_central_mixer, target_qc, b["next_state"]))
        central_loss = F.mse_loss(self.central_tot(self.central_mixer, chosen_c, b["state"]), y)
        q_tot = self.mixer(chosen, b["state"])
        td = q_tot - y
        with torch.no_grad():
            ones = torch.ones_like(td)
            alpha = torch.full_like(td, hp["w_alpha"])
            if hp["weighting"] == "cw":
                greedy = q_masked.argmax(-1)
                is_greedy = (b["actions"] == greedy).all(-1)
                q_star_greedy = self.central_tot(self.central_mixer, qc.gather(-1, greedy[..., None]).squeeze(-1),
                                                 b["state"])
                w = torch.where(is_greedy | (y > q_star_greedy), ones, alpha)
            else:  # "ow": full weight where Q_tot underestimates the target
                w = torch.where(td < 0, ones, alpha)
        qmix_loss = (w * td ** 2).mean()
        loss = qmix_loss + hp["central_loss_coef"] * central_loss
        return loss, {"loss": loss.item(), "qmix_loss": qmix_loss.item(), "central_loss": central_loss.item()}

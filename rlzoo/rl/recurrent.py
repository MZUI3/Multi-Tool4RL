"""Recurrent agents for partially observable tasks: DRQN and recurrent PPO (GRU)."""
from __future__ import annotations

import copy
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from rlzoo.common.nets import RecurrentActorCritic, RecurrentQNetwork
from rlzoo.common.utils import linear_schedule, update_target
from rlzoo.rl.base import OffPolicyAlgo, RLAlgorithm


class EpisodeBuffer:
    """Stores whole episodes and samples fixed-length sub-sequences (zero padded, with a mask)."""

    def __init__(self, max_episodes: int, seq_len: int):
        self.episodes = deque(maxlen=int(max_episodes))
        self.seq_len = int(seq_len)

    def __len__(self):
        return len(self.episodes)

    def add_episode(self, steps):
        if steps:
            self.episodes.append({k: np.stack([s[k] for s in steps]) for k in steps[0]})

    def sample(self, batch_size: int):
        L = self.seq_len
        idx = np.random.randint(0, len(self.episodes), batch_size)
        out, mask = None, np.zeros((batch_size, L), np.float32)
        for i, j in enumerate(idx):
            ep = self.episodes[j]
            T = ep["reward"].shape[0]
            start = np.random.randint(0, max(1, T - L + 1))
            n = min(L, T - start)
            if out is None:
                out = {k: np.zeros((batch_size, L, *v.shape[1:]), v.dtype) for k, v in ep.items()}
            for k, v in ep.items():
                out[k][i, :n] = v[start:start + n]
            mask[i, :n] = 1.0
        out["mask"] = mask
        return out


class DRQN(OffPolicyAlgo):
    """Deep Recurrent Q-Network (Hausknecht & Stone, 2015). The GRU hidden state is carried across steps while
    acting; training samples `seq_len`-step windows of stored episodes and unrolls both networks from a zero state."""

    name = "drqn"
    default_hparams = dict(lr=1e-3, buffer_episodes=500, seq_len=16, rnn_hidden=64, batch_size=32, double=True,
                           target_update_interval=500, tau=1.0, eps_start=1.0, eps_end=0.05, eps_decay_steps=10_000,
                           max_grad_norm=10.0, learning_starts=500)

    def __init__(self, spec, cfg, device, logger=None, accel=None):
        super().__init__(spec, cfg, device, logger, accel)
        if not spec.discrete:
            raise TypeError("drqn requires a discrete action space")
        hp = self.hp
        self.q = RecurrentQNetwork(spec.obs_dim, spec.n_actions, hp["hidden_sizes"], hp["activation"],
                                   int(hp["rnn_hidden"]), encoder=self.make_encoder()).to(device)
        self.q_target = copy.deepcopy(self.q)
        self.opt = torch.optim.Adam(self.q.parameters(), lr=hp["lr"])
        self.buffer = EpisodeBuffer(hp["buffer_episodes"], hp["seq_len"])
        self._partial = {}
        self._hidden = {}
        self.n_updates = 0

    def networks(self):
        return {"q": self.q, "q_target": self.q_target}

    # ---------------------------------------------------------------- hidden state
    def reset_hidden(self, n: int):
        self._hidden[n] = torch.zeros(1, n, self.q.rnn_hidden, device=self.device)

    def mask_hidden(self, dones):
        n = len(dones)
        if n in self._hidden:
            self._hidden[n] = self._hidden[n] * (1.0 - self.t(dones)).view(1, n, 1)

    def epsilon(self):
        hp = self.hp
        return linear_schedule(hp["eps_start"], hp["eps_end"], hp["eps_decay_steps"], self.global_step)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        n = obs.shape[0]
        if n not in self._hidden:
            self.reset_hidden(n)
        q, h = self.q(self.t(obs).unsqueeze(1), self._hidden[n])
        self._hidden[n] = h
        greedy = q[:, 0].argmax(-1).cpu().numpy()
        if deterministic:
            return greedy
        explore = np.random.rand(n) < self.epsilon()
        return np.where(explore, np.random.randint(self.spec.n_actions, size=n), greedy)

    @torch.no_grad()
    def explain(self, obs):
        n = 1
        if n not in self._hidden:
            self.reset_hidden(n)
        q, _ = self.q(self.t(np.asarray(obs)[None]).unsqueeze(1), self._hidden[n])  # peek, do not advance
        return {"q": q[0, 0].cpu().numpy()}

    # ---------------------------------------------------------------- storage
    def store(self, env_idx, obs, action, reward, next_obs, terminated, done):
        self._partial.setdefault(env_idx, []).append(
            dict(obs=obs, action=np.int64(action), reward=np.float32(reward), next_obs=next_obs, done=np.float32(terminated)))
        if done:
            self.buffer.add_episode(self._partial.pop(env_idx))

    # ---------------------------------------------------------------- learning
    def update(self):
        hp = self.hp
        b = self.to_torch(self.buffer.sample(int(hp["batch_size"])))
        q, _ = self.q(b["obs"])
        q = q.gather(-1, b["action"][..., None]).squeeze(-1)  # (B, L)
        with torch.no_grad():
            next_target, _ = self.q_target(b["next_obs"])
            if hp["double"]:
                next_online, _ = self.q(b["next_obs"])
                next_q = next_target.gather(-1, next_online.argmax(-1, keepdim=True)).squeeze(-1)
            else:
                next_q = next_target.max(-1).values
            target = b["reward"] + hp["gamma"] * (1 - b["done"]) * next_q
        loss = (F.smooth_l1_loss(q, target, reduction="none") * b["mask"]).sum() / b["mask"].sum()
        self.clip_and_step(self.opt, loss, self.q.parameters(), hp["max_grad_norm"])
        self.n_updates += 1
        update_target(self.q_target, self.q, hp["tau"], self.n_updates, hp["target_update_interval"])
        return {"loss": loss.item(), "epsilon": self.epsilon()}


class RecurrentPPO(RLAlgorithm):
    """PPO with a GRU policy/value core (Schulman et al., 2017 + recurrent state, as in SB3-contrib's
    RecurrentPPO). Minibatches are whole per-environment rollouts so the recurrent state can be replayed."""

    name = "ppo_rnn"
    default_hparams = dict(lr=3e-4, n_steps=128, n_epochs=4, n_minibatches=4, gae_lambda=0.95, clip_range=0.2,
                           ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, rnn_hidden=64, activation="tanh",
                           normalize_advantage=True, log_std_init=0.0)

    def __init__(self, spec, cfg, device, logger=None, accel=None):
        super().__init__(spec, cfg, device, logger, accel)
        hp = self.hp
        self.ac = RecurrentActorCritic(spec.obs_dim, spec, hp["hidden_sizes"], hp["activation"], int(hp["rnn_hidden"]),
                                       hp["log_std_init"], encoder=self.make_encoder()).to(device)
        self.opt = torch.optim.Adam(self.ac.parameters(), lr=hp["lr"], eps=1e-5)
        self._hidden = {}

    def networks(self):
        return {"ac": self.ac}

    def reset_hidden(self, n: int):
        self._hidden[n] = torch.zeros(1, n, self.ac.rnn_hidden, device=self.device)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        n = obs.shape[0]
        if n not in self._hidden:
            self.reset_hidden(n)
        y, h = self.ac.core(self.t(obs).unsqueeze(1), self._hidden[n])
        self._hidden[n] = h
        dist = self.ac.dist_from(y[:, 0])
        a = self.ac.mode(dist) if deterministic else dist.sample()
        return a.cpu().numpy()

    @torch.no_grad()
    def explain(self, obs):
        n = 1
        if n not in self._hidden:
            self.reset_hidden(n)
        y, _ = self.ac.core(self.t(np.asarray(obs)[None]).unsqueeze(1), self._hidden[n])
        dist = self.ac.dist_from(y[:, 0])
        out = {"value": float(self.ac.v(y[:, 0])[0, 0])}
        if self.ac.discrete:
            out["probs"] = dist.probs[0].cpu().numpy()
        else:
            out["mean"] = dist.mean[0].cpu().numpy()
        return out

    def train(self, venv, total_steps):
        hp = self.hp
        E, T, gamma, lam = venv.num_envs, int(hp["n_steps"]), hp["gamma"], hp["gae_lambda"]
        obs = venv.reset()
        self.reset_hidden(E)
        while self.global_step < total_steps:
            h0 = self._hidden[E].clone()
            buf = {k: [] for k in ("obs", "actions", "logp", "values", "rewards", "dones")}
            episodes = []
            for _ in range(T):
                with torch.no_grad():
                    y, h = self.ac.core(self.t(obs).unsqueeze(1), self._hidden[E])
                    dist = self.ac.dist_from(y[:, 0])
                    action = dist.sample()
                    logp, value = dist.log_prob(action), self.ac.v(y[:, 0]).squeeze(-1)
                a_np = action.cpu().numpy()
                next_obs, rewards, terms, truncs, info = venv.step(a_np)
                rewards = rewards.astype(np.float32).copy()
                dones = np.logical_or(terms, truncs)
                for i in range(E):
                    if truncs[i] and not terms[i]:
                        with torch.no_grad():
                            yf, _ = self.ac.core(self.t(info["final_obs"][i][None]).unsqueeze(1), h[:, i:i + 1])
                            rewards[i] += gamma * self.ac.v(yf[:, 0]).item()
                self._hidden[E] = h * (1.0 - self.t(dones)).view(1, E, 1)
                for k, v in (("obs", obs), ("actions", a_np), ("logp", logp.cpu().numpy()),
                             ("values", value.cpu().numpy()), ("rewards", rewards), ("dones", dones.astype(np.float32))):
                    buf[k].append(v)
                episodes.extend(info["episodes"])
                obs = next_obs
            with torch.no_grad():
                y, _ = self.ac.core(self.t(obs).unsqueeze(1), self._hidden[E])
                last_value = self.ac.v(y[:, 0]).squeeze(-1).cpu().numpy()
            data = {k: np.stack(v) for k, v in buf.items()}  # (T, E, ...)
            from rlzoo.common.buffers import compute_gae

            adv, ret = compute_gae(data["rewards"], data["values"], data["dones"], last_value, gamma, lam)
            data["advantages"], data["returns"] = adv, ret
            self.global_step += T * E * self.step_scale
            self.record_episodes(episodes)
            self.train_metrics = self.update(data, h0)
            self.maybe_log()

    def update(self, data, h0):
        hp = self.hp
        E, clip = data["obs"].shape[1], hp["clip_range"]
        b = self.to_torch({k: np.swapaxes(v, 0, 1) for k, v in data.items()})  # (E, T, ...)
        n_mb = max(1, min(int(hp["n_minibatches"]), E))
        stats = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": []}
        for _ in range(int(hp["n_epochs"])):
            perm = torch.randperm(E, device=self.device)
            for chunk in perm.chunk(n_mb):
                y, _ = self.ac.core(b["obs"][chunk], h0[:, chunk], b["dones"][chunk])
                dist = self.ac.dist_from(y)
                logp = dist.log_prob(b["actions"][chunk])
                entropy = dist.entropy().mean()
                log_ratio = logp - b["logp"][chunk]
                ratio = log_ratio.exp()
                adv = b["advantages"][chunk]
                if hp["normalize_advantage"] and adv.numel() > 1:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                policy_loss = -torch.min(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv).mean()
                value_loss = 0.5 * ((self.ac.v(y).squeeze(-1) - b["returns"][chunk]) ** 2).mean()
                loss = policy_loss + hp["vf_coef"] * value_loss - hp["ent_coef"] * entropy
                self.clip_and_step(self.opt, loss, self.ac.parameters(), hp["max_grad_norm"])
                stats["policy_loss"].append(policy_loss.item())
                stats["value_loss"].append(value_loss.item())
                stats["entropy"].append(entropy.item())
                stats["approx_kl"].append(((ratio - 1) - log_ratio).mean().item())
        return {k: float(np.mean(v)) for k, v in stats.items()}

"""Training loops shared by single-agent off-policy and on-policy algorithms."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from rlzoo.common.base import Algorithm
from rlzoo.common.buffers import ReplayBuffer, compute_gae
from rlzoo.common.nets import ActorCritic, ObsEncoder


class RLAlgorithm(Algorithm):
    kind = "rl"
    default_hparams = dict(
        gamma=0.99,
        hidden_sizes=[64, 64],
        activation="relu",
        encoder="auto",  # mlp | auto | nature_cnn | small_cnn | impala_cnn (CNNs need image observations)
        cnn_features=256,
    )

    def make_encoder(self):
        """A fresh observation encoder for one network (None for plain vector observations)."""
        enc = ObsEncoder(self.spec, self.hp["encoder"], int(self.hp["cnn_features"]))
        return None if enc.kind == "mlp" else enc


class OffPolicyAlgo(RLAlgorithm):
    default_hparams = dict(
        buffer_size=100_000,
        batch_size=64,
        learning_starts=1_000,  # env steps before the first gradient update
        train_freq=1,  # vector-env steps between update rounds
        gradient_steps=1,  # gradient updates per round
        random_steps=0,  # env steps with uniformly random actions
        her=False,  # Hindsight Experience Replay (needs a goal-conditioned env: BitFlip-v0, PointReach-v0, ...)
        her_k=4,  # relabelled copies per transition
        her_strategy="future",  # future | final | episode
    )

    def buffer_fields(self):
        spec = self.spec
        act = ((), np.int64) if spec.discrete else ((spec.act_dim,), np.float32)
        return {
            "obs": ((spec.obs_dim,), spec.obs_dtype),
            "action": act,
            "reward": ((), np.float32),
            "next_obs": ((spec.obs_dim,), spec.obs_dtype),
            "done": ((), np.float32),
            "discount": ((), np.float32),
        }

    def make_buffer(self):
        return ReplayBuffer(int(self.hp["buffer_size"]), self.buffer_fields())

    def store(self, env_idx, obs, action, reward, next_obs, terminated, done):
        self.buffer.add(obs=obs, action=action, reward=reward, next_obs=next_obs, done=float(terminated),
                        discount=self.hp["gamma"])

    # ---------------------------------------------------------------- HER
    def her_store(self, env_idx, obs, action, reward, next_obs, terminated, done):
        """Buffer an episode and, when it ends, store it plus `her_k` goal-relabelled copies per step."""
        eps = self._her_episodes
        eps[env_idx].append((obs, action, reward, next_obs, terminated))
        if not done:
            return
        episode, eps[env_idx] = eps[env_idx], []
        T, g, k, strategy = len(episode), self.spec.goal_dim, int(self.hp["her_k"]), self.hp["her_strategy"]
        for t, (o, a, r, o2, term) in enumerate(episode):
            self.store(env_idx, o, a, r, o2, term, t == T - 1)
        for t, (o, a, r, o2, term) in enumerate(episode):
            for _ in range(k):
                if strategy == "final":
                    j = T - 1
                elif strategy == "episode":
                    j = np.random.randint(0, T)
                else:  # future
                    j = np.random.randint(t, T)
                new_goal = episode[j][3][-2 * g:-g]  # achieved goal after step j
                o_new, o2_new = o.copy(), o2.copy()
                o_new[-g:] = new_goal
                o2_new[-g:] = new_goal
                r_new = float(self._compute_reward(o2_new[-2 * g:-g][None], new_goal[None])[0])
                success = r_new > r_new - 1 and r_new >= 0.0  # sparse envs: 0 on success, negative otherwise
                # each relabelled step is pushed as its own (truncated) episode so n-step chains never mix goals
                self.store(env_idx, o_new, a, r_new, o2_new, bool(success), True)

    def update(self):
        raise NotImplementedError

    def train(self, venv, total_steps):
        hp = self.hp
        use_her = bool(hp["her"])
        if use_her:
            if self.spec.goal_dim <= 0:
                raise ValueError("her=true needs a goal-conditioned env with a Dict observation "
                                 "(observation / achieved_goal / desired_goal), e.g. BitFlip-v0 or PointReach-v0")
            self._her_episodes = [[] for _ in range(venv.num_envs)]
            self._compute_reward = venv.compute_reward
        store = self.her_store if use_her else self.store
        obs = venv.reset()
        if hasattr(self, "reset_hidden"):
            self.reset_hidden(venv.num_envs)
        it = 0
        while self.global_step < total_steps:
            if self.global_step < hp["random_steps"]:
                actions = venv.sample_actions()
            else:
                actions = self.act(obs, deterministic=False)
            next_obs, rewards, terms, truncs, info = venv.step(actions)
            for i in range(venv.num_envs):
                done = bool(terms[i] or truncs[i])
                real_next = info["final_obs"][i] if done else next_obs[i]
                store(i, obs[i], actions[i], rewards[i], real_next, bool(terms[i]), done)
            if hasattr(self, "reset_hidden"):
                self.mask_hidden(np.logical_or(terms, truncs))
            obs = next_obs
            self.global_step += venv.num_envs * self.step_scale
            it += 1
            self.record_episodes(info["episodes"])
            ready = (self.global_step >= hp["learning_starts"] and it % int(hp["train_freq"]) == 0
                     and len(self.buffer) >= hp["batch_size"])
            if self.agree(ready):
                for _ in range(int(hp["gradient_steps"])):
                    self.train_metrics = self.update()
            self.maybe_log()


class OnPolicyAlgo(RLAlgorithm):
    default_hparams = dict(
        lr=3e-4,
        n_steps=128,  # rollout length per env
        gae_lambda=0.95,
        activation="tanh",
        normalize_advantage=True,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        log_std_init=0.0,
        ortho_init=True,
    )

    def __init__(self, spec, cfg, device, logger=None, accel=None):
        super().__init__(spec, cfg, device, logger, accel)
        hp = self.hp
        self.ac = self.build_actor_critic().to(device)

    def build_actor_critic(self):
        hp = self.hp
        return ActorCritic(self.spec.obs_dim, self.spec, hp["hidden_sizes"], hp["activation"], hp["log_std_init"],
                           hp["ortho_init"], encoder_fn=self.make_encoder)

    def networks(self):
        return {"ac": self.ac}

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        dist = self.ac.dist(self.t(obs))
        a = self.ac.mode(dist) if deterministic else dist.sample()
        return a.cpu().numpy()

    def reward_hook(self, obs, actions, next_obs, rewards):
        """Hook for intrinsic-reward methods (RND / ICM); returns the rewards actually used for training."""
        return rewards

    def collect_rollout(self, ac, venv, obs, n_steps):
        """Collect n_steps x num_envs transitions with policy `ac`; returns (flat batch, last obs, episodes)."""
        gamma, E = self.hp["gamma"], venv.num_envs
        keys = ("obs", "actions", "logp", "values", "rewards", "dones")
        buf = {k: [] for k in keys}
        episodes = []
        for _ in range(int(n_steps)):
            with torch.no_grad():
                obs_t = self.t(obs)
                dist = ac.dist(obs_t)
                action = dist.sample()
                logp = dist.log_prob(action)
                value = ac.value(obs_t)
            a_np = action.cpu().numpy()
            next_obs, rewards, terms, truncs, info = venv.step(a_np)
            rewards = rewards.astype(np.float32).copy()
            true_next = next_obs
            if any(f is not None for f in info["final_obs"]):
                true_next = np.stack([f if f is not None else n for f, n in zip(info["final_obs"], next_obs)])
            rewards = self.reward_hook(obs, a_np, true_next, rewards)
            for i in range(E):  # bootstrap time-limit truncations
                if truncs[i] and not terms[i]:
                    with torch.no_grad():
                        rewards[i] += gamma * ac.value(self.t(info["final_obs"][i][None])).item()
            buf["obs"].append(obs)
            buf["actions"].append(a_np)
            buf["logp"].append(logp.cpu().numpy())
            buf["values"].append(value.cpu().numpy())
            buf["rewards"].append(rewards)
            buf["dones"].append(np.logical_or(terms, truncs).astype(np.float32))
            episodes.extend(info["episodes"])
            obs = next_obs
        with torch.no_grad():
            last_value = ac.value(self.t(obs)).cpu().numpy()
        data = {k: np.stack(v) for k, v in buf.items()}
        adv, ret = compute_gae(data["rewards"], data["values"], data["dones"], last_value, gamma,
                               self.hp["gae_lambda"])
        data["advantages"], data["returns"] = adv, ret
        T = data["rewards"].shape[0]
        batch = {k: v.reshape(T * E, *v.shape[2:]) for k, v in data.items() if k not in ("rewards", "dones")}
        return batch, obs, episodes

    def actor_critic_loss(self, ac, b):
        """Vanilla advantage actor-critic loss (A2C / A3C)."""
        dist = ac.dist(b["obs"])
        logp = dist.log_prob(b["actions"])
        entropy = dist.entropy().mean()
        adv = b["advantages"]
        if self.hp["normalize_advantage"] and adv.numel() > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        policy_loss = -(logp * adv).mean()
        value_loss = F.mse_loss(ac.value(b["obs"]), b["returns"])
        loss = policy_loss + self.hp["vf_coef"] * value_loss - self.hp["ent_coef"] * entropy
        return loss, {"policy_loss": policy_loss.item(), "value_loss": value_loss.item(), "entropy": entropy.item()}

    def update(self, batch):
        raise NotImplementedError

    def train(self, venv, total_steps):
        obs = venv.reset()
        while self.global_step < total_steps:
            batch, obs, episodes = self.collect_rollout(self.ac, venv, obs, self.hp["n_steps"])
            self.global_step += int(self.hp["n_steps"]) * venv.num_envs * self.step_scale
            self.record_episodes(episodes)
            self.train_metrics = self.update(self.to_torch(batch))
            self.maybe_log()

    @torch.no_grad()
    def explain(self, obs):
        obs_t = self.t(np.asarray(obs)[None])
        dist = self.ac.dist(obs_t)
        out = {"value": float(self.ac.value(obs_t)[0])}
        if self.ac.discrete:
            out["probs"] = dist.probs[0].cpu().numpy()
        else:
            out["mean"] = dist.mean[0].cpu().numpy()
            out["std"] = dist.stddev[0].cpu().numpy()
        return out

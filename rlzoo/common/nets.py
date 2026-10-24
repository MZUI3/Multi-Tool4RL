from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Independent, Normal

ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "leaky_relu": nn.LeakyReLU,
    "silu": nn.SiLU,
}


def get_activation(name: str):
    try:
        return ACTIVATIONS[name.lower()]
    except KeyError:
        raise ValueError(f"Unknown activation '{name}'. Choose from {sorted(ACTIVATIONS)}") from None


def mlp_body(in_dim: int, hidden_sizes: Sequence[int], activation: str = "relu", layer_norm: bool = False,
             dropout: float = 0.0) -> Tuple[nn.Sequential, int]:
    layers, last = [], in_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(last, int(h)))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        if layer_norm:
            layers.append(nn.LayerNorm(int(h)))
        layers.append(get_activation(activation)())
        last = int(h)
    return nn.Sequential(*layers), last


def mlp(in_dim: int, out_dim: int, hidden_sizes: Sequence[int] = (64, 64), activation: str = "relu",
        out_activation: str | None = None, layer_norm: bool = False, dropout: float = 0.0) -> nn.Sequential:
    body, last = mlp_body(in_dim, hidden_sizes, activation, layer_norm, dropout)
    layers = list(body) + [nn.Linear(last, out_dim)]
    if out_activation:
        layers.append(get_activation(out_activation)())
    return nn.Sequential(*layers)


def orthogonal_init(module: nn.Module, gain: float = math.sqrt(2), last_gain: float | None = None) -> None:
    linears = [m for m in module.modules() if isinstance(m, nn.Linear)]
    for i, m in enumerate(linears):
        g = last_gain if (last_gain is not None and i == len(linears) - 1) else gain
        nn.init.orthogonal_(m.weight, g)
        nn.init.zeros_(m.bias)


# ------------------------------------------------------------------------------ observation encoders


class ObsEncoder(nn.Module):
    """Maps flat observations (B, obs_dim) to a feature vector.

    kind: "mlp" (identity: the MLP heads consume the raw observation), "nature_cnn" (Mnih et al., 2015),
    "impala_cnn" (Espeholt et al., 2018), "small_cnn" (for tiny grids), or "auto" (CNN for image
    observations, identity otherwise). Image observations arrive flattened; `spec.obs_shape` restores
    (H, W, C) or (C, H, W) and uint8 images are scaled to [0, 1].
    """

    def __init__(self, spec, kind: str = "auto", cnn_features: int = 256):
        super().__init__()
        shape = tuple(spec.obs_shape) if getattr(spec, "obs_shape", None) is not None else (spec.obs_dim,)
        self.shape = shape
        self.is_image = len(shape) == 3
        self.channels_last = self.is_image and shape[-1] in (1, 3, 4) and shape[0] not in (1, 3, 4)
        self.scale = 1.0 / 255.0 if getattr(spec, "obs_dtype", None) == np.uint8 else 1.0
        if kind == "auto":
            kind = ("small_cnn" if min(shape[:2] if self.channels_last else shape[1:]) < 48 else "nature_cnn") if self.is_image else "mlp"
        self.kind = kind
        if kind == "mlp" or not self.is_image:
            if kind != "mlp":
                raise ValueError(f"encoder '{kind}' needs image observations, got obs_shape={shape}")
            self.net = nn.Identity()
            self.out_dim = int(np.prod(shape))
            return
        c, h, w = (shape[2], shape[0], shape[1]) if self.channels_last else shape
        if kind == "nature_cnn":
            conv = nn.Sequential(nn.Conv2d(c, 32, 8, 4), nn.ReLU(), nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, 1), nn.ReLU(), nn.Flatten())
        elif kind == "small_cnn":
            conv = nn.Sequential(nn.Conv2d(c, 16, 3, 1, 1), nn.ReLU(), nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
                                 nn.Conv2d(32, 32, 3, 2, 1), nn.ReLU(), nn.Flatten())
        elif kind == "impala_cnn":
            conv = ImpalaCNN(c)
        else:
            raise ValueError(f"Unknown encoder '{kind}'. Choose from mlp, auto, nature_cnn, small_cnn, impala_cnn")
        with torch.no_grad():
            n = conv(torch.zeros(1, c, h, w)).shape[1]
        self.net = nn.Sequential(conv, nn.Linear(n, cnn_features), nn.ReLU())
        self.out_dim = cnn_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.is_image or self.kind == "mlp":
            return x
        lead = x.shape[:-1]
        x = x.reshape(-1, *self.shape) * self.scale
        if self.channels_last:
            x = x.permute(0, 3, 1, 2)
        return self.net(x).reshape(*lead, self.out_dim)


class _ImpalaBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, 1, 1)
        self.res1 = nn.Sequential(nn.ReLU(), nn.Conv2d(c_out, c_out, 3, 1, 1), nn.ReLU(), nn.Conv2d(c_out, c_out, 3, 1, 1))
        self.res2 = nn.Sequential(nn.ReLU(), nn.Conv2d(c_out, c_out, 3, 1, 1), nn.ReLU(), nn.Conv2d(c_out, c_out, 3, 1, 1))

    def forward(self, x):
        x = F.max_pool2d(self.conv(x), 3, 2, 1)
        x = x + self.res1(x)
        return x + self.res2(x)


class ImpalaCNN(nn.Sequential):
    def __init__(self, in_channels, channels=(16, 32, 32)):
        blocks, c = [], in_channels
        for co in channels:
            blocks.append(_ImpalaBlock(c, co))
            c = co
        super().__init__(*blocks, nn.ReLU(), nn.Flatten())


def _encoder_or_identity(encoder: Optional[nn.Module], obs_dim: int) -> Tuple[nn.Module, int]:
    if encoder is None:
        return nn.Identity(), obs_dim
    return encoder, encoder.out_dim


# ------------------------------------------------------------------------------------ noisy layers


class NoisyLinear(nn.Module):
    """Factorised Gaussian NoisyNet layer (Fortunato et al., 2018). Uses mean weights in eval mode."""

    def __init__(self, in_features: int, out_features: int, sigma0: float = 0.5):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("weight_eps", torch.zeros(out_features, in_features))
        self.register_buffer("bias_eps", torch.zeros(out_features))
        bound = 1.0 / math.sqrt(in_features)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.weight_sigma, sigma0 / math.sqrt(in_features))
        nn.init.constant_(self.bias_sigma, sigma0 / math.sqrt(in_features))
        self.reset_noise()

    def _scaled_noise(self, size: int) -> torch.Tensor:
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign() * x.abs().sqrt()

    @torch.no_grad()
    def reset_noise(self) -> None:
        eps_in, eps_out = self._scaled_noise(self.in_features), self._scaled_noise(self.out_features)
        self.weight_eps.copy_(torch.outer(eps_out, eps_in))
        self.bias_eps.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return F.linear(x, self.weight_mu + self.weight_sigma * self.weight_eps,
                            self.bias_mu + self.bias_sigma * self.bias_eps)
        return F.linear(x, self.weight_mu, self.bias_mu)


# -------------------------------------------------------------------------------------- Q-networks


class QNetwork(nn.Module):
    """Q-network covering DQN variants: optional dueling heads, noisy heads and distributional atoms.

    Returns (B, A) when n_atoms == 1, otherwise (B, A, n_atoms) raw outputs (logits or quantiles).
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes: Sequence[int], activation: str = "relu",
                 dueling: bool = False, noisy: bool = False, n_atoms: int = 1, sigma0: float = 0.5,
                 encoder: Optional[nn.Module] = None):
        super().__init__()
        self.n_actions, self.n_atoms, self.dueling = n_actions, n_atoms, dueling
        self.encoder, in_dim = _encoder_or_identity(encoder, obs_dim)
        self.body, last = mlp_body(in_dim, hidden_sizes, activation)
        make = (lambda i, o: NoisyLinear(i, o, sigma0)) if noisy else nn.Linear
        self.adv_head = make(last, n_actions * n_atoms)
        self.val_head = make(last, n_atoms) if dueling else None

    def features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.body(self.encoder(obs))

    def heads(self, h: torch.Tensor) -> torch.Tensor:
        a = self.adv_head(h).view(-1, self.n_actions, self.n_atoms)
        if self.dueling:
            v = self.val_head(h).view(-1, 1, self.n_atoms)
            a = v + a - a.mean(1, keepdim=True)
        return a.squeeze(-1) if self.n_atoms == 1 else a

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.heads(self.features(obs))

    def reset_noise(self) -> None:
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()


class IQNNetwork(QNetwork):
    """Implicit Quantile Network (Dabney et al., 2018): quantile fractions tau are embedded with cosines and
    multiplied into the state features; the heads then output one value per sampled tau."""

    def __init__(self, obs_dim, n_actions, hidden_sizes, activation="relu", dueling=False, noisy=False,
                 n_cos: int = 64, sigma0: float = 0.5, encoder=None):
        super().__init__(obs_dim, n_actions, hidden_sizes, activation, dueling, noisy, 1, sigma0, encoder)
        self.n_cos = n_cos
        _, last = mlp_body(self.encoder.out_dim if encoder is not None else obs_dim, hidden_sizes, activation)
        self.cos_embed = nn.Linear(n_cos, last)
        self.register_buffer("i_pi", math.pi * torch.arange(1, n_cos + 1).float())

    def forward(self, obs: torch.Tensor, taus: torch.Tensor) -> torch.Tensor:
        """obs (B, d), taus (B, N) -> quantile values (B, N, A)"""
        B, N = taus.shape
        h = self.features(obs)  # (B, last)
        phi = F.relu(self.cos_embed(torch.cos(taus.unsqueeze(-1) * self.i_pi)))  # (B, N, last)
        out = self.heads((h.unsqueeze(1) * phi).reshape(B * N, -1))  # (B*N, A)
        return out.view(B, N, self.n_actions)


class RecurrentQNetwork(nn.Module):
    """DRQN (Hausknecht & Stone, 2015): encoder -> MLP -> GRU -> Q head. Works on (B, T, d) sequences."""

    def __init__(self, obs_dim, n_actions, hidden_sizes, activation="relu", rnn_hidden=64, encoder=None):
        super().__init__()
        self.encoder, in_dim = _encoder_or_identity(encoder, obs_dim)
        self.body, last = mlp_body(in_dim, hidden_sizes, activation)
        self.rnn = nn.GRU(last, rnn_hidden, batch_first=True)
        self.head = nn.Linear(rnn_hidden, n_actions)
        self.rnn_hidden = rnn_hidden

    def forward(self, obs: torch.Tensor, h0: Optional[torch.Tensor] = None):
        """obs (B, T, d), h0 (1, B, H) -> q (B, T, A), h (1, B, H)"""
        x = self.body(self.encoder(obs))
        y, h = self.rnn(x, h0)
        return self.head(y), h


# ------------------------------------------------------------------------------------ actor-critic


class ActorCritic(nn.Module):
    """Separate policy / value MLPs; categorical or diagonal Gaussian policy."""

    def __init__(self, obs_dim: int, spec, hidden_sizes: Sequence[int], activation: str = "tanh",
                 log_std_init: float = 0.0, ortho_init: bool = True, encoder_fn=None):
        super().__init__()
        self.discrete = spec.discrete
        out = spec.n_actions if self.discrete else spec.act_dim
        enc_pi, in_pi = _encoder_or_identity(encoder_fn() if encoder_fn else None, obs_dim)
        enc_v, in_v = _encoder_or_identity(encoder_fn() if encoder_fn else None, obs_dim)
        self.pi = nn.Sequential(enc_pi, mlp(in_pi, out, hidden_sizes, activation))
        self.v = nn.Sequential(enc_v, mlp(in_v, 1, hidden_sizes, activation))
        if not self.discrete:
            self.log_std = nn.Parameter(torch.full((out,), float(log_std_init)))
        if ortho_init:
            orthogonal_init(self.pi[1], last_gain=0.01)
            orthogonal_init(self.v[1], last_gain=1.0)

    def dist(self, obs: torch.Tensor):
        out = self.pi(obs)
        if self.discrete:
            return Categorical(logits=out)
        return Independent(Normal(out, self.log_std.exp().expand_as(out)), 1)

    def mode(self, dist) -> torch.Tensor:
        return dist.probs.argmax(-1) if self.discrete else dist.mean

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.v(obs).squeeze(-1)

    def policy_parameters(self):
        params = list(self.pi.parameters())
        if not self.discrete:
            params.append(self.log_std)
        return params


class RecurrentActorCritic(nn.Module):
    """Policy and value heads on top of a shared encoder -> MLP -> GRU (recurrent PPO)."""

    def __init__(self, obs_dim, spec, hidden_sizes, activation="tanh", rnn_hidden=64, log_std_init=0.0, encoder=None):
        super().__init__()
        self.discrete = spec.discrete
        out = spec.n_actions if self.discrete else spec.act_dim
        self.encoder, in_dim = _encoder_or_identity(encoder, obs_dim)
        self.body, last = mlp_body(in_dim, hidden_sizes, activation)
        self.rnn = nn.GRU(last, rnn_hidden, batch_first=True)
        self.pi = nn.Linear(rnn_hidden, out)
        self.v = nn.Linear(rnn_hidden, 1)
        self.rnn_hidden = rnn_hidden
        if not self.discrete:
            self.log_std = nn.Parameter(torch.full((out,), float(log_std_init)))
        orthogonal_init(self.body)
        nn.init.orthogonal_(self.pi.weight, 0.01)
        nn.init.orthogonal_(self.v.weight, 1.0)

    def core(self, obs, h0=None, dones=None):
        """obs (B, T, d); dones (B, T) resets the hidden state *after* a terminal step."""
        x = self.body(self.encoder(obs))
        if dones is None:
            y, h = self.rnn(x, h0)
            return y, h
        outs, h = [], h0
        for t in range(x.shape[1]):
            y, h = self.rnn(x[:, t:t + 1], h)
            outs.append(y)
            h = h * (1.0 - dones[:, t]).view(1, -1, 1)
        return torch.cat(outs, 1), h

    def dist_from(self, y):
        out = self.pi(y)
        if self.discrete:
            return Categorical(logits=out)
        return Independent(Normal(out, self.log_std.exp().expand_as(out)), 1)

    def mode(self, dist):
        return dist.probs.argmax(-1) if self.discrete else dist.mean


class PPGNetwork(ActorCritic):
    """Actor-critic with an auxiliary value head on the policy trunk (Phasic Policy Gradient)."""

    def __init__(self, obs_dim, spec, hidden_sizes, activation="tanh", log_std_init=0.0, ortho_init=True, encoder_fn=None):
        super().__init__(obs_dim, spec, hidden_sizes, activation, log_std_init, ortho_init, encoder_fn)
        trunk = list(self.pi[1])[:-1]
        self.pi_trunk = nn.Sequential(self.pi[0], *trunk)
        self.pi_head = self.pi[1][-1]
        self.aux_head = nn.Linear(self.pi_head.in_features, 1)
        nn.init.orthogonal_(self.aux_head.weight, 1.0)
        nn.init.zeros_(self.aux_head.bias)

    def dist(self, obs):
        out = self.pi_head(self.pi_trunk(obs))
        if self.discrete:
            return Categorical(logits=out)
        return Independent(Normal(out, self.log_std.exp().expand_as(out)), 1)

    def dist_and_aux(self, obs):
        h = self.pi_trunk(obs)
        out = self.pi_head(h)
        dist = Categorical(logits=out) if self.discrete else Independent(Normal(out, self.log_std.exp().expand_as(out)), 1)
        return dist, self.aux_head(h).squeeze(-1)


# ---------------------------------------------------------------------------------------- continuous


class DeterministicActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, low, high, encoder=None):
        super().__init__()
        self.encoder, in_dim = _encoder_or_identity(encoder, obs_dim)
        self.net = mlp(in_dim, act_dim, hidden_sizes, activation, out_activation="tanh")
        self.register_buffer("scale", torch.as_tensor((high - low) / 2.0, dtype=torch.float32))
        self.register_buffer("bias", torch.as_tensor((high + low) / 2.0, dtype=torch.float32))

    def forward(self, obs):
        return self.net(self.encoder(obs)) * self.scale + self.bias


class BatchRenorm1d(nn.Module):
    """Batch Renormalization (Ioffe, 2017) as used by CrossQ. Behaves like BatchNorm but corrects the batch
    statistics towards the running ones with clipped factors r and d, which keeps joint batches of
    (s, a) and (s', a') from drifting."""

    def __init__(self, num_features, momentum=0.01, eps=1e-5, r_max=3.0, d_max=5.0, warmup_steps=100_000):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("steps", torch.zeros((), dtype=torch.long))
        self.momentum, self.eps, self.r_max, self.d_max, self.warmup_steps = momentum, eps, r_max, d_max, warmup_steps

    def forward(self, x):
        if self.training:
            mean, var = x.mean(0), x.var(0, unbiased=False)
            std = (var + self.eps).sqrt()
            running_std = (self.running_var + self.eps).sqrt()
            warm = (self.steps >= self.warmup_steps).float()
            r = ((std / running_std).detach().clamp(1 / self.r_max, self.r_max) - 1) * warm + 1
            d = ((mean - self.running_mean).detach() / running_std).clamp(-self.d_max, self.d_max) * warm
            y = (x - mean) / std * r + d
            with torch.no_grad():
                self.running_mean.lerp_(mean, self.momentum)
                self.running_var.lerp_(var * x.shape[0] / max(x.shape[0] - 1, 1), self.momentum)
                self.steps += 1
        else:
            y = (x - self.running_mean) / (self.running_var + self.eps).sqrt()
        return y * self.weight + self.bias


class QCritic(nn.Module):
    """Q(s, a) critic. Options: dropout + LayerNorm (DroQ), batch renorm (CrossQ), quantile outputs (TQC)."""

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation="relu", encoder=None, dropout=0.0,
                 layer_norm=False, batch_renorm=False, n_quantiles=1):
        super().__init__()
        self.encoder, in_dim = _encoder_or_identity(encoder, obs_dim)
        self.n_quantiles = n_quantiles
        layers, last = [], in_dim + act_dim
        if batch_renorm:
            layers.append(BatchRenorm1d(last))
        for h in hidden_sizes:
            layers.append(nn.Linear(last, int(h)))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            if layer_norm:
                layers.append(nn.LayerNorm(int(h)))
            if batch_renorm:
                layers.append(BatchRenorm1d(int(h)))
            layers.append(get_activation(activation)())
            last = int(h)
        layers.append(nn.Linear(last, n_quantiles))
        self.net = nn.Sequential(*layers)

    def forward(self, obs, act):
        out = self.net(torch.cat([self.encoder(obs), act], -1))
        return out.squeeze(-1) if self.n_quantiles == 1 else out


class SquashedGaussianActor(nn.Module):
    """tanh-squashed Gaussian policy used by SAC."""

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, low, high, encoder=None):
        super().__init__()
        self.encoder, in_dim = _encoder_or_identity(encoder, obs_dim)
        self.body, last = mlp_body(in_dim, hidden_sizes, activation)
        self.mu = nn.Linear(last, act_dim)
        self.log_std = nn.Linear(last, act_dim)
        self.register_buffer("scale", torch.as_tensor((high - low) / 2.0, dtype=torch.float32))
        self.register_buffer("bias", torch.as_tensor((high + low) / 2.0, dtype=torch.float32))

    def forward(self, obs, deterministic: bool = False):
        h = self.body(self.encoder(obs))
        mu = self.mu(h)
        std = self.log_std(h).clamp(-20, 2).exp()
        u = mu if deterministic else mu + std * torch.randn_like(mu)
        logp = Normal(mu, std).log_prob(u).sum(-1)
        logp = logp - (2 * (math.log(2) - u - F.softplus(-2 * u))).sum(-1) - torch.log(self.scale).sum()
        return torch.tanh(u) * self.scale + self.bias, logp


class CategoricalActor(nn.Module):
    """Softmax policy over discrete actions (discrete SAC, AWAC / IQL on discrete envs)."""

    def __init__(self, obs_dim, n_actions, hidden_sizes, activation="relu", encoder=None):
        super().__init__()
        self.encoder, in_dim = _encoder_or_identity(encoder, obs_dim)
        self.net = mlp(in_dim, n_actions, hidden_sizes, activation)

    def forward(self, obs):
        return self.net(self.encoder(obs))

    def dist(self, obs):
        return Categorical(logits=self.forward(obs))


class VNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation="relu", encoder=None):
        super().__init__()
        self.encoder, in_dim = _encoder_or_identity(encoder, obs_dim)
        self.net = mlp(in_dim, 1, hidden_sizes, activation)

    def forward(self, obs):
        return self.net(self.encoder(obs)).squeeze(-1)


class DiscreteQCritic(nn.Module):
    """Q(s, .) for every discrete action (discrete SAC / CQL / IQL)."""

    def __init__(self, obs_dim, n_actions, hidden_sizes, activation="relu", encoder=None):
        super().__init__()
        self.encoder, in_dim = _encoder_or_identity(encoder, obs_dim)
        self.net = mlp(in_dim, n_actions, hidden_sizes, activation)

    def forward(self, obs):
        return self.net(self.encoder(obs))

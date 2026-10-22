from __future__ import annotations

import math
from typing import Sequence, Tuple

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


def mlp_body(in_dim: int, hidden_sizes: Sequence[int], activation: str = "relu") -> Tuple[nn.Sequential, int]:
    layers, last = [], in_dim
    for h in hidden_sizes:
        layers += [nn.Linear(last, int(h)), get_activation(activation)()]
        last = int(h)
    return nn.Sequential(*layers), last


def mlp(in_dim: int, out_dim: int, hidden_sizes: Sequence[int] = (64, 64), activation: str = "relu",
        out_activation: str | None = None) -> nn.Sequential:
    body, last = mlp_body(in_dim, hidden_sizes, activation)
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


class QNetwork(nn.Module):
    """Q-network covering DQN variants: optional dueling heads, noisy heads and distributional atoms.

    Returns (B, A) when n_atoms == 1, otherwise (B, A, n_atoms) raw outputs (logits or quantiles).
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes: Sequence[int], activation: str = "relu",
                 dueling: bool = False, noisy: bool = False, n_atoms: int = 1, sigma0: float = 0.5):
        super().__init__()
        self.n_actions, self.n_atoms, self.dueling = n_actions, n_atoms, dueling
        self.body, last = mlp_body(obs_dim, hidden_sizes, activation)
        make = (lambda i, o: NoisyLinear(i, o, sigma0)) if noisy else nn.Linear
        self.adv_head = make(last, n_actions * n_atoms)
        self.val_head = make(last, n_atoms) if dueling else None

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.body(obs)
        a = self.adv_head(h).view(-1, self.n_actions, self.n_atoms)
        if self.dueling:
            v = self.val_head(h).view(-1, 1, self.n_atoms)
            a = v + a - a.mean(1, keepdim=True)
        return a.squeeze(-1) if self.n_atoms == 1 else a

    def reset_noise(self) -> None:
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()


class ActorCritic(nn.Module):
    """Separate policy / value MLPs; categorical or diagonal Gaussian policy."""

    def __init__(self, obs_dim: int, spec, hidden_sizes: Sequence[int], activation: str = "tanh",
                 log_std_init: float = 0.0, ortho_init: bool = True):
        super().__init__()
        self.discrete = spec.discrete
        out = spec.n_actions if self.discrete else spec.act_dim
        self.pi = mlp(obs_dim, out, hidden_sizes, activation)
        self.v = mlp(obs_dim, 1, hidden_sizes, activation)
        if not self.discrete:
            self.log_std = nn.Parameter(torch.full((out,), float(log_std_init)))
        if ortho_init:
            orthogonal_init(self.pi, last_gain=0.01)
            orthogonal_init(self.v, last_gain=1.0)

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


class DeterministicActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, low, high):
        super().__init__()
        self.net = mlp(obs_dim, act_dim, hidden_sizes, activation, out_activation="tanh")
        self.register_buffer("scale", torch.as_tensor((high - low) / 2.0, dtype=torch.float32))
        self.register_buffer("bias", torch.as_tensor((high + low) / 2.0, dtype=torch.float32))

    def forward(self, obs):
        return self.net(obs) * self.scale + self.bias


class QCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation="relu"):
        super().__init__()
        self.net = mlp(obs_dim + act_dim, 1, hidden_sizes, activation)

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], -1)).squeeze(-1)


class SquashedGaussianActor(nn.Module):
    """tanh-squashed Gaussian policy used by SAC."""

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, low, high):
        super().__init__()
        self.body, last = mlp_body(obs_dim, hidden_sizes, activation)
        self.mu = nn.Linear(last, act_dim)
        self.log_std = nn.Linear(last, act_dim)
        self.register_buffer("scale", torch.as_tensor((high - low) / 2.0, dtype=torch.float32))
        self.register_buffer("bias", torch.as_tensor((high + low) / 2.0, dtype=torch.float32))

    def forward(self, obs, deterministic: bool = False):
        h = self.body(obs)
        mu = self.mu(h)
        std = self.log_std(h).clamp(-20, 2).exp()
        u = mu if deterministic else mu + std * torch.randn_like(mu)
        logp = Normal(mu, std).log_prob(u).sum(-1)
        logp = logp - (2 * (math.log(2) - u - F.softplus(-2 * u))).sum(-1) - torch.log(self.scale).sum()
        return torch.tanh(u) * self.scale + self.bias, logp

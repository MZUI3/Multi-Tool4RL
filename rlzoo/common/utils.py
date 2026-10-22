from __future__ import annotations

import random
from typing import Dict

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def linear_schedule(start: float, end: float, duration: float, t: float) -> float:
    if duration <= 0:
        return end
    frac = min(1.0, t / duration)
    return start + frac * (end - start)


@torch.no_grad()
def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.mul_(1.0 - tau).add_(sp, alpha=tau)


def hard_update(target: nn.Module, source: nn.Module) -> None:
    target.load_state_dict(source.state_dict())


def update_target(target: nn.Module, source: nn.Module, tau: float, n_updates: int, interval: int) -> None:
    """Polyak update when tau < 1, otherwise a hard copy every `interval` updates."""
    if tau < 1.0:
        soft_update(target, source, tau)
    elif n_updates % max(1, int(interval)) == 0:
        hard_update(target, source)


class RunningMeanStd:
    """Running mean/variance (parallel Welford). Used for value normalisation (MAPPO / MAT ValueNorm)."""

    def __init__(self, eps: float = 1e-4):
        self.mean, self.var, self.count = 0.0, 1.0, eps

    def update(self, x) -> None:
        x = np.asarray(x, dtype=np.float64)
        b_mean, b_var, b_n = float(x.mean()), float(x.var()), x.size
        delta, total = b_mean - self.mean, self.count + b_n
        self.mean += delta * b_n / total
        self.var = (self.var * self.count + b_var * b_n + delta ** 2 * self.count * b_n / total) / total
        self.count = total

    @property
    def std(self) -> float:
        return float(np.sqrt(self.var + 1e-8))

    def normalize(self, x):
        return (x - self.mean) / self.std

    def denormalize(self, x):
        return x * self.std + self.mean


def to_torch(batch: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        v = np.asarray(v)
        dtype = torch.long if np.issubdtype(v.dtype, np.integer) else torch.float32
        out[k] = torch.as_tensor(v, dtype=dtype, device=device)
    return out


def masked_logits(logits: torch.Tensor, avail: torch.Tensor) -> torch.Tensor:
    """Push logits/Q-values of unavailable actions to a very low value."""
    return logits.masked_fill(avail <= 0, -1e10)


def sample_available(avail: torch.Tensor) -> torch.Tensor:
    """Uniformly sample one available action per row. avail: (..., A) -> (...)."""
    flat = avail.reshape(-1, avail.shape[-1]).float()
    return torch.multinomial(flat, 1).view(avail.shape[:-1])

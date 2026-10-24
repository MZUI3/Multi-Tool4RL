from __future__ import annotations

import time
from collections import deque
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from rlzoo.common.hwaccel import AccelConfig, Accelerator, strip_module_prefix, unwrap_model
from rlzoo.common.utils import to_torch


def _local_accelerator(device: torch.device) -> Accelerator:
    return Accelerator(AccelConfig(device=str(device)))


class Algorithm:
    """Common base for every algorithm.

    Subclasses declare `default_hparams`; defaults are merged along the class hierarchy so a subclass
    only lists what it adds or changes. User hparams with unknown keys are rejected.

    Hardware acceleration (shared with the CV hub, see `rlzoo/common/hwaccel.py`):
    - `self.accel` is the Accelerator of the run (device, rank, DP/DDP helpers).
    - Under DDP every rank runs its own environments; `clip_and_step` averages gradients across ranks,
      `global_step` counts environment steps of *all* ranks, and `self.accel.agree(cond)` keeps the
      ranks in lockstep when a decision depends on per-rank state (buffer fill, finished episodes...).
    - `self.shared_rng` is a torch Generator seeded identically on every rank (for per-update random
      orders that must match across ranks, e.g. HAPPO's agent order).
    """

    name: str = "base"
    kind: str = "rl"  # "rl" or "marl"
    default_hparams: Dict[str, Any] = {}
    supports_ddp: bool = True

    def __init__(self, spec, cfg: Dict[str, Any], device: torch.device, logger=None, accel: Optional[Accelerator] = None):
        self.spec = spec
        self.cfg = cfg
        self.device = device
        self.logger = logger
        self.accel = accel if accel is not None else _local_accelerator(device)
        self.world_size = self.accel.world_size
        self.step_scale = self.world_size  # env steps per vector-env step across all ranks
        self.shared_rng = torch.Generator().manual_seed(int(cfg.get("seed", 0)))
        if self.accel.is_distributed and not self.supports_ddp:
            raise ValueError(f"'{self.name}' does not support --parallel ddp (it is asynchronous by itself); use none or dp")
        self.hp = self.resolve_hparams(cfg.get("hparams") or {})
        train_cfg = cfg.get("train") or {}
        self.log_interval = int(train_cfg.get("log_interval", 1000))
        self.eval_interval = int(train_cfg.get("eval_interval", 0))
        self.global_step = 0
        self.train_metrics: Dict[str, float] = {}
        self.evaluator: Optional[Callable[["Algorithm"], Dict[str, float]]] = None
        self.make_venv: Optional[Callable[[int, int], Any]] = None
        self._episodes = deque(maxlen=int(train_cfg.get("stats_window", 20)))
        self._next_log = self.log_interval
        self._next_eval = self.eval_interval
        self._t_last, self._step_last = time.time(), 0

    # ------------------------------------------------------------------ hparams
    @classmethod
    def defaults(cls) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        for klass in reversed(cls.__mro__):
            merged.update(klass.__dict__.get("default_hparams", {}))
        return merged

    @classmethod
    def resolve_hparams(cls, user: Dict[str, Any]) -> Dict[str, Any]:
        hp = cls.defaults()
        unknown = sorted(set(user) - set(hp))
        if unknown:
            raise ValueError(f"Unknown hparams for '{cls.name}': {unknown}. Valid keys: {sorted(hp)}")
        hp.update(user)
        return hp

    # ------------------------------------------------------------------ helpers
    def t(self, x, dtype=torch.float32) -> torch.Tensor:
        return torch.as_tensor(np.asarray(x), dtype=dtype, device=self.device)

    def to_torch(self, batch):
        return to_torch(batch, self.device)

    def clip_and_step(self, optimizer, loss, params, max_norm=None) -> float:
        """backward -> (DDP gradient all-reduce) -> clip -> step. Returns the gradient norm."""
        params = list(params)
        optimizer.zero_grad()
        loss.backward()
        self.accel.sync_gradients(params)
        norm = 0.0
        if max_norm:
            norm = float(nn.utils.clip_grad_norm_(params, max_norm))
        optimizer.step()
        return norm

    def agree(self, flag: bool) -> bool:
        """All-rank AND (no-op outside DDP). Use for per-rank conditions that gate collective updates."""
        return self.accel.agree(flag)

    def reduce_mean(self, value):
        """Average a tensor / float over ranks (identity outside DDP)."""
        if isinstance(value, torch.Tensor):
            return self.accel.reduce(value, "mean")
        return self.accel.reduce_scalar(float(value), "mean")

    def randperm(self, n: int) -> torch.Tensor:
        """Permutation that is identical on every rank."""
        return torch.randperm(n, generator=self.shared_rng)

    # ------------------------------------------------------------------ logging
    def record_episodes(self, episodes) -> None:
        self._episodes.extend(episodes)

    def maybe_log(self, force: bool = False) -> None:
        if self.evaluator is not None and self.eval_interval > 0 and self.global_step >= self._next_eval:
            while self._next_eval <= self.global_step:
                self._next_eval += self.eval_interval
            result = self.evaluator(self)
            if self.logger:
                self.logger.log(self.global_step, {f"eval/{k}": v for k, v in result.items()})
        if force or self.global_step >= self._next_log:
            while self._next_log <= self.global_step:
                self._next_log += self.log_interval
            metrics: Dict[str, float] = {}
            if self._episodes:
                keys = set().union(*(ep.keys() for ep in self._episodes))
                for k in sorted(keys):
                    vals = [ep[k] for ep in self._episodes if k in ep]
                    metrics[f"rollout/{k}"] = float(np.mean(vals))
            metrics.update({f"train/{k}": float(v) for k, v in self.train_metrics.items()})
            now = time.time()
            if self.global_step > self._step_last:
                metrics["time/fps"] = (self.global_step - self._step_last) / max(now - self._t_last, 1e-8)
            self._t_last, self._step_last = now, self.global_step
            if self.logger:
                self.logger.log(self.global_step, metrics)

    # ------------------------------------------------------------------ API
    def networks(self) -> Dict[str, nn.Module]:
        raise NotImplementedError

    def act(self, *args, deterministic: bool = False):
        raise NotImplementedError

    def explain(self, obs) -> Dict[str, Any]:
        """Optional per-step diagnostics for the visualiser (Q-values, action probabilities, value...).

        `obs` is a single flat observation (or (obs, state, avail) for MARL). Return {} when unavailable.
        """
        return {}

    def train(self, venv, total_steps: int) -> None:
        raise NotImplementedError

    def save(self, path) -> None:
        torch.save({k: strip_module_prefix(unwrap_model(m).state_dict()) for k, m in self.networks().items()}, path)

    def load(self, path) -> None:
        state = torch.load(path, map_location=self.device)
        for k, m in self.networks().items():
            unwrap_model(m).load_state_dict(strip_module_prefix(state[k]))

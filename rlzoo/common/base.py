from __future__ import annotations

import time
from collections import deque
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from rlzoo.common.utils import to_torch


class Algorithm:
    """Common base for every algorithm.

    Subclasses declare `default_hparams`; defaults are merged along the class hierarchy so a subclass
    only lists what it adds or changes. User hparams with unknown keys are rejected.
    """

    name: str = "base"
    kind: str = "rl"  # "rl" or "marl"
    default_hparams: Dict[str, Any] = {}

    def __init__(self, spec, cfg: Dict[str, Any], device: torch.device, logger=None):
        self.spec = spec
        self.cfg = cfg
        self.device = device
        self.logger = logger
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
        optimizer.zero_grad()
        loss.backward()
        norm = 0.0
        if max_norm:
            norm = float(nn.utils.clip_grad_norm_(params, max_norm))
        optimizer.step()
        return norm

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

    def train(self, venv, total_steps: int) -> None:
        raise NotImplementedError

    def save(self, path) -> None:
        torch.save({k: m.state_dict() for k, m in self.networks().items()}, path)

    def load(self, path) -> None:
        state = torch.load(path, map_location=self.device)
        for k, m in self.networks().items():
            m.load_state_dict(state[k])

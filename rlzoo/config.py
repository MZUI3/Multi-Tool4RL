"""YAML config loading, dotted-key overrides and defaults."""
from __future__ import annotations

import copy
import re
from typing import Any, Dict, Iterable

import yaml

# PyYAML (YAML 1.1) reads "3e-4" or "1e5" as strings; coerce them to numbers.
_NUM_RE = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")

DEFAULT_TRAIN = dict(
    total_steps=100_000,  # environment steps (vector env steps x num_envs x DDP ranks)
    log_interval=2_000,
    eval_interval=10_000,  # 0 disables periodic evaluation
    eval_episodes=5,
    save_dir="runs",
    tensorboard=False,
    save_model=True,
)

# Hardware acceleration (same options as the CV hub, see rlzoo/common/hwaccel.py).
# `device` at the top level of a config is an alias for accel.device.
DEFAULT_ACCEL = dict(
    parallel="none",  # none | dp | ddp
    device="auto",
    gpu_ids=None,
    sync_bn=False,
    backend=None,
    master_addr="127.0.0.1",
    master_port=29500,
)


def _coerce(value):
    if isinstance(value, dict):
        return {k: _coerce(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_coerce(v) for v in value]
    if isinstance(value, str) and _NUM_RE.match(value.strip()):
        num = float(value)
        if num.is_integer() and "." not in value and "-" not in value.lower().split("e")[-1]:
            return int(num)
        return num
    return value


def load_config(path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return _coerce(yaml.safe_load(f) or {})


def parse_value(text: str):
    return _coerce(yaml.safe_load(text))


def apply_overrides(cfg: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    """Apply ["a.b.c=value", ...] overrides; values are parsed as YAML (lists, bools, numbers, null)."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.path=value, got '{item}'")
        key, raw = item.split("=", 1)
        parts = key.strip().split(".")
        node = cfg
        for p in parts[:-1]:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                raise ValueError(f"Cannot set '{key}': '{p}' is not a mapping")
        node[parts[-1]] = parse_value(raw)
    return cfg


def finalize(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    if "algo" not in cfg:
        raise ValueError("config needs an 'algo' key")
    cfg["algo"] = str(cfg["algo"]).lower()
    cfg.setdefault("seed", 0)
    accel = {**DEFAULT_ACCEL, **(cfg.get("accel") or {})}
    if "device" in cfg and not (cfg.get("accel") or {}).get("device"):
        accel["device"] = cfg["device"]
    cfg["accel"] = accel
    cfg["device"] = accel["device"]
    env = cfg.setdefault("env", {})
    if "id" not in env:
        raise ValueError("config needs env.id")
    env["kwargs"] = env.get("kwargs") or {}
    env.setdefault("num_envs", 1)
    cfg["train"] = {**DEFAULT_TRAIN, **(cfg.get("train") or {})}
    cfg["hparams"] = cfg.get("hparams") or {}
    return cfg

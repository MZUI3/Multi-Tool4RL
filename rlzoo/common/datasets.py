"""Offline datasets (.npz) for offline RL / imitation learning.

A dataset holds flat transitions: obs, actions, rewards, next_obs, terminals (true episode end), timeouts
(time-limit truncation) and episode_starts. `collect.py` records one from a random policy or a trained run;
`dataset: auto` in a config records a random-policy dataset on the fly when the file is missing.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from rlzoo.envs.gym_env import VecEnv, make_gym_env

DEFAULT_DATASET_DIR = "datasets"


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(name))


def default_dataset_path(env_id: str, source: str = "random") -> Path:
    return Path(DEFAULT_DATASET_DIR) / f"{_safe(env_id)}_{_safe(source)}.npz"


def collect_dataset(env_cfg: Dict[str, Any], steps: int, out_path, policy=None, seed: int = 0, noise: float = 0.0,
                    epsilon: float = 0.0, verbose: bool = True) -> Path:
    """Roll out `policy` (None = uniform random) for `steps` environment steps and save an .npz dataset.

    `noise` adds Gaussian noise (scaled by the action range) to continuous actions and `epsilon` picks a
    random discrete action with that probability, which turns an expert into a "medium" dataset.
    """
    venv = VecEnv([lambda: make_gym_env(env_cfg)], seed=seed)
    spec = venv.spec
    obs = venv.reset()
    if hasattr(policy, "reset_hidden"):
        policy.reset_hidden(1)
    buf = {k: [] for k in ("obs", "actions", "rewards", "next_obs", "terminals", "timeouts", "episode_starts")}
    start, returns, ret = True, [], 0.0
    for _ in range(int(steps)):
        if policy is None:
            a = venv.sample_actions()
        else:
            a = policy.act(obs, deterministic=True)
            if spec.discrete and epsilon > 0 and np.random.rand() < epsilon:
                a = venv.sample_actions()
            elif not spec.discrete and noise > 0:
                a = a + np.random.normal(0, noise, a.shape) * (spec.act_high - spec.act_low) / 2
                a = np.clip(a, spec.act_low, spec.act_high)
        next_obs, r, term, trunc, info = venv.step(a)
        done = bool(term[0] or trunc[0])
        real_next = info["final_obs"][0] if done else next_obs[0]
        buf["obs"].append(obs[0])
        buf["actions"].append(a[0])
        buf["rewards"].append(np.float32(r[0]))
        buf["next_obs"].append(real_next)
        buf["terminals"].append(np.float32(term[0]))
        buf["timeouts"].append(np.float32(trunc[0] and not term[0]))
        buf["episode_starts"].append(np.float32(start))
        if hasattr(policy, "observe"):
            policy.observe(float(r[0]))
        ret += float(r[0])
        start = done
        if done:
            returns.append(ret)
            ret = 0.0
            if hasattr(policy, "reset_hidden"):
                policy.reset_hidden(1)
        obs = next_obs
    venv.close()
    data = {k: np.stack(v) for k, v in buf.items()}
    data["actions"] = data["actions"].astype(np.int64 if spec.discrete else np.float32)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **data)
    if verbose:
        mean_ret = float(np.mean(returns)) if returns else float("nan")
        print(f"[dataset] {steps} steps, {len(returns)} episodes, mean return {mean_ret:.2f} -> {out_path}")
    return out_path


def load_dataset(path) -> Dict[str, np.ndarray]:
    with np.load(path) as f:
        data = {k: f[k] for k in f.files}
    data["dones"] = np.maximum(data["terminals"], data["timeouts"])
    return data


def resolve_dataset(cfg: Dict[str, Any], verbose: bool = True) -> Dict[str, np.ndarray]:
    """Load the dataset named by cfg['dataset'] (path / 'auto' / run directory), collecting it if needed."""
    spec = cfg.get("dataset", "auto") or "auto"
    env_cfg = cfg["env"]
    steps = int(cfg.get("dataset_steps", 20_000))
    if isinstance(spec, str) and spec.endswith(".npz"):
        path = Path(spec)
        if not path.exists():
            raise FileNotFoundError(f"dataset {path} not found; record one with `python collect.py --env {env_cfg['id']} --out {path}`")
    elif spec == "auto":
        path = default_dataset_path(env_cfg["id"], "random")
        if not path.exists():
            if verbose:
                print(f"[dataset] {path} missing; recording {steps} random-policy steps on {env_cfg['id']}")
            collect_dataset(env_cfg, steps, path, policy=None, seed=int(cfg.get("seed", 0)), verbose=verbose)
    elif os.path.isdir(spec):  # a trained run directory -> expert dataset
        path = default_dataset_path(env_cfg["id"], Path(spec).name)
        if not path.exists():
            from rlzoo.runner import load_run

            algo, _ = load_run(spec)
            if verbose:
                print(f"[dataset] recording {steps} steps from policy {spec}")
            collect_dataset(env_cfg, steps, path, policy=algo, seed=int(cfg.get("seed", 0)), verbose=verbose)
    else:
        raise ValueError(f"dataset must be an .npz path, a run directory or 'auto', got {spec!r}")
    return load_dataset(path)


def episode_slices(data: Dict[str, np.ndarray]):
    """[(start, end)] index ranges of the episodes in a flat dataset."""
    starts = np.flatnonzero(data["episode_starts"]).tolist()
    ends = starts[1:] + [len(data["rewards"])]
    return list(zip(starts, ends))

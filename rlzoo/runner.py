"""Builds environments + algorithm from a config and runs training / evaluation."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from rlzoo.common.hwaccel import AccelConfig, Accelerator
from rlzoo.common.logger import Logger
from rlzoo.common.utils import get_device, set_seed
from rlzoo.config import finalize
from rlzoo.envs.gym_env import VecEnv, evaluate_rl, make_gym_env
from rlzoo.envs.marl import MAVecEnv, evaluate_marl, make_marl_env
from rlzoo.registry import get_algo_class, get_info

RANK_SEED_STRIDE = 7919  # environments of DDP rank r are seeded with seed + r * stride


def env_factory(kind: str, env_cfg: Dict[str, Any]):
    if kind == "rl":
        return lambda: make_gym_env(env_cfg)
    return lambda: make_marl_env(env_cfg)


def make_venv(kind: str, env_cfg: Dict[str, Any], num_envs: int, seed: int):
    fn = env_factory(kind, env_cfg)
    cls = VecEnv if kind == "rl" else MAVecEnv
    return cls([fn] * int(num_envs), seed=seed)


def evaluate(kind: str, algo, env_cfg: Dict[str, Any], episodes: int, seed: int) -> Dict[str, float]:
    fn = env_factory(kind, env_cfg)
    if kind == "rl":
        return evaluate_rl(algo, fn, episodes, seed)
    return evaluate_marl(algo, fn, episodes, seed)


def make_accelerator(cfg: Dict[str, Any], verbose: bool = True) -> Accelerator:
    cfg = finalize(cfg)
    return Accelerator(AccelConfig.from_dict(cfg["accel"]), log=print if verbose else None)


def build(cfg: Dict[str, Any], logger: Optional[Logger] = None, accel: Optional[Accelerator] = None, verbose: bool = True):
    """Create (algo, venv, cfg) from a config. cfg['hparams'] is expanded to the full resolved set.

    Networks are built with the same seed on every DDP rank (and broadcast from rank 0 as a safety net),
    then each rank re-seeds with its own offset so exploration / minibatch orders differ across ranks.
    """
    cfg = finalize(cfg)
    accel = accel or make_accelerator(cfg, verbose)
    info = get_info(cfg["algo"])
    cls = get_algo_class(cfg["algo"])
    cfg["hparams"] = cls.resolve_hparams(cfg["hparams"])
    cfg["device"] = cfg["accel"]["device"] = str(accel.device)
    seed = int(cfg["seed"])
    set_seed(seed)
    venv = make_venv(info.kind, cfg["env"], cfg["env"]["num_envs"], seed + RANK_SEED_STRIDE * accel.rank)
    algo = cls(venv.spec, cfg, accel.device, logger, accel)
    accel.prepare_tree(algo)
    accel.seed(seed)  # rank-specific randomness from here on
    algo.make_venv = lambda n, s: make_venv(info.kind, cfg["env"], n, s)
    if accel.is_main:  # evaluation only on the main process
        episodes, eval_seed = int(cfg["train"]["eval_episodes"]), seed + 100_000
        algo.evaluator = lambda a: evaluate(info.kind, a, cfg["env"], episodes, eval_seed)
    return algo, venv, cfg


def _run_name(cfg) -> str:
    env_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(cfg["env"]["id"]))
    return f"{cfg['algo']}_{env_id}_s{cfg['seed']}_{time.strftime('%Y%m%d-%H%M%S')}"


def run(cfg: Dict[str, Any], run_dir: Optional[str] = None, verbose: bool = True,
        accel: Optional[Accelerator] = None) -> Tuple[Path, Dict[str, float]]:
    cfg = finalize(cfg)
    accel = accel or make_accelerator(cfg, verbose)
    run_name = accel.broadcast_object(str(run_dir) if run_dir else str(Path(cfg["train"]["save_dir"]) / _run_name(cfg)))
    run_path = Path(run_name)
    logger = Logger(run_path, tensorboard=bool(cfg["train"]["tensorboard"]), verbose=verbose) if accel.is_main else None
    algo, venv, cfg = build(cfg, logger, accel, verbose)
    if accel.is_main:
        with open(run_path / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    accel.print(f"[rlzoo] {cfg['algo']} on {cfg['env']['id']} ({get_info(cfg['algo']).kind}) -> {run_path} "
                f"[{accel.summary()}]") if verbose else None
    t0 = time.time()
    try:
        algo.train(venv, int(cfg["train"]["total_steps"]))
    except KeyboardInterrupt:
        accel.print("[rlzoo] interrupted; saving current model")
    finally:
        venv.close()
    algo.maybe_log(force=True)
    if accel.is_distributed:  # every rank must hold identical weights after training
        digest = float(sum(p.detach().double().sum().item() for m in algo.networks().values() for p in m.parameters()))
        digests = accel.gather_objects(digest)
        if max(digests) - min(digests) > 1e-3 * max(1.0, abs(digests[0])):
            accel.print(f"[rlzoo] WARNING: parameters differ across DDP ranks ({digests}); gradient sync is incomplete")
        else:
            accel.print(f"[rlzoo] DDP parameter sync check: OK ({accel.world_size} ranks)")
    final: Dict[str, float] = {}
    if accel.is_main:
        final = evaluate(get_info(cfg["algo"]).kind, algo, cfg["env"], int(cfg["train"]["eval_episodes"]),
                         int(cfg["seed"]) + 200_000)
        logger.log(algo.global_step, {f"final/{k}": v for k, v in final.items()})
        if cfg["train"]["save_model"]:
            algo.save(run_path / "model.pt")
        summary = {"algo": cfg["algo"], "env": cfg["env"]["id"], "steps": algo.global_step,
                   "wall_time_sec": round(time.time() - t0, 1), "final_eval": final, "hardware": accel.summary()}
        with open(run_path / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.close()
    final = accel.broadcast_object(final)
    accel.barrier()
    return run_path, final


def load_run(run_dir, device: Optional[str] = None):
    """Rebuild the algorithm of a finished run and load its weights. Returns (algo, cfg)."""
    from rlzoo.config import load_config

    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "config.yaml")
    cfg["accel"] = {**(cfg.get("accel") or {}), "parallel": "none"}
    if device:
        cfg["accel"]["device"] = device
        cfg["device"] = device
    algo, venv, cfg = build(cfg, verbose=False)
    venv.close()
    algo.load(run_dir / "model.pt")
    return algo, cfg

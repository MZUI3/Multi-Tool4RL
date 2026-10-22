"""Builds environments + algorithm from a config and runs training / evaluation."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from rlzoo.common.logger import Logger
from rlzoo.common.utils import get_device, set_seed
from rlzoo.config import finalize
from rlzoo.envs.gym_env import VecEnv, evaluate_rl, make_gym_env
from rlzoo.envs.marl import MAVecEnv, evaluate_marl, make_marl_env
from rlzoo.registry import get_algo_class, get_info


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


def build(cfg: Dict[str, Any], logger: Optional[Logger] = None):
    """Create (algo, venv, cfg) from a config. cfg['hparams'] is expanded to the full resolved set."""
    cfg = finalize(cfg)
    info = get_info(cfg["algo"])
    cls = get_algo_class(cfg["algo"])
    cfg["hparams"] = cls.resolve_hparams(cfg["hparams"])
    set_seed(int(cfg["seed"]))
    device = get_device(cfg["device"])
    venv = make_venv(info.kind, cfg["env"], cfg["env"]["num_envs"], int(cfg["seed"]))
    algo = cls(venv.spec, cfg, device, logger)
    algo.make_venv = lambda n, seed: make_venv(info.kind, cfg["env"], n, seed)
    episodes, eval_seed = int(cfg["train"]["eval_episodes"]), int(cfg["seed"]) + 100_000
    algo.evaluator = lambda a: evaluate(info.kind, a, cfg["env"], episodes, eval_seed)
    return algo, venv, cfg


def _run_name(cfg) -> str:
    env_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(cfg["env"]["id"]))
    return f"{cfg['algo']}_{env_id}_s{cfg['seed']}_{time.strftime('%Y%m%d-%H%M%S')}"


def run(cfg: Dict[str, Any], run_dir: Optional[str] = None, verbose: bool = True) -> Tuple[Path, Dict[str, float]]:
    cfg = finalize(cfg)
    run_path = Path(run_dir) if run_dir else Path(cfg["train"]["save_dir"]) / _run_name(cfg)
    logger = Logger(run_path, tensorboard=bool(cfg["train"]["tensorboard"]), verbose=verbose)
    algo, venv, cfg = build(cfg, logger)
    with open(run_path / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    if verbose:
        print(f"[rlzoo] {cfg['algo']} on {cfg['env']['id']} ({get_info(cfg['algo']).kind}) -> {run_path}")
    t0 = time.time()
    try:
        algo.train(venv, int(cfg["train"]["total_steps"]))
    except KeyboardInterrupt:
        print("[rlzoo] interrupted; saving current model")
    finally:
        venv.close()
    algo.maybe_log(force=True)
    final = evaluate(get_info(cfg["algo"]).kind, algo, cfg["env"], int(cfg["train"]["eval_episodes"]),
                     int(cfg["seed"]) + 200_000)
    logger.log(algo.global_step, {f"final/{k}": v for k, v in final.items()})
    if cfg["train"]["save_model"]:
        algo.save(run_path / "model.pt")
    summary = {"algo": cfg["algo"], "env": cfg["env"]["id"], "steps": algo.global_step,
               "wall_time_sec": round(time.time() - t0, 1), "final_eval": final}
    with open(run_path / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.close()
    return run_path, final

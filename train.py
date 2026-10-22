"""Train any of the 30 algorithms from a YAML config.

Examples
  python train.py --list
  python train.py --config configs/rl/ppo.yaml
  python train.py --algo qmix                                  # uses configs/marl/qmix.yaml
  python train.py --algo sac --set train.total_steps=20000 hparams.actor_lr=0.001
  python train.py --algo vdn --env matrix_game --set env.kwargs.payoff=climbing
"""
from __future__ import annotations

import argparse
from pathlib import Path

from rlzoo.config import apply_overrides, load_config
from rlzoo.registry import get_info, list_algorithms
from rlzoo.runner import run

ROOT = Path(__file__).resolve().parent


def default_config_path(algo: str) -> Path:
    info = get_info(algo)
    return ROOT / "configs" / info.kind / f"{info.name}.yaml"


def print_algorithms() -> None:
    for kind, title in (("rl", "Single-agent RL"), ("marl", "Multi-agent RL")):
        algos = list_algorithms(kind)
        print(f"\n{title} ({len(algos)})")
        print(f"  {'name':<12} {'family':<22} reference")
        for a in algos:
            print(f"  {a.name:<12} {a.family:<22} {a.reference}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=str, help="YAML config file")
    p.add_argument("--algo", type=str, help="algorithm name (loads configs/<kind>/<algo>.yaml if --config omitted)")
    p.add_argument("--env", type=str, help="override env.id")
    p.add_argument("--seed", type=int, help="override seed")
    p.add_argument("--steps", type=int, help="override train.total_steps")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="dotted overrides, e.g. hparams.lr=1e-3")
    p.add_argument("--run-dir", type=str, help="explicit output directory")
    p.add_argument("--list", action="store_true", help="list available algorithms")
    args = p.parse_args()

    if args.list:
        print_algorithms()
        return
    if not args.config and not args.algo:
        p.error("give --config or --algo (or --list)")

    cfg = load_config(args.config or default_config_path(args.algo))
    if args.algo:
        cfg["algo"] = args.algo
    if args.env:
        cfg.setdefault("env", {})["id"] = args.env
        cfg["env"]["kwargs"] = {}  # kwargs of the config's env rarely fit another env; re-add with --set
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.steps is not None:
        cfg.setdefault("train", {})["total_steps"] = args.steps
    cfg = apply_overrides(cfg, args.set)
    run_dir, final = run(cfg, run_dir=args.run_dir)
    print(f"\nFinal evaluation: {final}\nOutputs: {run_dir}")


if __name__ == "__main__":
    main()

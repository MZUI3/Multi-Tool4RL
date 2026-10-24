"""Evaluate a trained run directory (uses its saved config.yaml and model.pt).

  python evaluate.py runs/ppo_CartPole-v1_s0_20261023-120000 --episodes 20
  python evaluate.py runs/sac_Pendulum-v1_s0_... --watch          # open the simulation viewer (see watch.py)
"""
from __future__ import annotations

import argparse
from pathlib import Path

from rlzoo.registry import get_info
from rlzoo.runner import evaluate, load_run


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=str)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--device", type=str, default=None, help="cpu | cuda | cuda:N (default: as trained, or cpu)")
    p.add_argument("--render", "--watch", dest="watch", action="store_true", help="open the interactive simulation viewer")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    algo, cfg = load_run(run_dir, args.device)
    kind = get_info(cfg["algo"]).kind
    result = evaluate(kind, algo, cfg["env"], args.episodes, args.seed)
    print(f"{cfg['algo']} on {cfg['env']['id']} over {args.episodes} episodes:")
    for k, v in result.items():
        print(f"  {k:<12} {v:.3f}")
    if args.watch:
        from rlzoo.viz.player import watch_run

        watch_run(run_dir, window=True, algo=algo, cfg=cfg)


if __name__ == "__main__":
    main()

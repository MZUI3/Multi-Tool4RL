"""Evaluate a trained run directory (uses its saved config.yaml and model.pt).

  python evaluate.py runs/ppo_CartPole-v1_s0_20261023-120000 --episodes 20
  python evaluate.py runs/sac_Pendulum-v1_s0_... --render        # gymnasium human rendering (RL only)
"""
from __future__ import annotations

import argparse
from pathlib import Path

from rlzoo.config import load_config
from rlzoo.registry import get_info
from rlzoo.runner import build, evaluate


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=str)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--render", action="store_true", help="render with render_mode='human' (single-agent only)")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    cfg = load_config(run_dir / "config.yaml")
    kind = get_info(cfg["algo"]).kind
    algo, venv, cfg = build(cfg)
    venv.close()
    algo.load(run_dir / "model.pt")

    env_cfg = dict(cfg["env"])
    if args.render:
        if kind != "rl":
            print("--render is only supported for single-agent gymnasium envs; evaluating without rendering")
        else:
            env_cfg["kwargs"] = {**env_cfg.get("kwargs", {}), "render_mode": "human"}
    result = evaluate(kind, algo, env_cfg, args.episodes, args.seed)
    print(f"{cfg['algo']} on {cfg['env']['id']} over {args.episodes} episodes:")
    for k, v in result.items():
        print(f"  {k:<12} {v:.3f}")


if __name__ == "__main__":
    main()

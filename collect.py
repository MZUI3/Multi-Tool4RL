"""Record an offline dataset (.npz) from a random policy or a trained run.

  python collect.py --env CartPole-v1 --steps 20000                      # random policy -> datasets/CartPole-v1_random.npz
  python collect.py --run runs/ppo_CartPole-v1_s0_... --steps 50000      # expert policy -> datasets/CartPole-v1_<run>.npz
  python collect.py --run runs/sac_Pendulum-v1_... --noise 0.3 --out datasets/pendulum_medium.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

from rlzoo.common.datasets import collect_dataset, default_dataset_path
from rlzoo.config import load_config


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", type=str, help="gymnasium env id (random policy)")
    p.add_argument("--run", type=str, help="trained run directory (its config.yaml gives the env)")
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noise", type=float, default=0.0, help="Gaussian action noise (continuous), fraction of the action range")
    p.add_argument("--epsilon", type=float, default=0.0, help="random-action probability (discrete)")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()
    if not args.env and not args.run:
        p.error("give --env or --run")
    policy, env_cfg, source = None, {"id": args.env, "kwargs": {}}, "random"
    if args.run:
        from rlzoo.runner import load_run

        policy, cfg = load_run(args.run)
        env_cfg, source = cfg["env"], Path(args.run).name
    out = args.out or default_dataset_path(env_cfg["id"], source)
    collect_dataset(env_cfg, args.steps, out, policy, args.seed, args.noise, args.epsilon)


if __name__ == "__main__":
    main()

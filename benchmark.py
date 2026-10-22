"""Run several algorithms back to back and print a summary table.

  python benchmark.py --kind rl                                   # all 15 RL algos, default configs
  python benchmark.py --algos qmix vdn qplex --env matrix_game --steps 10000
  python benchmark.py --kind marl --seeds 0 1 2 --set train.eval_interval=0
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rlzoo.config import apply_overrides, load_config
from rlzoo.registry import get_info, list_algorithms
from rlzoo.runner import run

ROOT = Path(__file__).resolve().parent


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kind", choices=["rl", "marl"], help="run every algorithm of this kind")
    p.add_argument("--algos", nargs="*", default=[], help="explicit algorithm names")
    p.add_argument("--env", type=str, help="override env.id (env.kwargs are cleared)")
    p.add_argument("--steps", type=int, help="override train.total_steps")
    p.add_argument("--seeds", nargs="*", type=int, default=[0])
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    p.add_argument("--save-dir", default="runs/benchmark")
    p.add_argument("--quiet", action="store_true", help="hide per-run training logs")
    args = p.parse_args()

    names = args.algos or [a.name for a in list_algorithms(args.kind)]
    if not names:
        p.error("give --kind or --algos")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    rows = []
    for name in names:
        info = get_info(name)
        for seed in args.seeds:
            cfg = load_config(ROOT / "configs" / info.kind / f"{name}.yaml")
            cfg["seed"] = seed
            if args.env:
                cfg["env"]["id"] = args.env
                cfg["env"]["kwargs"] = {}
            if args.steps:
                cfg.setdefault("train", {})["total_steps"] = args.steps
            cfg = apply_overrides(cfg, args.set)
            t0 = time.time()
            run_dir, final = run(cfg, run_dir=f"{args.save_dir}/{stamp}/{name}_s{seed}", verbose=not args.quiet)
            rows.append({"algo": name, "seed": seed, "env": cfg["env"]["id"], "return": final["return"],
                         "return_std": final["return_std"], "minutes": (time.time() - t0) / 60})
            print(f"-> {name} seed {seed}: eval return {final['return']:.2f} ({rows[-1]['minutes']:.1f} min)",
                  flush=True)

    print(f"\n{'algo':<12} {'seed':>4} {'env':<18} {'eval return':>12} {'min':>6}")
    for r in rows:
        print(f"{r['algo']:<12} {r['seed']:>4} {r['env']:<18} {r['return']:>8.2f}±{r['return_std']:<6.1f} "
              f"{r['minutes']:>5.1f}")
    out = Path(args.save_dir) / stamp / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()

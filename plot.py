"""Plot learning curves from one or more run directories (requires matplotlib).

  python plot.py runs/dqn_* runs/rainbow_* --metric eval/return --out compare.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_curve(run_dir: Path, metric: str):
    steps, values = [], []
    with open(run_dir / "progress.jsonl", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if metric in rec:
                steps.append(rec["step"])
                values.append(rec[metric])
    return steps, values


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="run directories")
    p.add_argument("--metric", default="rollout/return", help="e.g. rollout/return, eval/return, train/loss")
    p.add_argument("--out", default=None, help="save to file instead of showing a window")
    args = p.parse_args()

    import matplotlib
    if args.out:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for run in args.runs:
        run_dir = Path(run)
        steps, values = load_curve(run_dir, args.metric)
        if steps:
            ax.plot(steps, values, label=run_dir.name)
    ax.set_xlabel("environment steps")
    ax.set_ylabel(args.metric)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"saved {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

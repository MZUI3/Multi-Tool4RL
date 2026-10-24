"""Runs every algorithm for a few hundred steps on its default config (no learning check).

  python tests/test_smoke.py            # all algorithms
  python tests/test_smoke.py qmix ppo   # a subset
  pytest tests/                         # same, if pytest is installed
"""
from __future__ import annotations

import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rlzoo.config import apply_overrides, load_config  # noqa: E402
from rlzoo.registry import ALGORITHMS, get_info  # noqa: E402
from rlzoo.runner import run  # noqa: E402

SMOKE_OVERRIDES = [
    "train.total_steps=600",
    "train.log_interval=300",
    "train.eval_interval=300",
    "train.eval_episodes=1",
]


def smoke(name: str, out_dir: str) -> float:
    info = get_info(name)
    cfg = load_config(ROOT / "configs" / info.kind / f"{name}.yaml")
    overrides = list(SMOKE_OVERRIDES)
    hp = cfg.get("hparams", {})
    if "learning_starts" in hp:
        overrides += ["hparams.learning_starts=100"]
    if "batch_size" in hp:
        overrides += ["hparams.batch_size=32"]
    if "dataset" in cfg:  # offline / imitation: record a tiny random dataset on the fly
        overrides += ["dataset=auto", "dataset_steps=1500"]
    cfg = apply_overrides(cfg, overrides)
    _, final = run(cfg, run_dir=str(Path(out_dir) / name), verbose=False)
    return final["return"]


def test_all_algorithms(tmp_path):
    for name in ALGORITHMS:
        smoke(name, str(tmp_path))


if __name__ == "__main__":
    names = sys.argv[1:] or list(ALGORITHMS)
    failed = []
    with tempfile.TemporaryDirectory() as tmp:
        for name in names:
            t0 = time.time()
            try:
                ret = smoke(name, tmp)
                print(f"PASS {name:<12} {time.time() - t0:5.1f}s  eval return {ret:.2f}")
            except Exception:
                failed.append(name)
                print(f"FAIL {name}")
                traceback.print_exc()
    print(f"\n{len(names) - len(failed)}/{len(names)} passed" + (f"; failed: {failed}" if failed else ""))
    sys.exit(1 if failed else 0)

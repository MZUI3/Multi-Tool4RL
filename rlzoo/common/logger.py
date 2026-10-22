from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


class Logger:
    """Writes metrics to progress.jsonl, the console and (optionally) TensorBoard."""

    def __init__(self, run_dir, tensorboard: bool = False, verbose: bool = True):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.file = open(self.run_dir / "progress.jsonl", "a", encoding="utf-8")
        self.verbose = verbose
        self.tb = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb = SummaryWriter(str(self.run_dir))
            except ImportError:
                print("[logger] tensorboard not installed; skipping TensorBoard output")

    def log(self, step: int, metrics: Dict[str, float]) -> None:
        record = {"step": int(step), **{k: float(v) for k, v in metrics.items()}}
        self.file.write(json.dumps(record) + "\n")
        self.file.flush()
        if self.tb is not None:
            for k, v in metrics.items():
                self.tb.add_scalar(k, float(v), step)
        if self.verbose:
            parts = [f"{k}={v:.4g}" for k, v in metrics.items()]
            print(f"[{step:>9d}] " + " | ".join(parts), flush=True)

    def close(self) -> None:
        self.file.close()
        if self.tb is not None:
            self.tb.close()

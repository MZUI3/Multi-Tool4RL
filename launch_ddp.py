"""torchrun stand-in: ``python launch_ddp.py --nproc 2 train.py --algo ppo ...``

Starts one process per GPU (or ``--nproc`` processes on CPU) with RANK / WORLD_SIZE / LOCAL_RANK /
MASTER_ADDR / MASTER_PORT set, which the shared hardware-acceleration module picks up as ``--parallel ddp``.
Use it where ``torchrun`` is missing or broken (Windows wheels without libuv).
"""
import sys

from rlzoo.common.hwaccel import launch_main

if __name__ == "__main__":
    sys.exit(launch_main())

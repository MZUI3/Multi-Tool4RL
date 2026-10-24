"""Watch a trained agent act in its environment (live window) or export episodes to GIF / MP4 / HTML.

  python watch.py runs/ppo_CartPole-v1_s0_...                    # live viewer (space pause, n step, r reset, d det/stoch, +/- speed, q quit)
  python watch.py runs/qmix_spread_s0_... --episodes 5 --fps 8   # built-in MARL envs are drawn with matplotlib
  python watch.py runs/sac_Pendulum-v1_... --save pendulum.gif --no-window
  python watch.py runs/dqn_CartPole-v1_... --save episode.html   # self-contained HTML player with Q-values per step
  python watch.py runs/... --stochastic --seed 3

Rendering: the env's own rgb_array renderer is used when available (classic control needs `pip install pygame`);
otherwise rlzoo draws the state itself (CartPole, MountainCar, Pendulum, Acrobot, BitFlip, PointReach,
PixelGrid, Spread, Predator-Prey, Matrix game).
"""
from __future__ import annotations

import argparse

from rlzoo.viz.player import watch_run


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=str)
    p.add_argument("--episodes", type=int, default=3, help="episodes to record when saving")
    p.add_argument("--fps", type=int, default=None, help="playback speed (default: the env's render_fps)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--stochastic", action="store_true", help="sample actions instead of acting greedily")
    p.add_argument("--save", type=str, default=None, help="export to this file (.gif / .mp4 / .html)")
    p.add_argument("--no-window", action="store_true", help="do not open the live viewer")
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()
    watch_run(args.run_dir, save=args.save, window=not args.no_window, episodes=args.episodes, fps=args.fps,
              seed=args.seed, deterministic=not args.stochastic, max_steps=args.max_steps, device=args.device)


if __name__ == "__main__":
    main()

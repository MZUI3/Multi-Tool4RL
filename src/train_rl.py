import argparse
import os

from src.algos_registry import run_algorithm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--algo", type=str, default="ppo")
    parser.add_argument("--total-timesteps", type=int, default=100000)
    parser.add_argument("--atari", action="store_true")
    parser.add_argument("--save-dir", type=str, default="models")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    res = run_algorithm(args.algo, args.env, atari=args.atari, total_timesteps=args.total_timesteps, save_path=args.save_dir)
    if res.get("status") == "ok":
        print("Saved:", res.get("path"))
    else:
        print("Error:", res.get("msg"))


if __name__ == "__main__":
    main()

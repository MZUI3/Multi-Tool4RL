"""
SMAC (StarCraft Multi-Agent Challenge) launcher skeleton.

SMAC requires StarCraft II and the `smac` package from the SMAC repo.
This file provides instructions and a simple entry point for integrating SMAC-based trainers.
"""
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str, default="3m")
    parser.add_argument("--train-script", type=str, default="path/to/pymarl/train.py")
    parser.add_argument("--sc2-path", type=str, default=None, help="Path to StarCraft II installation")
    args = parser.parse_args()

    print("SMAC launcher stub")
    print("Ensure SMAC and StarCraft II are installed. See README for details.")
    print(f"Map: {args.map}")
    print(f"Train script: {args.train_script}")
    if args.sc2_path:
        print(f"SC2 path: {args.sc2_path}")


if __name__ == "__main__":
    main()

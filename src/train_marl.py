import argparse
import os

import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env

from pettingzoo.mpe import simple_spread_v3
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv


def make_rllib_env(env_name):
    # Currently supports simple_spread from PettingZoo MPE family.
    if env_name == "simple_spread_v3":
        return PettingZooEnv(simple_spread_v3.parallel_env())
    raise ValueError("Unsupported env: " + env_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="simple_spread_v3")
    parser.add_argument("--stop-timesteps", type=int, default=200000)
    parser.add_argument("--checkpoint-dir", type=str, default="marl_checkpoints")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    ray.init(ignore_reinit_error=True)

    register_env("pettingzoo_simple_spread", lambda config: make_rllib_env(args.env))

    # single shared policy for all agents (simple setup)
    tune_config = {
        "env": "pettingzoo_simple_spread",
        "framework": "torch",
        "num_workers": 1,
        "env_config": {},
        "multiagent": {
            "policies": {"shared_policy": (None, None, None, {})},
            "policy_mapping_fn": (lambda agent_id, episode, **kwargs: "shared_policy"),
        },
    }

    stop = {"timesteps_total": args.stop_timesteps}

    analysis = tune.run(PPOTrainer, config=tune_config, stop=stop, local_dir=args.checkpoint_dir)
    print("Training finished. Checkpoints in:", args.checkpoint_dir)


if __name__ == "__main__":
    main()

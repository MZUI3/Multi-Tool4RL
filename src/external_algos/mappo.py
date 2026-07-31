def train(env_id: str, stop_timesteps: int = 200000, checkpoint_dir: str = "mappo_checkpoints", **kwargs):
    """Run a MAPPO-like training using Ray RLlib's PPOTrainer with a multiagent setup.

    This provides a practical MAPPO-style runner for PettingZoo parallel envs. It expects
    the `env_id` to be a PettingZoo parallel env factory name handled in this wrapper.
    """
    import os
    import ray
    from ray import tune
    from ray.rllib.agents.ppo import PPOTrainer
    from ray.tune.registry import register_env

    # Support a small set of pettingzoo envs; user can extend mapping
    def make_rllib_env(env_name):
        # import here to avoid heavy deps if not used
        from pettingzoo import AECEnv
        try:
            # try common MPE
            from pettingzoo.mpe import simple_spread_v3
            if env_name == "simple_spread_v3":
                from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
                return PettingZooEnv(simple_spread_v3.parallel_env())
        except Exception:
            pass
        raise ValueError(f"Unsupported PettingZoo env for MAPPO wrapper: {env_name}")

    os.makedirs(checkpoint_dir, exist_ok=True)
    ray.init(ignore_reinit_error=True)
    register_env("mappo_env", lambda cfg: make_rllib_env(env_id))

    tune_config = {
        "env": "mappo_env",
        "framework": "torch",
        "num_workers": 1,
        "multiagent": {
            "policies": {"shared_policy": (None, None, None, {})},
            "policy_mapping_fn": (lambda agent_id, episode, **kwargs: "shared_policy"),
        },
        # MAPPO-like PPO hyperparams
        "sgd_minibatch_size": 64,
        "train_batch_size": 4000,
        "rollout_fragment_length": 200,
        "lambda": 0.95,
        "clip_param": 0.2,
        "num_sgd_iter": 10,
    }

    stop = {"timesteps_total": stop_timesteps}

    analysis = tune.run(PPOTrainer, config=tune_config, stop=stop, local_dir=checkpoint_dir)
    return {"status": "ok", "msg": f"Completed. Results: {analysis.best_config if hasattr(analysis, 'best_config') else 'n/a'}"}

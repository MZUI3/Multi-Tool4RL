"""
Algorithm registry and adapters.
Supports calling SB3 algos, RLlib-based MARL algos, and external plugin modules.
To add advanced algorithms (MuZero, Agent57, DecisionTransformer, Rainbow, QMIX, MAPPO, etc.),
create a file under `src/external_algos/<name>.py` exposing `train(env_id, **kwargs)`.
"""
from typing import Dict, Any

SB3_ALGOS = {"dqn", "ppo", "sac", "td3"}


def run_sb3(algo: str, env, total_timesteps: int = 100000, save_path: str = "models", **kwargs):
    from stable_baselines3 import PPO, DQN, SAC, TD3
    algo = algo.lower()
    if algo == "ppo":
        model = PPO("CnnPolicy" if kwargs.get("atari") else "MlpPolicy", env, verbose=1)
    elif algo == "dqn":
        model = DQN("CnnPolicy" if kwargs.get("atari") else "MlpPolicy", env, verbose=1)
    elif algo == "sac":
        model = SAC("MlpPolicy", env, verbose=1)
    elif algo in ("ddpg", "td3", "td3_bc"):
        # TD3/DDPG are in sb3-contrib or stable-baselines3 contrib; try importing TD3 then fallback
        try:
            from sb3_contrib import TRPO  # dummy import to hint contrib
        except Exception:
            pass
        from stable_baselines3 import TD3
        model = TD3("MlpPolicy", env, verbose=1)
    else:
        raise ValueError(f"SB3 algo not supported: {algo}")

    model.learn(total_timesteps=total_timesteps)
    import os
    os.makedirs(save_path, exist_ok=True)
    out = os.path.join(save_path, f"{env.spec.id}-{algo}.zip")
    model.save(out)
    return out


def run_external(algo: str, env_id: str, **kwargs):
    """Load an external plugin from src/external_algos/<algo>.py and call train(env_id, **kwargs)."""
    import importlib
    try:
        mod = importlib.import_module(f"src.external_algos.{algo}")
    except ModuleNotFoundError:
        raise
    if hasattr(mod, "train"):
        return mod.train(env_id=env_id, **kwargs)
    raise AttributeError(f"external alg module {algo} has no train(env_id, **kwargs)")


def run_algorithm(algo: str, env_id: str, atari: bool = False, total_timesteps: int = 100000, save_path: str = "models", **kwargs) -> Dict[str, Any]:
    algo_l = algo.lower()
    import gymnasium as gym
    if algo_l in SB3_ALGOS:
        # build a simple env wrapper
        env = gym.make(env_id)
        from stable_baselines3.common.vec_env import DummyVecEnv
        venv = DummyVecEnv([lambda: env])
        out = run_sb3(algo_l, venv, total_timesteps=total_timesteps, save_path=save_path, atari=atari, **kwargs)
        return {"status": "ok", "path": out}

    # MARL / RLlib handled separately by user scripts
    # try external plugin
    try:
        out = run_external(algo_l, env_id, atari=atari, total_timesteps=total_timesteps, save_path=save_path, **kwargs)
        return {"status": "ok", "path": out}
    except ModuleNotFoundError:
        return {"status": "error", "msg": f"Algorithm {algo} not implemented. Add src/external_algos/{algo}.py plugin."}

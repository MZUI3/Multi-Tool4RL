"""Algorithm registry: name -> implementation, with lazy imports."""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class AlgoInfo:
    name: str
    kind: str  # "rl" | "marl"
    family: str
    target: str  # "module:Class"
    reference: str


_ENTRIES = [
    # ---------------------------------------------------------------- single-agent RL (15)
    ("dqn", "rl", "value-based", "rlzoo.rl.dqn:DQN", "Mnih et al., 2015"),
    ("double_dqn", "rl", "value-based", "rlzoo.rl.dqn:DoubleDQN", "van Hasselt et al., 2016"),
    ("dueling_dqn", "rl", "value-based", "rlzoo.rl.dqn:DuelingDQN", "Wang et al., 2016"),
    ("per_dqn", "rl", "value-based", "rlzoo.rl.dqn:PERDQN", "Schaul et al., 2016"),
    ("c51", "rl", "distributional", "rlzoo.rl.dqn:C51", "Bellemare et al., 2017"),
    ("qr_dqn", "rl", "distributional", "rlzoo.rl.dqn:QRDQN", "Dabney et al., 2018"),
    ("rainbow", "rl", "distributional", "rlzoo.rl.dqn:Rainbow", "Hessel et al., 2018"),
    ("reinforce", "rl", "policy-gradient", "rlzoo.rl.actor_critic:REINFORCE", "Williams, 1992"),
    ("a2c", "rl", "actor-critic", "rlzoo.rl.actor_critic:A2C", "Mnih et al., 2016"),
    ("a3c", "rl", "actor-critic", "rlzoo.rl.actor_critic:A3C", "Mnih et al., 2016"),
    ("trpo", "rl", "trust-region", "rlzoo.rl.trpo:TRPO", "Schulman et al., 2015"),
    ("ppo", "rl", "trust-region", "rlzoo.rl.ppo:PPO", "Schulman et al., 2017"),
    ("ddpg", "rl", "continuous-control", "rlzoo.rl.continuous:DDPG", "Lillicrap et al., 2016"),
    ("td3", "rl", "continuous-control", "rlzoo.rl.continuous:TD3", "Fujimoto et al., 2018"),
    ("sac", "rl", "continuous-control", "rlzoo.rl.continuous:SAC", "Haarnoja et al., 2018"),
    # ---------------------------------------------------------------- multi-agent RL (15)
    ("iql", "marl", "independent", "rlzoo.marl.value_based:IQL", "Tan, 1993"),
    ("vdn", "marl", "value-decomposition", "rlzoo.marl.value_based:VDN", "Sunehag et al., 2018"),
    ("qmix", "marl", "value-decomposition", "rlzoo.marl.value_based:QMIX", "Rashid et al., 2018"),
    ("qtran", "marl", "value-decomposition", "rlzoo.marl.value_based:QTRAN", "Son et al., 2019"),
    ("qplex", "marl", "value-decomposition", "rlzoo.marl.value_based:QPLEX", "Wang et al., 2021"),
    ("wqmix", "marl", "value-decomposition", "rlzoo.marl.value_based:WQMIX", "Rashid et al., 2020"),
    ("coma", "marl", "centralized-critic", "rlzoo.marl.coma:COMA", "Foerster et al., 2018"),
    ("maddpg", "marl", "centralized-critic", "rlzoo.marl.maddpg:MADDPG", "Lowe et al., 2017"),
    ("matd3", "marl", "centralized-critic", "rlzoo.marl.maddpg:MATD3", "Ackermann et al., 2019"),
    ("ippo", "marl", "policy-gradient", "rlzoo.marl.ppo_family:IPPO", "de Witt et al., 2020"),
    ("mappo", "marl", "policy-gradient", "rlzoo.marl.ppo_family:MAPPO", "Yu et al., 2022"),
    ("happo", "marl", "policy-gradient", "rlzoo.marl.ppo_family:HAPPO", "Kuba et al., 2022"),
    ("mat", "marl", "transformer", "rlzoo.marl.mat:MAT", "Wen et al., 2022"),
    ("commnet", "marl", "communication", "rlzoo.marl.ppo_family:CommNet", "Sukhbaatar et al., 2016"),
    ("g2anet", "marl", "communication", "rlzoo.marl.ppo_family:G2ANet", "Liu et al., 2020"),
]

ALGORITHMS: Dict[str, AlgoInfo] = {e[0]: AlgoInfo(*e) for e in _ENTRIES}


def get_info(name: str) -> AlgoInfo:
    try:
        return ALGORITHMS[name.lower()]
    except KeyError:
        raise ValueError(f"Unknown algorithm '{name}'. Available: {', '.join(ALGORITHMS)}") from None


def get_algo_class(name: str):
    module, cls = get_info(name).target.split(":")
    return getattr(importlib.import_module(module), cls)


def list_algorithms(kind: str | None = None):
    return [info for info in ALGORITHMS.values() if kind is None or info.kind == kind]

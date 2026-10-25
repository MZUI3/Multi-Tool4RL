# rlzoo

rlzoo is a set of 60 reinforcement learning algorithms in PyTorch (40 single-agent, 20 multi-agent), all trained from YAML configs by one script. It does not depend on another RL library such as Stable-Baselines3, RLlib or PyMARL: replay buffers, vector environments, the built-in environments and the viewer are part of this repository.

## Installation

```bash
pip install -r requirements.txt       # torch>=2.0, numpy, gymnasium>=0.29, pyyaml, matplotlib, pillow
conda env create -f environment.yml   # conda env "rlzoo" with Python 3.11
bash setup.sh                         # conda env "rlzoo" if conda is found, otherwise .venv
```

`setup.sh` also takes `--venv` (force a venv), `--conda` and `--extras` (adds `gymnasium[box2d]`, `pettingzoo`, `mpe2`), and ends with `python train.py --list`. There is no package to install; the scripts run from the repository root and create `runs/` and `datasets/` in the working directory. Optional: `pygame` (Gymnasium's own classic-control renderer), `ffmpeg` (system package, `.mp4` export), `tensorboard`, `pytest`, and the env packages `gymnasium[box2d]`, `gymnasium[mujoco]`, `pettingzoo` + `mpe2`, `smac` (needs StarCraft II).

## Quick start

```bash
git clone https://github.com/MZUI3/Multi-Tool4RL.git && cd Multi-Tool4RL
pip install -r requirements.txt
python train.py --list                                   # algorithms with family and paper reference
python train.py --algo ppo --run-dir runs/ppo_cartpole   # configs/rl/ppo.yaml: PPO on CartPole-v1
python watch.py runs/ppo_cartpole                        # live viewer
```

## Usage

- `train.py` trains from `configs/<rl|marl>/<algo>.yaml` (`--algo`) or any YAML file (`--config`). `--env` replaces `env.id` and clears `env.kwargs`; `--set` takes dotted `KEY=VALUE` pairs parsed as YAML. `--watch` opens the viewer after training; `--watch-save FILE` records episodes to `.gif`, `.mp4` or `.html`. Ctrl+C stops training; the model is still saved and evaluated.
- `evaluate.py` runs the saved policy deterministically and prints mean return, its std, mean length and env metrics. Defaults: `--episodes 10`, `--seed 12345`. `--watch` opens the viewer.
- `watch.py` opens the [viewer](#viewer), or exports episodes with `--save`. Other flags: `--no-window`, `--episodes` (3, for export), `--stochastic`, `--seed`, `--fps`, `--max-steps`, `--device`.
- `plot.py` plots one metric from `progress.jsonl` (default `rollout/return`) for one or more runs. `--out` writes an image instead of opening a window.
- `collect.py` records a single-agent offline dataset from a random policy (`--env ID`, saved as `datasets/<env>_random.npz`) or a trained run (`--run DIR`). `--noise` (continuous) and `--epsilon` (discrete) perturb a trained policy.
- `benchmark.py` trains every algorithm of `--kind rl|marl`, or the `--algos` listed, and prints a table. Runs and `results.json` go to `runs/benchmark/<timestamp>/` (`--save-dir`).

```bash
python train.py --algo sac --steps 20000 --seed 1 --set hparams.actor_lr=0.001 "hparams.hidden_sizes=[128,128]"
python train.py --algo vdn --env matrix_game --set env.kwargs.payoff=climbing
python train.py --config configs/rl/ppo.yaml --watch
python evaluate.py runs/ppo_cartpole --episodes 20
python watch.py runs/ppo_cartpole --save ppo.html --no-window
python plot.py runs/dqn_* runs/rainbow_* --metric eval/return --out compare.png
python collect.py --run runs/ppo_cartpole --steps 50000 --epsilon 0.1 --out datasets/cartpole_medium.npz
python benchmark.py --algos qmix vdn qplex --env matrix_game --steps 10000 --seeds 0 1 2
```

A run directory is `runs/<algo>_<env>_s<seed>_<YYYYmmdd-HHMMSS>/` unless `--run-dir` is given. It contains `config.yaml` (the full resolved config, every default filled in), `model.pt` (weights), `progress.jsonl` (one JSON line per log step: `rollout/*`, `train/*`, `eval/*`, `time/fps`, `final/*`) and `summary.json` (algo, env, steps, wall time, final evaluation, hardware), plus TensorBoard files when `train.tensorboard` is true.

```python
from rlzoo.runner import evaluate, load_run

algo, cfg = load_run("runs/ppo_cartpole", device="cpu")     # rebuilds from config.yaml, loads model.pt
print(evaluate("rl", algo, cfg["env"], episodes=5, seed=0))   # {'return': ..., 'return_std': ..., 'length': ...}
actions = algo.act(obs_batch, deterministic=True)            # obs_batch: (N, obs_dim) flat observations
# multi-agent runs: evaluate("marl", ...) and algo.act(obs, state, avail, deterministic=True)
```

## Algorithms

Names are as used by `--algo`; `python train.py --list` adds the paper reference for each.

| Category | Algorithms |
| --- | --- |
| **Single-agent (40)** | |
| Value-based | `dqn`, `double_dqn`, `dueling_dqn`, `per_dqn`, `noisy_dqn`, `m_dqn` |
| Distributional | `c51`, `qr_dqn`, `iqn`, `rainbow` |
| Policy gradient | `reinforce` |
| Actor-critic | `a2c`, `a3c`, `impala`, `sac_discrete` |
| Trust region | `trpo`, `ppo`, `ppg` |
| Continuous control | `ddpg`, `td3`, `sac`, `tqc`, `redq`, `droq`, `crossq` |
| Recurrent | `drqn`, `ppo_rnn` |
| Exploration | `rnd`, `icm` |
| Goal-conditioned | `her_dqn`, `her_td3`, `her_sac` |
| Model-based | `mbpo` |
| Offline RL / imitation learning | `bc`, `td3_bc`, `cql`, `iql_offline`, `awac`, `dt`, `gail` |
| **Multi-agent (20)** | |
| Independent | `iql` |
| Value decomposition | `vdn`, `qmix`, `qtran`, `qplex`, `wqmix`, `qatten` |
| Centralized critic | `coma`, `maddpg`, `matd3`, `masac`, `facmac` |
| Policy gradient | `ippo`, `mappo`, `happo`, `maa2c`, `ia2c` |
| Transformer | `mat` |
| Communication | `commnet`, `g2anet` |

- `rlzoo/rl/dqn.py` holds the whole DQN family. Variants change defaults (`double`, `dueling`, `noisy`, `prioritized`, `n_step`) or override the loss (`c51`, `qr_dqn`, `iqn`, `m_dqn`), so the ingredients can be switched in a config. `her_dqn`, `her_td3` and `her_sac` are the `double_dqn`, `td3` and `sac` classes with `hparams.her: true`; HER (`her`, `her_k`, `her_strategy`) belongs to the off-policy base class.
- `td3` is the `ddpg` class with twin critics, delayed actor updates and target smoothing. `tqc`, `redq`, `droq`, `crossq` and `mbpo` subclass `sac` (`redq` and `droq` only change its hyperparameters); `ppg`, `rnd`, `icm` and `gail` subclass `ppo`. `a3c` runs `n_workers` Python threads with a shared optimizer. `impala` is synchronous: the behaviour policy lags by `policy_lag` updates and V-trace corrects for it.
- `bc`, `td3_bc`, `cql`, `iql_offline`, `awac` and `dt` share a base class in `rlzoo/rl/offline.py` and count `train.total_steps` in gradient steps. `gail` trains online with PPO.
- The value-decomposition methods extend `iql`, whose Q-network is shared by all agents, with a mixer from `rlzoo/marl/mixers.py` (`qtran` uses joint Q and V networks instead). `ippo`, `happo`, `commnet`, `g2anet`, `maa2c` and `ia2c` subclass `mappo`; `matd3` and `facmac` subclass `maddpg`.
- The DQN family, `drqn` and `sac_discrete` need discrete actions; `ddpg`, `td3`, `sac`, their subclasses and `td3_bc` need continuous ones. All multi-agent algorithms use discrete actions (`maddpg`, `matd3` and `facmac` through Gumbel-Softmax).

## Environments

- **Gymnasium:** `env.id` is a Gymnasium id and `env.kwargs` go to `gym.make`; `env.num_envs` copies are stepped in one process. The observation space must be a `Box` (vectors, or images such as `uint8` `(H, W, C)`) or a goal `Dict` with `observation`, `achieved_goal` and `desired_goal`; the action space must be `Discrete` or `Box`. Other spaces raise a `TypeError`. Images go through a CNN chosen by `hparams.encoder` (`auto`, `mlp`, `nature_cnn`, `small_cnn`, `impala_cnn`).
- **Built-in single-agent envs**, registered on import: `BitFlip-v0` (goal-conditioned bit flipping, discrete) and `PointReach-v0` (goal-conditioned 2-D point mass, continuous), used by the `her_*` configs, and `PixelGrid-v0` (grid world with an RGB image observation).
- **Built-in MARL envs**, pure NumPy, with per-agent observations, a global state, available-action masks and a shared team reward: `spread` (cooperative navigation, like MPE simple_spread), `predator_prey` (grid world with partial observability; a capture needs several predators), `matrix_game` (one-step two-agent game, `payoff: qtran | climbing | penalty` or a custom matrix).
- **MARL adapters:** `pz:<module>` wraps a PettingZoo parallel env with discrete actions (e.g. `pz:mpe2.simple_spread_v3`). `smac:<map>` wraps SMAC (e.g. `smac:3m`; needs `smac` and StarCraft II).
- **Offline datasets** are `.npz` files with `obs`, `actions`, `rewards`, `next_obs`, `terminals`, `timeouts` and `episode_starts`. The top-level config key `dataset` is `auto` (records `dataset_steps` random-policy steps to `datasets/<env>_random.npz` if the file is missing), a `.npz` path, or a run directory (records from that policy). `bc`, `td3_bc`, `cql`, `iql_offline`, `awac`, `dt` and `gail` use it.

## Viewer

`watch.py`, `evaluate.py --watch` and `train.py --watch` open a matplotlib window with the environment frame; the episode, step, action, reward, return, action mode and fps; the policy's diagnostics (Q-values, action probabilities or a value estimate, depending on the algorithm; a per-agent heatmap for multi-agent runs); and a plot of episode returns. Frames come from the env's own `rgb_array` renderer. When that is unavailable (for example without `pygame`), rlzoo draws CartPole, MountainCar, Pendulum, Acrobot and the built-in MARL envs itself; `pz:` and `smac:` envs have no renderer. Without an interactive matplotlib backend the viewer writes `<run_dir>/episode.html` instead. Exports are `.gif` (Pillow), `.mp4` (needs ffmpeg) or `.html` (a self-contained player with a step slider and the same diagnostics).

| Key | Action |
| --- | --- |
| `space` | play / pause |
| `n` or `→` | one step, then pause |
| `r` | reset to a new episode |
| `d` | toggle deterministic / stochastic actions |
| `+` (or `=`) / `-` | faster / slower |
| `s` | save a screenshot PNG |
| `q` | quit |

## Multi-GPU

```bash
python train.py --algo ppo --parallel dp --gpu-ids 0,1   # nn.DataParallel in one process
torchrun --nproc_per_node=2 train.py --algo ppo          # DistributedDataParallel, one process per GPU
python launch_ddp.py --nproc 2 train.py --algo ppo       # the same without torchrun; also runs on CPU (gloo)
```

The flags `--device`, `--parallel` (`--ddp` is short for `--parallel ddp`), `--gpu-ids`, `--backend`, `--master-addr` and `--master-port` come from `rlzoo/common/hwaccel.py`, which is kept byte-identical with `utils/hwaccel.py` in [Mutli-Tool4Vision](https://github.com/MZUI3/Mutli-Tool4Vision); the YAML `accel:` block takes the same options. Under `dp`, every sub-network with its own `forward` is wrapped in `nn.DataParallel` and each batch is split across the GPUs; with fewer than two GPUs it falls back to one device. Under `ddp`, each rank steps its own environments with its own seed, parameters are broadcast from rank 0, and gradients are averaged before every optimizer step. Batch sizes are per process, and `train.total_steps` counts the steps of all ranks. Evaluation, logging and saving run on rank 0; after training, rlzoo checks that all ranks hold the same weights. `torchrun` is detected automatically. `a3c` does not support `ddp`.

## Configuration

```yaml
algo: ppo                 # registry name
seed: 0
device: auto              # alias for accel.device: auto | cpu | cuda | cuda:N
env:
  id: CartPole-v1         # Gymnasium id, spread | predator_prey | matrix_game, pz:<module>, smac:<map>
train:
  total_steps: 100000     # env steps over all envs and ranks
  eval_interval: 20000    # 0 turns periodic evaluation off
hparams:                  # only what differs from the algorithm's defaults
  lr: 3.0e-4
```

- `algo` and `env.id` are required. `seed` defaults to 0, `env.num_envs` to 1, `env.kwargs` to `{}`. `train` defaults: `total_steps: 100000`, `log_interval: 2000`, `eval_interval: 10000`, `eval_episodes: 5`, `save_dir: runs`, `tensorboard: false`, `save_model: true`.
- Each algorithm class declares its `hparams` defaults, merged along the class hierarchy. An unknown hparam stops the run with an error that lists the valid keys; unknown `accel` keys are rejected the same way. Keys under `train` and at the top level are not checked, so a misspelled one is ignored.
- Precedence: the YAML file, then `--algo`, `--env`, `--seed` and `--steps`, then `--set`, then hardware flags given on the command line. Values such as `3e-4` are read as numbers, not strings.

## Project structure

```
train.py, evaluate.py, watch.py, plot.py, collect.py, benchmark.py, launch_ddp.py   # entry points
configs/rl/, configs/marl/   # one default config per algorithm (40 + 20)
rlzoo/registry.py            # algorithm name -> class, family, paper reference
rlzoo/config.py, runner.py   # YAML loading, overrides, defaults; build, train, evaluate, load_run
rlzoo/common/                # base.py (Algorithm), nets.py, buffers.py, datasets.py, logger.py, utils.py, hwaccel.py
rlzoo/envs/                  # gym_env.py (VecEnv), builtin.py (BitFlip, PointReach, PixelGrid), marl/
rlzoo/envs/marl/             # base.py, spread.py, predator_prey.py, matrix_game.py, external.py (PettingZoo, SMAC)
rlzoo/rl/                    # base, dqn, actor_critic, trpo, ppo, ppg, continuous, sac_discrete, recurrent,
                             # impala, curiosity, mbpo, offline, imitation (.py)
rlzoo/marl/                  # base, value_based, mixers, coma, maddpg, masac, ppo_family, mat, networks (.py)
rlzoo/viz/                   # player.py (viewer, export), renderers.py (fallback drawings)
tests/test_smoke.py, requirements.txt, environment.yml, setup.sh
```

## Tests

```bash
python tests/test_smoke.py            # every algorithm for 600 steps; checks that it runs, not that it learns
python tests/test_smoke.py qmix ppo   # a subset
pytest tests/                         # the same through pytest
```

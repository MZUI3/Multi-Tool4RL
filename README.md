# rlzoo

## Overview

강화학습 알고리즘 60개(단일 에이전트 40, 멀티에이전트 20)를 PyTorch로 직접 구현한 저장소입니다. SB3, RLlib, PyMARL 같은 RL 라이브러리는 쓰지 않습니다.
알고리즘, 환경, 하이퍼파라미터는 YAML 하나로 정하고 `train.py`로 돌립니다. 학습이 끝나면 `--watch`로 에이전트가 움직이는 것을 바로 볼 수 있습니다.

## Usage

```bash
pip install -r requirements.txt

python train.py --list                           # 알고리즘 목록
python train.py --algo ppo --watch               # CartPole 학습 후 뷰어
python train.py --algo qmix --env predator_prey  # MARL (내장 numpy 환경)
python train.py --algo sac --steps 20000 --set hparams.actor_lr=1e-3 hparams.hidden_sizes=[128,128]

python evaluate.py runs/<run_dir> --episodes 20
python watch.py runs/<run_dir> --save episode.html   # .gif / .mp4 / .html
python plot.py runs/dqn_* runs/rainbow_* --metric eval/return --out compare.png
```

결과는 `runs/<algo>_<env>_s<seed>_<시각>/`에 `config.yaml`, `progress.jsonl`, `model.pt`, `summary.json`으로 저장됩니다.

```python
from rlzoo.runner import load_run

algo, cfg = load_run("runs/ppo_CartPole-v1_s0_...")
action = algo.act(obs[None], deterministic=True)[0]
```

## Algorithms

| | |
|---|---|
| Value-based | DQN, Double DQN, Dueling DQN, PER, Noisy DQN, Munchausen DQN, C51, QR-DQN, IQN, Rainbow, DRQN |
| Policy gradient | REINFORCE, A2C, A3C, IMPALA, TRPO, PPO, PPO-GRU, PPG |
| Continuous control | DDPG, TD3, SAC, Discrete SAC, TQC, REDQ, DroQ, CrossQ |
| Exploration / goals | RND, ICM, HER (DQN, TD3, SAC) |
| Model-based | MBPO |
| Offline / imitation | BC, TD3+BC, CQL, IQL, AWAC, Decision Transformer, GAIL |
| Multi-agent | IQL, VDN, QMIX, QTRAN, QPLEX, WQMIX, Qatten, COMA, MADDPG, MATD3, MASAC, FACMAC, IPPO, MAPPO, HAPPO, MAA2C, IA2C, MAT, CommNet, G2ANet |

DQN 계열은 구현 하나를 공유해서 `double`, `dueling`, `noisy`, `prioritized`, `n_step`을 아무 조합으로나 켤 수 있고, REDQ / DroQ / CrossQ도 SAC의 하이퍼파라미터 변형입니다. 논문 출처는 `python train.py --list`에 있습니다.

## Environments

- gymnasium 환경 전부. Box 관측, 이미지(uint8) 관측, 목표 조건 Dict 관측을 지원하고 이미지면 `hparams.encoder`(nature_cnn, impala_cnn)가 CNN을 붙입니다.
- 내장: `BitFlip-v0`, `PointReach-v0` (HER용), `PixelGrid-v0` (이미지 관측)
- MARL 내장: `spread`, `predator_prey`, `matrix_game`. PettingZoo(`pz:<모듈>`), SMAC(`smac:<맵>`) 어댑터.

오프라인 알고리즘은 `dataset: path.npz | auto | runs/<run_dir>`로 데이터를 받습니다. `collect.py`로 무작위 정책이나 학습된 정책에서 데이터셋을 만듭니다.

## Viewer

`watch.py`는 matplotlib 창에 환경 화면, 행동/보상/리턴, 알고리즘 진단(Q값, 행동 확률, 가치)을 보여줍니다. `space` 일시정지, `n` 한 스텝, `r` 리셋, `d` 결정적/확률적, `+`/`-` 속도.
환경에 렌더러가 없으면 상태를 보고 직접 그리므로 pygame 없이도 classic control과 내장 환경이 보입니다.

## Multi-GPU

```bash
python train.py --algo sac --parallel dp --gpu-ids 0,1
torchrun --nproc_per_node=4 train.py --algo redq
python launch_ddp.py --nproc 4 train.py --algo redq     # torchrun이 안 될 때
```

DDP는 rank마다 환경을 따로 돌리고 업데이트 때 그래디언트를 평균합니다. 가속 코드(`rlzoo/common/hwaccel.py`)는 [Mutli-Tool4Vision](https://github.com/MZUI3/Mutli-Tool4Vision)과 같은 파일입니다.

## Config

```yaml
algo: qmix
seed: 0
env: {id: spread, kwargs: {n_agents: 3}, num_envs: 1}
train: {total_steps: 200000, eval_interval: 25000}
hparams: {lr: 5.0e-4, batch_size: 512}   # 안 적으면 알고리즘 기본값, 없는 키는 에러
```

## Structure

```
train.py / evaluate.py / watch.py / collect.py / plot.py / benchmark.py
configs/rl/*.yaml  configs/marl/*.yaml
rlzoo/
  registry.py  runner.py  config.py
  common/      base.py, nets.py, buffers.py, datasets.py, hwaccel.py
  envs/        gymnasium VecEnv, builtin.py, marl/
  rl/          dqn, actor_critic, ppo, trpo, continuous, recurrent, ppg, impala, curiosity, offline, imitation, mbpo
  marl/        value_based, mixers, coma, maddpg, masac, ppo_family, mat
  viz/         renderers.py, player.py
tests/test_smoke.py
```

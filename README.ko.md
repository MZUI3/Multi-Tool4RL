# rlzoo

[English](README.md) | 한국어

rlzoo는 강화학습 알고리즘 60개(단일 에이전트 40개, 멀티에이전트 20개)를 PyTorch로 구현한 저장소입니다. 모든 알고리즘을 YAML 설정을 읽는 스크립트 하나로 학습합니다. Stable-Baselines3, RLlib, PyMARL 같은 다른 RL 라이브러리에는 의존하지 않으며, 리플레이 버퍼, 벡터 환경, 내장 환경, 뷰어가 모두 이 저장소에 들어 있습니다.

## 설치

```bash
pip install -r requirements.txt       # torch>=2.0, numpy, gymnasium>=0.29, pyyaml, matplotlib, pillow
conda env create -f environment.yml   # Python 3.11 conda 환경 "rlzoo"
bash setup.sh                         # conda가 있으면 conda 환경 "rlzoo", 없으면 .venv
```

`setup.sh`는 `--venv`(venv 강제), `--conda`, `--extras`(`gymnasium[box2d]`, `pettingzoo`, `mpe2` 추가 설치)도 받고, 마지막에 `python train.py --list`를 실행합니다. rlzoo 자체는 패키지로 설치하지 않습니다. 스크립트는 저장소 루트에서 실행하고, `runs/`와 `datasets/`는 작업 디렉터리에 생깁니다. 선택 사항: `pygame`(Gymnasium 자체의 classic control 렌더러), `ffmpeg`(시스템 패키지, `.mp4` 내보내기), `tensorboard`, `pytest`, 그리고 환경 패키지 `gymnasium[box2d]`, `gymnasium[mujoco]`, `pettingzoo` + `mpe2`, `smac`(StarCraft II 필요).

## 빠른 시작

```bash
git clone https://github.com/MZUI3/Multi-Tool4RL.git && cd Multi-Tool4RL
pip install -r requirements.txt
python train.py --list                                   # 알고리즘 목록(계열, 논문 출처 포함)
python train.py --algo ppo --run-dir runs/ppo_cartpole   # configs/rl/ppo.yaml: CartPole-v1에서 PPO 학습
python watch.py runs/ppo_cartpole                        # 실시간 뷰어
```

## 사용법

- `train.py`는 `configs/<rl|marl>/<algo>.yaml`(`--algo`)이나 임의의 YAML 파일(`--config`)로 학습합니다. `--env`는 `env.id`를 바꾸고 `env.kwargs`를 비웁니다. `--set`은 점으로 구분한 `KEY=VALUE` 쌍을 받으며, 값은 YAML로 해석합니다. `--watch`는 학습이 끝난 뒤 뷰어를 열고, `--watch-save FILE`은 에피소드를 `.gif`, `.mp4`, `.html`로 저장합니다. Ctrl+C로 학습을 멈춰도 그때까지의 모델을 저장하고 평가합니다.
- `evaluate.py`는 저장된 정책을 결정적으로 실행해 평균 리턴, 리턴의 표준편차, 평균 에피소드 길이, 환경 지표를 출력합니다. 기본값은 `--episodes 10`, `--seed 12345`입니다. `--watch`는 뷰어를 엽니다.
- `watch.py`는 [뷰어](#뷰어)를 열거나 `--save`로 에피소드를 내보냅니다. 그 밖의 플래그: `--no-window`, `--episodes`(3, 내보내기용), `--stochastic`, `--seed`, `--fps`, `--max-steps`, `--device`.
- `plot.py`는 `progress.jsonl`의 지표 하나(기본값 `rollout/return`)를 run 하나 또는 여러 개에 대해 그립니다. `--out`을 주면 창을 여는 대신 이미지 파일로 저장합니다.
- `collect.py`는 무작위 정책(`--env ID`, `datasets/<env>_random.npz`로 저장)이나 학습된 run(`--run DIR`)으로 단일 에이전트용 오프라인 데이터셋을 기록합니다. `--noise`(연속 행동)와 `--epsilon`(이산 행동)으로 학습된 정책을 일부러 흐트러뜨릴 수 있습니다.
- `benchmark.py`는 `--kind rl|marl`에 속한 모든 알고리즘이나 `--algos`로 준 알고리즘을 학습하고 표를 출력합니다. run과 `results.json`은 `runs/benchmark/<timestamp>/`(`--save-dir`)에 저장됩니다.

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

run 디렉터리는 `--run-dir`를 주지 않으면 `runs/<algo>_<env>_s<seed>_<YYYYmmdd-HHMMSS>/`입니다. 여기에는 `config.yaml`(기본값까지 모두 채운 최종 설정), `model.pt`(가중치), `progress.jsonl`(로그 단계마다 JSON 한 줄: `rollout/*`, `train/*`, `eval/*`, `time/fps`, `final/*`), `summary.json`(알고리즘, 환경, 스텝 수, 소요 시간, 최종 평가, 하드웨어)이 들어 있고, `train.tensorboard`가 true면 TensorBoard 파일도 생깁니다.

```python
from rlzoo.runner import evaluate, load_run

algo, cfg = load_run("runs/ppo_cartpole", device="cpu")     # config.yaml로 알고리즘을 다시 만들고 model.pt를 불러옴
print(evaluate("rl", algo, cfg["env"], episodes=5, seed=0))   # {'return': ..., 'return_std': ..., 'length': ...}
actions = algo.act(obs_batch, deterministic=True)            # obs_batch: 평탄화한 관측 (N, obs_dim)
# 멀티에이전트 run은 evaluate("marl", ...)와 algo.act(obs, state, avail, deterministic=True)를 씀
```

## 알고리즘

이름은 `--algo`에 넣는 그대로입니다. `python train.py --list`는 알고리즘마다 논문 출처도 보여 줍니다.

| 분류 | 알고리즘 |
| --- | --- |
| **단일 에이전트 (40)** | |
| 가치 기반 | `dqn`, `double_dqn`, `dueling_dqn`, `per_dqn`, `noisy_dqn`, `m_dqn` |
| 분포형 | `c51`, `qr_dqn`, `iqn`, `rainbow` |
| 정책 경사 | `reinforce` |
| 액터-크리틱 | `a2c`, `a3c`, `impala`, `sac_discrete` |
| 신뢰 영역 | `trpo`, `ppo`, `ppg` |
| 연속 제어 | `ddpg`, `td3`, `sac`, `tqc`, `redq`, `droq`, `crossq` |
| 순환 신경망 | `drqn`, `ppo_rnn` |
| 탐색 | `rnd`, `icm` |
| 목표 조건 | `her_dqn`, `her_td3`, `her_sac` |
| 모델 기반 | `mbpo` |
| 오프라인 RL / 모방 학습 | `bc`, `td3_bc`, `cql`, `iql_offline`, `awac`, `dt`, `gail` |
| **멀티에이전트 (20)** | |
| 독립 학습 | `iql` |
| 가치 분해 | `vdn`, `qmix`, `qtran`, `qplex`, `wqmix`, `qatten` |
| 중앙집중 크리틱 | `coma`, `maddpg`, `matd3`, `masac`, `facmac` |
| 정책 경사 | `ippo`, `mappo`, `happo`, `maa2c`, `ia2c` |
| 트랜스포머 | `mat` |
| 통신 | `commnet`, `g2anet` |

- `rlzoo/rl/dqn.py`에 DQN 계열이 모두 들어 있습니다. 변형들은 기본값(`double`, `dueling`, `noisy`, `prioritized`, `n_step`)을 바꾸거나 손실 함수를 재정의(`c51`, `qr_dqn`, `iqn`, `m_dqn`)하므로, 이 요소들을 설정에서 켜고 끌 수 있습니다. `her_dqn`, `her_td3`, `her_sac`는 `double_dqn`, `td3`, `sac` 클래스에 `hparams.her: true`를 켠 것이고, HER(`her`, `her_k`, `her_strategy`)는 오프폴리시 기본 클래스에 들어 있습니다.
- `td3`는 `ddpg` 클래스에 쌍둥이 크리틱, 지연된 액터 업데이트, 타깃 스무딩을 더한 것입니다. `tqc`, `redq`, `droq`, `crossq`, `mbpo`는 `sac`의 하위 클래스이고(`redq`와 `droq`는 하이퍼파라미터만 바꿈), `ppg`, `rnd`, `icm`, `gail`은 `ppo`의 하위 클래스입니다. `a3c`는 `n_workers`개의 Python 스레드로 돌며 옵티마이저 하나를 공유합니다. `impala`는 동기식입니다. 행동 정책이 학습자보다 `policy_lag` 업데이트만큼 늦게 따라오고, 그 차이는 V-trace가 보정합니다.
- `bc`, `td3_bc`, `cql`, `iql_offline`, `awac`, `dt`는 `rlzoo/rl/offline.py`의 기본 클래스를 공유하며, `train.total_steps`를 그래디언트 스텝 수로 셉니다. `gail`은 PPO로 온라인 학습합니다.
- 가치 분해 방법들은 `iql`(모든 에이전트가 Q 네트워크 하나를 공유)에 `rlzoo/marl/mixers.py`의 믹서를 더한 것입니다(`qtran`은 믹서 대신 결합 Q 네트워크와 V 네트워크를 씀). `ippo`, `happo`, `commnet`, `g2anet`, `maa2c`, `ia2c`는 `mappo`의 하위 클래스이고, `matd3`와 `facmac`은 `maddpg`의 하위 클래스입니다.
- DQN 계열, `drqn`, `sac_discrete`는 이산 행동이 필요하고, `ddpg`, `td3`, `sac`와 그 하위 클래스, `td3_bc`는 연속 행동이 필요합니다. 멀티에이전트 알고리즘은 모두 이산 행동을 씁니다(`maddpg`, `matd3`, `facmac`은 Gumbel-Softmax를 거침).

## 환경

- **Gymnasium:** `env.id`는 Gymnasium id이고, `env.kwargs`는 `gym.make`에 넘어갑니다. `env.num_envs`개의 환경 사본을 프로세스 하나에서 돌립니다. 관측 공간은 `Box`(벡터, 또는 `uint8` `(H, W, C)` 같은 이미지)이거나 `observation`, `achieved_goal`, `desired_goal`을 가진 목표 조건 `Dict`여야 하고, 행동 공간은 `Discrete`나 `Box`여야 합니다. 그 밖의 공간은 `TypeError`가 납니다. 이미지는 `hparams.encoder`(`auto`, `mlp`, `nature_cnn`, `small_cnn`, `impala_cnn`)로 고른 CNN을 거칩니다.
- **내장 단일 에이전트 환경**(import할 때 등록): `her_*` 설정이 쓰는 `BitFlip-v0`(목표 조건 비트 뒤집기, 이산)과 `PointReach-v0`(목표 조건 2차원 질점, 연속), 그리고 `PixelGrid-v0`(RGB 이미지를 관측으로 주는 격자 세계).
- **내장 MARL 환경**: 순수 NumPy로 만들었고 에이전트별 관측, 전역 상태, 가능한 행동 마스크, 공유 팀 보상을 줍니다. `spread`(MPE simple_spread와 비슷한 협력 내비게이션), `predator_prey`(부분 관측 격자 세계, 먹이를 잡으려면 포식자 여럿이 필요), `matrix_game`(한 스텝짜리 2인 게임, `payoff: qtran | climbing | penalty` 또는 직접 준 행렬).
- **MARL 어댑터:** `pz:<module>`은 이산 행동을 쓰는 PettingZoo parallel 환경을 감쌉니다(예: `pz:mpe2.simple_spread_v3`). `smac:<map>`은 SMAC을 감쌉니다(예: `smac:3m`, `smac`과 StarCraft II 필요).
- **오프라인 데이터셋**은 `obs`, `actions`, `rewards`, `next_obs`, `terminals`, `timeouts`, `episode_starts`를 담은 `.npz` 파일입니다. 최상위 설정 키 `dataset`은 `auto`(파일이 없으면 무작위 정책으로 `dataset_steps` 스텝을 기록해 `datasets/<env>_random.npz`에 저장), `.npz` 경로, run 디렉터리(그 정책으로 기록) 중 하나입니다. `bc`, `td3_bc`, `cql`, `iql_offline`, `awac`, `dt`, `gail`이 이 키를 씁니다.

## 뷰어

`watch.py`, `evaluate.py --watch`, `train.py --watch`는 matplotlib 창을 엽니다. 창에는 환경 화면, 에피소드·스텝·행동·보상·리턴·행동 방식(결정적/확률적)·fps, 정책의 진단 정보(알고리즘에 따라 Q값, 행동 확률, 가치 추정, 멀티에이전트 run은 에이전트별 히트맵), 에피소드 리턴 그래프가 나옵니다. 화면은 환경 자체의 `rgb_array` 렌더러로 그립니다. 그 렌더러를 쓸 수 없으면(예: `pygame`이 없을 때) CartPole, MountainCar, Pendulum, Acrobot과 내장 MARL 환경은 rlzoo가 직접 그립니다. `pz:`와 `smac:` 환경에는 렌더러가 없습니다. 대화형 matplotlib 백엔드가 없으면 창 대신 `<run_dir>/episode.html`을 씁니다. 내보내기는 `.gif`(Pillow), `.mp4`(ffmpeg 필요), `.html`(스텝 슬라이더와 같은 진단 정보를 담은 독립 실행형 플레이어)을 지원합니다.

| 키 | 동작 |
| --- | --- |
| `space` | 재생 / 일시정지 |
| `n` 또는 `→` | 한 스텝 진행 후 일시정지 |
| `r` | 새 에피소드로 리셋 |
| `d` | 결정적 / 확률적 행동 전환 |
| `+` (또는 `=`) / `-` | 빠르게 / 느리게 |
| `s` | 스크린샷 PNG 저장 |
| `q` | 종료 |

## 멀티 GPU

```bash
python train.py --algo ppo --parallel dp --gpu-ids 0,1   # 프로세스 하나에서 nn.DataParallel
torchrun --nproc_per_node=2 train.py --algo ppo          # DistributedDataParallel, GPU마다 프로세스 하나
python launch_ddp.py --nproc 2 train.py --algo ppo       # torchrun 없이 같은 방식, CPU(gloo)에서도 동작
```

`--device`, `--parallel`(`--ddp`는 `--parallel ddp`의 줄임), `--gpu-ids`, `--backend`, `--master-addr`, `--master-port` 플래그는 `rlzoo/common/hwaccel.py`에서 옵니다. 이 파일은 [Mutli-Tool4Vision](https://github.com/MZUI3/Mutli-Tool4Vision)의 `utils/hwaccel.py`와 바이트 단위로 같게 유지하며, YAML의 `accel:` 블록도 같은 옵션을 받습니다. `dp`에서는 자체 `forward`가 있는 하위 네트워크를 모두 `nn.DataParallel`로 감싸고 배치를 GPU들에 나눕니다. GPU가 두 개 미만이면 장치 하나로 돌아갑니다. `ddp`에서는 각 랭크가 자기 시드로 자기 환경을 돌리고, 랭크 0의 파라미터를 다른 랭크에 복사하며, 옵티마이저 스텝 전마다 그래디언트를 평균합니다. 배치 크기는 프로세스당 크기이고, `train.total_steps`는 모든 랭크의 스텝을 합쳐 셉니다. 평가, 로그, 저장은 랭크 0에서만 하며, 학습이 끝나면 모든 랭크의 가중치가 같은지 확인합니다. `torchrun`으로 실행하면 자동으로 감지합니다. `a3c`는 `ddp`를 지원하지 않습니다.

## 설정

```yaml
algo: ppo                 # 레지스트리 이름
seed: 0
device: auto              # accel.device의 별칭: auto | cpu | cuda | cuda:N
env:
  id: CartPole-v1         # Gymnasium id, spread | predator_prey | matrix_game, pz:<module>, smac:<map>
train:
  total_steps: 100000     # 모든 환경과 랭크를 합친 환경 스텝 수
  eval_interval: 20000    # 0이면 주기적 평가를 끔
hparams:                  # 알고리즘 기본값과 다른 것만 적음
  lr: 3.0e-4
```

- `algo`와 `env.id`는 필수입니다. 기본값은 `seed` 0, `env.num_envs` 1, `env.kwargs` `{}`입니다. `train`의 기본값: `total_steps: 100000`, `log_interval: 2000`, `eval_interval: 10000`, `eval_episodes: 5`, `save_dir: runs`, `tensorboard: false`, `save_model: true`.
- 각 알고리즘 클래스가 `hparams` 기본값을 선언하고, 이 값은 클래스 상속 계층을 따라 합쳐집니다. 모르는 hparam이 있으면 올바른 키 목록을 보여 주며 오류로 멈추고, 모르는 `accel` 키도 같은 식으로 거부합니다. `train` 아래와 최상위 키는 검사하지 않으므로, 오타가 있어도 조용히 무시됩니다.
- 우선순위: YAML 파일, 그다음 `--algo`, `--env`, `--seed`, `--steps`, 그다음 `--set`, 마지막으로 명령줄에서 준 하드웨어 플래그. `3e-4` 같은 값은 문자열이 아니라 숫자로 읽습니다.

## 프로젝트 구조

```
train.py, evaluate.py, watch.py, plot.py, collect.py, benchmark.py, launch_ddp.py   # 진입점
configs/rl/, configs/marl/   # 알고리즘마다 기본 설정 하나(40 + 20)
rlzoo/registry.py            # 알고리즘 이름 -> 클래스, 계열, 논문 출처
rlzoo/config.py, runner.py   # YAML 로딩, 오버라이드, 기본값; 빌드, 학습, 평가, load_run
rlzoo/common/                # base.py(Algorithm), nets.py, buffers.py, datasets.py, logger.py, utils.py, hwaccel.py
rlzoo/envs/                  # gym_env.py(VecEnv), builtin.py(BitFlip, PointReach, PixelGrid), marl/
rlzoo/envs/marl/             # base.py, spread.py, predator_prey.py, matrix_game.py, external.py(PettingZoo, SMAC)
rlzoo/rl/                    # base, dqn, actor_critic, trpo, ppo, ppg, continuous, sac_discrete, recurrent,
                             # impala, curiosity, mbpo, offline, imitation (.py)
rlzoo/marl/                  # base, value_based, mixers, coma, maddpg, masac, ppo_family, mat, networks (.py)
rlzoo/viz/                   # player.py(뷰어, 내보내기), renderers.py(대체 렌더링)
tests/test_smoke.py, requirements.txt, environment.yml, setup.sh
```

## 테스트

```bash
python tests/test_smoke.py            # 모든 알고리즘을 600스텝씩 실행; 학습이 되는지가 아니라 돌아가는지만 확인
python tests/test_smoke.py qmix ppo   # 일부만
pytest tests/                         # pytest로 같은 테스트
```

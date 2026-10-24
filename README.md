# rlzoo

강화학습 알고리즘을 PyTorch로 하나씩 직접 구현해서 모아 둔 저장소입니다. SB3, RLlib, PyMARL 같은 RL 라이브러리는 쓰지 않습니다.
단일 에이전트 40개, 멀티에이전트 20개가 들어 있고, 전부 YAML 설정 하나와 `train.py`로 돌립니다.
학습이 끝나면 `--watch`로 에이전트가 움직이는 걸 바로 볼 수 있습니다.

```bash
pip install -r requirements.txt
python train.py --list                       # 알고리즘 목록
python train.py --algo ppo --watch           # CartPole 학습 후 뷰어 열기
python train.py --algo qmix                  # MARL (내장 numpy 환경)
```

의존성은 torch, numpy, gymnasium, pyyaml, matplotlib, pillow입니다. MARL 환경 3개(Spread, Predator-Prey, Matrix game)는 numpy로 직접 짜서 추가 설치가 없고, PettingZoo와 SMAC은 어댑터만 있습니다.

## 설치

```bash
pip install -r requirements.txt
```

또는 `bash setup.sh`. conda가 있으면 `rlzoo` 환경을, 없으면 `.venv`를 만듭니다. `--extras`를 주면 PettingZoo와 Box2D도 설치합니다.

## 사용법

```bash
python train.py --algo ppo                      # configs/rl/ppo.yaml
python train.py --config configs/marl/qmix.yaml # 설정 파일 직접 지정
python train.py --algo sac --steps 20000 --seed 3 --set hparams.actor_lr=1e-3 hparams.hidden_sizes=[128,128]
python train.py --algo mappo --set env.kwargs.n_agents=4 env.num_envs=16
python train.py --algo qmix --env predator_prey --set env.kwargs.grid_size=9 env.kwargs.n_prey=3
python train.py --algo rainbow --set hparams.noisy=false hparams.n_step=1   # Rainbow 구성요소 끄기
```

결과는 `runs/<algo>_<env>_s<seed>_<시각>/`에 들어갑니다.

| 파일 | 내용 |
|---|---|
| `config.yaml` | 실제로 쓰인 전체 설정. 기본 하이퍼파라미터까지 다 적혀 있어서 그대로 재현 가능 |
| `progress.jsonl` | 학습 지표 (`rollout/return`, `eval/return`, `train/*`, `time/fps`) |
| `model.pt` | 네트워크 가중치 |
| `summary.json` | 최종 평가 결과 |

```bash
python evaluate.py runs/<run_dir> --episodes 20
python watch.py runs/<run_dir>                           # 뷰어
python plot.py runs/dqn_* runs/rainbow_* --metric eval/return --out compare.png
python benchmark.py --kind rl                            # 여러 알고리즘 연속 실행 후 표로
python benchmark.py --algos qmix vdn qplex --seeds 0 1 2 --env matrix_game --steps 10000
python tests/test_smoke.py                               # 60개 전부 몇백 스텝씩 돌려서 깨진 게 없는지 확인
```

## 시뮬레이션 보기

학습된 모델이 환경에서 어떻게 움직이는지 창을 띄워서 보거나, GIF / MP4 / HTML로 저장합니다.

```bash
python train.py --algo ppo --watch                       # 학습 끝나면 바로 열기
python train.py --algo qmix --watch-save qmix.html       # 학습 끝나면 HTML로 저장
python watch.py runs/<run_dir>                           # 저장된 run 열기
python watch.py runs/<run_dir> --save ep.gif --no-window --episodes 3
python watch.py runs/<run_dir> --save ep.html --stochastic --fps 20
python evaluate.py runs/<run_dir> --watch
```

창은 matplotlib으로 만들었습니다. 왼쪽이 환경 화면, 오른쪽에 에피소드/스텝/행동/보상/리턴, 그리고 알고리즘별 진단(DQN 계열은 Q값, 정책 계열은 행동 확률과 가치, MARL은 에이전트×행동 Q/확률 히트맵)과 에피소드 리턴 그래프가 나옵니다.
키는 `space` 일시정지, `n` 또는 `→` 한 스텝, `r` 리셋, `d` 결정적/확률적 전환, `+`/`-` 속도, `s` 스크린샷, `q` 종료.

HTML로 저장하면 파일 하나에 프레임과 진단 정보가 다 들어가서 브라우저에서 재생하고 스크럽할 수 있습니다. GIF는 pillow, MP4는 ffmpeg가 필요합니다.

렌더링은 환경의 `rgb_array` 렌더러를 먼저 쓰고, 그게 없으면 상태를 보고 직접 그립니다. gymnasium의 classic control은 pygame이 있어야 자체 렌더링이 되는데, 없어도 CartPole, MountainCar, Pendulum, Acrobot은 직접 그리는 쪽으로 넘어갑니다. 내장 환경(BitFlip, PointReach, PixelGrid, Spread, Predator-Prey, Matrix game)도 마찬가지입니다. 다른 환경을 그리려면 `rlzoo/viz/renderers.py`의 `_BY_CLASS`에 함수를 추가하면 됩니다.

## 알고리즘

### 단일 에이전트 (40)

| 이름 | 계열 | 기본 환경 | 논문 |
|---|---|---|---|
| `dqn` | Value-based | CartPole-v1 | Mnih et al., 2015 |
| `double_dqn` | Value-based | CartPole-v1 | van Hasselt et al., 2016 |
| `dueling_dqn` | Value-based (+Double) | CartPole-v1 | Wang et al., 2016 |
| `per_dqn` | Value-based (+Double, PER) | CartPole-v1 | Schaul et al., 2016 |
| `noisy_dqn` | Value-based (NoisyNet) | CartPole-v1 | Fortunato et al., 2018 |
| `m_dqn` | Value-based (Munchausen) | CartPole-v1 | Vieillard et al., 2020 |
| `c51` | Distributional | CartPole-v1 | Bellemare et al., 2017 |
| `qr_dqn` | Distributional | CartPole-v1 | Dabney et al., 2018 |
| `iqn` | Distributional (implicit quantile) | CartPole-v1 | Dabney et al., 2018 |
| `rainbow` | C51 + Double + Dueling + Noisy + PER + n-step | CartPole-v1 | Hessel et al., 2018 |
| `drqn` | Recurrent (GRU Q-network) | CartPole-v1 | Hausknecht & Stone, 2015 |
| `reinforce` | Policy gradient (+baseline) | CartPole-v1 | Williams, 1992 |
| `a2c` | Actor-critic | CartPole-v1 | Mnih et al., 2016 |
| `a3c` | Actor-critic (스레드 비동기) | CartPole-v1 | Mnih et al., 2016 |
| `impala` | Actor-critic (V-trace, 동기 버전) | CartPole-v1 | Espeholt et al., 2018 |
| `trpo` | Trust region | CartPole-v1 | Schulman et al., 2015 |
| `ppo` | Trust region (clip) | CartPole-v1 | Schulman et al., 2017 |
| `ppo_rnn` | PPO + GRU | CartPole-v1 | Schulman et al., 2017 |
| `ppg` | Phasic policy gradient | CartPole-v1 | Cobbe et al., 2020 |
| `sac_discrete` | 이산 SAC | CartPole-v1 | Christodoulou, 2019 |
| `ddpg` | 연속 제어 | Pendulum-v1 | Lillicrap et al., 2016 |
| `td3` | 연속 제어 | Pendulum-v1 | Fujimoto et al., 2018 |
| `sac` | 연속 제어 (최대 엔트로피) | Pendulum-v1 | Haarnoja et al., 2018 |
| `tqc` | 연속 제어 (truncated quantile critics) | Pendulum-v1 | Kuznetsov et al., 2020 |
| `redq` | 연속 제어 (critic 10개, UTD 20) | Pendulum-v1 | Chen et al., 2021 |
| `droq` | 연속 제어 (dropout + LayerNorm critic) | Pendulum-v1 | Hiraoka et al., 2022 |
| `crossq` | 연속 제어 (BatchRenorm, 타깃 네트워크 없음) | Pendulum-v1 | Bhatt et al., 2024 |
| `rnd` | 탐험 (PPO + RND) | MountainCar-v0 | Burda et al., 2019 |
| `icm` | 탐험 (PPO + ICM) | MountainCar-v0 | Pathak et al., 2017 |
| `her_dqn` | 목표 조건 (Double DQN + HER) | BitFlip-v0 | Andrychowicz et al., 2017 |
| `her_td3`, `her_sac` | 목표 조건 (TD3 / SAC + HER) | PointReach-v0 | Andrychowicz et al., 2017 |
| `mbpo` | 모델 기반 (앙상블 모델 + SAC) | Pendulum-v1 | Janner et al., 2019 |
| `bc` | 오프라인 (behaviour cloning) | CartPole-v1 | Pomerleau, 1991 |
| `td3_bc` | 오프라인 | Pendulum-v1 | Fujimoto & Gu, 2021 |
| `cql` | 오프라인 (이산 / 연속) | CartPole-v1 | Kumar et al., 2020 |
| `iql_offline` | 오프라인 | CartPole-v1 | Kostrikov et al., 2022 |
| `awac` | 오프라인 | CartPole-v1 | Nair et al., 2020 |
| `dt` | 오프라인 (Decision Transformer) | CartPole-v1 | Chen et al., 2021 |
| `gail` | 모방 (PPO + discriminator) | CartPole-v1 | Ho & Ermon, 2016 |

- DQN 계열은 구현 하나를 공유합니다. `double`, `dueling`, `noisy`, `prioritized`, `n_step`을 아무 조합으로나 켜고 끌 수 있습니다.
- `redq`, `droq`, `crossq`도 `sac`의 하이퍼파라미터 변형(`target_subset`, `actor_q`, `policy_delay`, `critic_dropout`, `critic_layer_norm`)입니다.
- `reinforce`, `a2c`, `a3c`, `trpo`, `ppo`, `ppg`, `impala`, `ppo_rnn`, `rnd`, `icm`은 이산/연속 둘 다 됩니다. `ddpg` 이하 연속 제어 계열과 `td3_bc`, `mbpo`는 Box 행동만, DQN 계열과 `drqn`, `sac_discrete`는 이산 행동만 됩니다.
- `her: true`는 off-policy 알고리즘 어디에나 줄 수 있습니다 (Dict 관측의 목표 조건 환경이 필요). `her_dqn` 등은 이걸 켜 둔 설정일 뿐입니다.
- 오프라인 알고리즘의 `train.total_steps`는 환경 스텝이 아니라 그래디언트 스텝입니다.

### 멀티에이전트 (20)

| 이름 | 계열 | 내용 | 논문 |
|---|---|---|---|
| `iql` | Independent | 에이전트별 독립 Q-learning (파라미터 공유) | Tan, 1993 |
| `vdn` | Value decomposition | Q_tot = Σ Q_i | Sunehag et al., 2018 |
| `qmix` | Value decomposition | 상태 조건 하이퍼네트워크, 단조 믹싱 | Rashid et al., 2018 |
| `qtran` | Value decomposition | Joint Q + 분해 제약 | Son et al., 2019 |
| `qplex` | Value decomposition | Duplex dueling + 어텐션 가중치 | Wang et al., 2021 |
| `wqmix` | Value decomposition | 비제약 중앙 Q* + 가중 QMIX 손실 (OW/CW) | Rashid et al., 2020 |
| `qatten` | Value decomposition | 멀티헤드 어텐션 가중 믹싱 | Yang et al., 2020 |
| `coma` | Centralized critic | counterfactual 베이스라인 | Foerster et al., 2018 |
| `maddpg` | Centralized critic | 중앙 critic + Gumbel-Softmax 액터 | Lowe et al., 2017 |
| `matd3` | Centralized critic | MADDPG + twin critic, delayed update | Ackermann et al., 2019 |
| `masac` | Centralized critic | 이산 SAC + 에이전트별 중앙 critic | — |
| `facmac` | Centralized critic | 분해 critic + QMIX 믹서 | Peng et al., 2021 |
| `ippo` | Policy gradient | 독립 PPO (로컬 critic) | de Witt et al., 2020 |
| `mappo` | Policy gradient | PPO + 전역 상태 critic | Yu et al., 2022 |
| `happo` | Policy gradient | 에이전트별 정책, 순차 업데이트 | Kuba et al., 2022 |
| `maa2c` | Policy gradient | 중앙 critic A2C | Papoudakis et al., 2021 |
| `ia2c` | Policy gradient | 독립 A2C | Papoudakis et al., 2021 |
| `mat` | Transformer | 인코더-디코더, 자기회귀 행동 생성 | Wen et al., 2022 |
| `commnet` | Communication | 은닉 상태 평균 통신 (MAPPO로 학습) | Sukhbaatar et al., 2016 |
| `g2anet` | Communication | hard + soft 어텐션 (MAPPO로 학습) | Liu et al., 2020 |

MARL은 전부 협력(team reward), 이산 행동, CTDE 설정입니다.

## 환경

gymnasium 환경은 id만 적으면 됩니다. Box 관측과 이미지(uint8 HxWxC) 관측, 목표 조건 Dict 관측(`observation` / `achieved_goal` / `desired_goal`)을 지원합니다.
이미지 관측이면 `hparams.encoder`(`auto`, `nature_cnn`, `small_cnn`, `impala_cnn`)가 CNN을 붙이고 리플레이 버퍼는 uint8로 저장합니다.

추가 설치 없이 쓰는 내장 환경:

| `env.id` | 내용 | 용도 |
|---|---|---|
| `BitFlip-v0` | n비트 뒤집기 (목표 조건) | HER |
| `PointReach-v0` | 2D 점 질량을 목표 위치로 (연속 행동, 목표 조건) | HER |
| `PixelGrid-v0` | RGB 이미지 관측 격자 길찾기 | CNN 인코더, DRQN, PPO-RNN |

MARL 환경:

| `env.id` | 내용 | 주요 `env.kwargs` |
|---|---|---|
| `spread` | MPE simple_spread 비슷. N 에이전트가 N 랜드마크를 덮고 충돌 회피 | `n_agents`, `n_landmarks`, `episode_limit`, `world_bound` |
| `predator_prey` | 격자 추격. 2명 이상이 인접해야 포획, 부분 관측 | `grid_size`, `n_agents`, `n_prey`, `sight`, `agents_to_capture` |
| `matrix_game` | 1-step 행렬 게임. 값 분해 방법 표현력 비교용 | `payoff`: `qtran` / `climbing` / `penalty` |
| `pz:<모듈>` | PettingZoo parallel env (예: `pz:mpe2.simple_spread_v3`) | 해당 env 인자 |
| `smac:<맵>` | SMAC (StarCraft II 필요) | StarCraft2Env 인자 |

```bash
# VDN/QMIX는 못 풀고 QTRAN/QPLEX/WQMIX는 푸는 고전 예제 (최적 8)
python train.py --algo qplex --env matrix_game --steps 10000
python train.py --algo mappo --env pz:mpe2.simple_spread_v3 --set env.kwargs.max_cycles=25
```

새 MARL 환경은 `rlzoo/envs/marl/base.py`의 `MultiAgentEnv`(reset/step이 obs, state, avail_actions를 돌려주는 SMAC 스타일)를 구현하고 `BUILTIN_ENVS`에 넣으면 됩니다.

## 오프라인 데이터셋

`bc`, `td3_bc`, `cql`, `iql_offline`, `awac`, `dt`, `gail`은 설정의 `dataset` 키로 데이터를 받습니다.

```bash
python collect.py --env CartPole-v1 --steps 20000                   # 무작위 정책 -> datasets/CartPole-v1_random.npz
python collect.py --run runs/ppo_CartPole-v1_s0_... --steps 50000   # 학습된 정책 -> datasets/CartPole-v1_<run>.npz
python collect.py --run runs/sac_Pendulum-v1_... --noise 0.3 --out datasets/pendulum_medium.npz
python train.py --algo dt --set dataset=datasets/CartPole-v1_ppo_....npz
python train.py --algo gail --set dataset=runs/ppo_CartPole-v1_s0_...   # run 디렉터리를 주면 알아서 수집
```

`dataset: auto`(기본값)는 파일이 없으면 무작위 정책으로 `dataset_steps`만큼 모읍니다. 동작 확인용이고, 제대로 된 모방 학습을 하려면 학습된 run을 주세요.

## 멀티 GPU

`rlzoo/common/hwaccel.py`가 디바이스, DP, DDP를 담당합니다. CV 쪽 저장소([Mutli-Tool4Vision](https://github.com/MZUI3/Mutli-Tool4Vision))의 `utils/hwaccel.py`와 같은 파일이고 옵션도 같습니다. YAML의 `accel:` 섹션이나 커맨드라인 플래그로 설정하고, 플래그가 우선합니다.

```yaml
accel:
  parallel: none      # none | dp | ddp
  device: auto        # auto | cpu | cuda | cuda:N
  gpu_ids: null       # dp 전용, 예: [0, 1]
  sync_bn: false
  backend: null       # ddp: nccl (CUDA) / gloo (CPU)
  master_addr: 127.0.0.1
  master_port: 29500
```

```bash
python train.py --algo ppo --device cuda:0
python train.py --algo sac --parallel dp --gpu-ids 0,1
torchrun --nproc_per_node=4 train.py --algo redq --set env.num_envs=4
python launch_ddp.py --nproc 4 train.py --algo redq --set env.num_envs=4    # torchrun 대신
python launch_ddp.py --nproc 2 train.py --algo ppo --device cpu             # CPU + gloo 로 확인
```

- RL 코드는 네트워크를 `forward`가 아니라 `dist()`, `q_values()` 같은 메서드로 부르기 때문에 `DistributedDataParallel` 래퍼의 훅이 안 걸립니다. 그래서 DDP는 rank마다 자기 환경을 돌리고 optimizer step 직전에 그래디언트를 직접 평균하는 식으로 했습니다. `total_steps`, `log_interval` 같은 값은 모든 rank의 환경 스텝 합계 기준입니다.
- rank 0만 로그, 평가, 저장을 합니다. 학습이 끝나면 rank 간 파라미터가 같은지 검사해서 `DDP parameter sync check: OK`를 찍습니다.
- 환경 시드는 rank마다 다르고(`seed + 7919 * rank`), 네트워크 초기화는 같은 시드로 한 뒤 rank 0에서 브로드캐스트합니다.
- DP는 forward가 있는 서브네트워크를 전부 `DataParallel`로 감쌉니다. GPU가 2장 미만이면 `none`으로 돌아갑니다.
- `a3c`는 그 자체가 비동기라 ddp는 안 됩니다.
- Windows용 PyTorch 휠은 libuv 없이 빌드되어 torchrun이 실패하는 경우가 있습니다. 그래서 `launch_ddp.py`를 따로 두었습니다.

## 설정 파일

```yaml
algo: qmix            # 알고리즘 이름
seed: 0
device: auto          # accel.device 와 같음
accel:                # 선택 (위 참고)
  parallel: none
dataset: auto         # 오프라인/모방 알고리즘만
env:
  id: spread          # gymnasium id 또는 MARL env id
  kwargs: {n_agents: 3}
  num_envs: 1         # 동기 병렬 환경 수
train:
  total_steps: 200000
  log_interval: 5000
  eval_interval: 25000  # 0이면 주기적 평가 안 함
  eval_episodes: 10
  save_dir: runs
  tensorboard: false
hparams:              # 안 적으면 알고리즘 기본값
  lr: 5.0e-4
```

- `hparams`는 클래스마다 `default_hparams`가 있고 부모 클래스 것과 합쳐집니다. 없는 키를 쓰면 가능한 키 목록과 함께 에러가 납니다.
- 공통 키: `gamma`, `hidden_sizes`, `activation`, `encoder`. MARL 공통: `agent_id`. 값 기반 MARL: `reward_scale`. PPO 계열 MARL과 MAT: `value_norm`.

## 학습 결과 (CPU, seed 0, 기본 설정)

기본 설정이 실제로 학습이 되는지 확인하려고 돌린 결과입니다. 시드 하나라 분산이 크니 참고만 하세요. 나중에 추가한 30개는 스모크 테스트만 했고 아직 이 표에 없습니다.

| 환경 (무작위 정책) | 결과 |
|---|---|
| CartPole-v1 (≈20, 최대 500) | dueling_dqn, rainbow, reinforce, a2c, a3c, trpo 500 / qr_dqn 266, ppo 250, c51 227, dqn 224, double_dqn 193, per_dqn 142 |
| Pendulum-v1 (≈−1200) | ddpg −169, td3 −170, sac −170 |
| spread, 3 agents (≈−57) | matd3 −27, happo −32, g2anet −33, mappo −33, commnet −33, vdn −33, qplex −33, ippo −34, qmix −36 (3시드 −31~−36), wqmix −37, mat −38, maddpg −39, qtran −41, iql −43, coma −47 |
| matrix_game `qtran` (최적 8) | qtran 3/3, wqmix 3/3, qplex 2/3 시드가 최적 도달. iql, vdn, qmix는 0에 수렴 (논문과 같은 경향) |

- 값 기반 MARL과 MADDPG/MATD3는 60k~100k 스텝, 온폴리시 계열은 300k 스텝입니다.
- COMA는 spread 같은 dense reward 과제에서 약한 편이라(EPyMARL 벤치마크와 같은 경향) `critic_epochs: 10`, `normalize_advantage: true`를 기본으로 뒀습니다.
- QMIX는 환경 1스텝당 업데이트 1회로 두면 spread에서 발산해서 pymarl처럼 `batch_size: 512`, `train_freq: 25`로 뒀습니다.
- HER + Double DQN은 BitFlip(8비트) 6k 스텝에서 성공률 1.0이 나옵니다.

## 코드 구조

```
train.py / evaluate.py / watch.py / collect.py / plot.py / benchmark.py / launch_ddp.py
configs/rl/*.yaml   configs/marl/*.yaml      알고리즘별 기본 설정
rlzoo/
  registry.py        이름 -> 클래스
  runner.py          설정 -> 환경/알고리즘 생성, 학습, 평가, 저장
  config.py          YAML 로드, --set 덮어쓰기
  common/
    hwaccel.py       디바이스 / DP / DDP (CV 저장소와 동일 파일)
    base.py          Algorithm 베이스 (학습 루프 공통, DDP 그래디언트 동기화, explain())
    nets.py          MLP/CNN 인코더, Q / IQN / recurrent 네트워크, actor-critic, critic 변형
    datasets.py      오프라인 데이터셋
    buffers.py, utils.py, logger.py
  envs/              gymnasium VecEnv, builtin.py, MARL 환경과 어댑터
  rl/                dqn, actor_critic, trpo, ppo, continuous, recurrent, sac_discrete, ppg, impala,
                     curiosity, offline, imitation, mbpo
  marl/              value_based, mixers, coma, maddpg, ppo_family, mat, masac, networks
  viz/               renderers.py (프레임 소스, 상태 그리기), player.py (뷰어, GIF/MP4/HTML)
tests/test_smoke.py
```

새 알고리즘은 `OffPolicyAlgo` / `OnPolicyAlgo` / `MAOffPolicyAlgo` / `MAOnPolicyAlgo` 중 하나를 상속해서 `act`와 `update`(on-policy MARL은 `policy_step`, `bootstrap_value`도)를 구현하고, `default_hparams`를 선언한 다음 `rlzoo/registry.py`에 한 줄 추가하고 `configs/`에 yaml을 만들면 됩니다.

## 원 논문과 다른 점

읽기 쉬운 쪽을 우선해서 단순화한 부분들입니다.

- MARL 에이전트 네트워크는 GRU 대신 MLP + 에이전트 ID이고, 리플레이 버퍼는 에피소드가 아니라 전이 단위입니다.
- QTRAN의 joint Q는 에이전트 은닉 상태 대신 전역 상태 + joint action을 받습니다.
- G2ANet의 hard attention은 Bi-LSTM 대신 쌍별 MLP 게이트(straight-through)이고, CommNet/G2ANet은 MAPPO로 학습합니다.
- MADDPG/MATD3/FACMAC는 이산 행동용 Gumbel-Softmax를 쓰고 critic 파라미터를 에이전트끼리 공유합니다.
- A3C는 Python 스레드라 GIL 영향을 받습니다.
- RND/ICM은 외재/내재 보상을 하나의 가치 헤드로 학습합니다. IMPALA는 비동기 액터 대신 `policy_lag`만큼 뒤처진 행동 정책을 쓰는 동기 버전입니다. MBPO는 환경 종료 함수 없이 모델 롤아웃을 비종료로 둡니다.
- 기본 설정은 CPU에서 몇 분 안에 끝나게 작게 잡았습니다. Atari나 MuJoCo, SMAC을 돌리려면 `total_steps`, 네트워크 크기, `num_envs`를 키우고 이미지 입력이면 `hparams.encoder`를 지정하세요.

# rlzoo — RL 15종 + MARL 15종을 YAML 설정으로 구성하는 PyTorch 프로젝트

대표적인 단일 에이전트 강화학습(RL) 알고리즘 15개와 멀티에이전트 강화학습(MARL) 알고리즘 15개, 총 30개를
**PyTorch로 직접 구현**했습니다. 외부 RL 라이브러리(SB3, RLlib, PyMARL 등)에 의존하지 않습니다.
알고리즘·환경·하이퍼파라미터는 모두 YAML 설정 파일 또는 커맨드라인 `--set`으로 바꿀 수 있습니다.

- 필수 의존성: `torch`, `numpy`, `gymnasium`, `pyyaml`
- MARL 기본 환경(Spread, Predator-Prey, Matrix game)은 numpy로 내장되어 있어 추가 설치가 필요 없습니다.
- 선택: PettingZoo(MPE 등), SMAC(StarCraft II) 어댑터 포함

## 설치

```bash
pip install -r requirements.txt
```

또는 `bash setup.sh` (conda가 있으면 conda 환경 `rlzoo`, 없으면 `.venv`를 만듭니다. `--extras`로 PettingZoo/Box2D 추가 설치).

## 빠른 시작

```bash
python train.py --list                          # 30개 알고리즘 목록
python train.py --algo ppo                      # configs/rl/ppo.yaml 로 학습
python train.py --config configs/marl/qmix.yaml # 설정 파일 직접 지정
```

설정 덮어쓰기:

```bash
python train.py --algo sac --steps 20000 --seed 3 --set hparams.actor_lr=1e-3 hparams.hidden_sizes=[128,128]
python train.py --algo mappo --set env.kwargs.n_agents=4 env.num_envs=16
python train.py --algo qmix --env predator_prey --set env.kwargs.grid_size=9 env.kwargs.n_prey=3
python train.py --algo rainbow --set hparams.noisy=false hparams.n_step=1   # Rainbow 구성요소 ablation
```

학습 결과는 `runs/<algo>_<env>_s<seed>_<시각>/`에 저장됩니다.

| 파일 | 내용 |
|---|---|
| `config.yaml` | 실제 사용된 전체 설정(기본 하이퍼파라미터 포함, 그대로 재현 가능) |
| `progress.jsonl` | 학습 지표 (`rollout/return`, `eval/return`, `train/*`, `time/fps`) |
| `model.pt` | 학습된 네트워크 가중치 |
| `summary.json` | 최종 평가 결과 |

```bash
python evaluate.py runs/<run_dir> --episodes 20          # 저장된 모델 평가
python evaluate.py runs/<run_dir> --render               # gymnasium 렌더링(단일 에이전트)
python plot.py runs/dqn_* runs/rainbow_* --metric eval/return --out compare.png
python tests/test_smoke.py                               # 30개 전부 짧게 실행해 동작 확인
python benchmark.py --kind rl                            # 여러 알고리즘 연속 실행 + 결과 표
python benchmark.py --algos qmix vdn qplex --seeds 0 1 2 --env matrix_game --steps 10000
```

## 알고리즘

### 단일 에이전트 RL (15)

| 이름 | 계열 | 기본 환경 | 논문 |
|---|---|---|---|
| `dqn` | Value-based | CartPole-v1 | Mnih et al., 2015 |
| `double_dqn` | Value-based | CartPole-v1 | van Hasselt et al., 2016 |
| `dueling_dqn` | Value-based (+Double) | CartPole-v1 | Wang et al., 2016 |
| `per_dqn` | Value-based (+Double, PER) | CartPole-v1 | Schaul et al., 2016 |
| `c51` | Distributional | CartPole-v1 | Bellemare et al., 2017 |
| `qr_dqn` | Distributional | CartPole-v1 | Dabney et al., 2018 |
| `rainbow` | Distributional (C51+Double+Dueling+Noisy+PER+n-step) | CartPole-v1 | Hessel et al., 2018 |
| `reinforce` | Policy gradient (+baseline) | CartPole-v1 | Williams, 1992 |
| `a2c` | Actor-critic | CartPole-v1 | Mnih et al., 2016 |
| `a3c` | Actor-critic (비동기, 스레드) | CartPole-v1 | Mnih et al., 2016 |
| `trpo` | Trust region | CartPole-v1 | Schulman et al., 2015 |
| `ppo` | Trust region (clip) | CartPole-v1 | Schulman et al., 2017 |
| `ddpg` | 연속 제어 | Pendulum-v1 | Lillicrap et al., 2016 |
| `td3` | 연속 제어 | Pendulum-v1 | Fujimoto et al., 2018 |
| `sac` | 연속 제어 (최대 엔트로피) | Pendulum-v1 | Haarnoja et al., 2018 |

- DQN 계열(`dqn`~`rainbow`)은 하나의 구현을 공유하므로 `double`, `dueling`, `noisy`, `prioritized`, `n_step`을 어떤 조합으로든 켜고 끌 수 있습니다.
- `reinforce`, `a2c`, `a3c`, `trpo`, `ppo`는 이산·연속 행동 공간 모두 지원합니다(`--env Pendulum-v1` 등).
- `ddpg`, `td3`, `sac`는 연속 행동(Box) 전용, DQN 계열은 이산 행동 전용입니다.

### 멀티에이전트 RL (15)

| 이름 | 계열 | 핵심 아이디어 | 논문 |
|---|---|---|---|
| `iql` | Independent | 에이전트별 독립 Q-learning (파라미터 공유) | Tan, 1993 |
| `vdn` | Value decomposition | Q_tot = Σ Q_i | Sunehag et al., 2018 |
| `qmix` | Value decomposition | 상태 조건 하이퍼네트워크, 단조(monotonic) 믹싱 | Rashid et al., 2018 |
| `qtran` | Value decomposition | Joint Q + 분해 제약(L_opt, L_nopt) | Son et al., 2019 |
| `qplex` | Value decomposition | Duplex dueling + 어텐션 가중치 | Wang et al., 2021 |
| `wqmix` | Value decomposition | 비제약 중앙 Q* + 가중 QMIX 손실 (OW/CW) | Rashid et al., 2020 |
| `coma` | Centralized critic | 반사실적(counterfactual) 베이스라인 | Foerster et al., 2018 |
| `maddpg` | Centralized critic | 중앙 critic + Gumbel-Softmax 액터 | Lowe et al., 2017 |
| `matd3` | Centralized critic | MADDPG + twin critic, delayed update, smoothing | Ackermann et al., 2019 |
| `ippo` | Policy gradient | 독립 PPO (로컬 관측 critic) | de Witt et al., 2020 |
| `mappo` | Policy gradient | PPO + 전역 상태 critic | Yu et al., 2022 |
| `happo` | Policy gradient | 에이전트별 정책, 순차 업데이트 | Kuba et al., 2022 |
| `mat` | Transformer | 인코더-디코더, 자기회귀적 행동 생성 | Wen et al., 2022 |
| `commnet` | Communication | 은닉 상태 평균 통신 (MAPPO로 학습) | Sukhbaatar et al., 2016 |
| `g2anet` | Communication | Hard + soft 2단계 어텐션 (MAPPO로 학습) | Liu et al., 2020 |

모든 MARL 알고리즘은 **협력(team reward) + 이산 행동 + CTDE** 설정을 가정합니다.

## MARL 환경

| `env.id` | 설명 | 주요 `env.kwargs` |
|---|---|---|
| `spread` | MPE simple_spread 유사. N 에이전트가 N 랜드마크를 덮고 충돌 회피 | `n_agents`, `n_landmarks`, `episode_limit`, `world_bound` |
| `predator_prey` | 격자 추격. 2명 이상이 인접해야 포획(협조 필요), 부분 관측 | `grid_size`, `n_agents`, `n_prey`, `sight`, `agents_to_capture` |
| `matrix_game` | 1-step 행렬 게임. 값 분해 방법의 표현력 비교용 | `payoff`: `qtran` / `climbing` / `penalty` |
| `pz:<모듈>` | PettingZoo parallel env (예: `pz:mpe2.simple_spread_v3`) | 해당 env의 인자 |
| `smac:<맵>` | SMAC (예: `smac:3m`, StarCraft II 필요) | StarCraft2Env 인자 |

```bash
# VDN/QMIX는 비단조 보상을 표현하지 못하고, QTRAN/QPLEX/WQMIX는 최적해(8)를 찾을 수 있는 고전 예제
python train.py --algo qplex --env matrix_game --steps 10000
python train.py --algo mappo --env pz:mpe2.simple_spread_v3 --set env.kwargs.max_cycles=25
```

새 환경은 `rlzoo/envs/marl/base.py`의 `MultiAgentEnv` 인터페이스(`reset`, `step`이 obs/state/avail_actions를 반환하는 SMAC 스타일)를 구현하고 `BUILTIN_ENVS`에 등록하면 됩니다.

## 설정 파일 구조

```yaml
algo: qmix            # 알고리즘 이름 (python train.py --list)
seed: 0
device: auto          # auto | cpu | cuda
env:
  id: spread          # gymnasium id 또는 MARL env id
  kwargs: {n_agents: 3}
  num_envs: 1         # 병렬(동기) 환경 수
train:
  total_steps: 200000 # 환경 스텝 수
  log_interval: 5000
  eval_interval: 25000  # 0이면 주기적 평가 끔
  eval_episodes: 10
  save_dir: runs
  tensorboard: false
hparams:              # 알고리즘 하이퍼파라미터 (생략 시 기본값)
  lr: 5.0e-4
  ...
```

- `hparams`에 적지 않은 값은 각 알고리즘 클래스의 `default_hparams`가 사용되며, 실제 사용된 전체 값은 `runs/.../config.yaml`에 기록됩니다.
- 존재하지 않는 하이퍼파라미터 이름을 쓰면 사용 가능한 키 목록과 함께 오류가 납니다(오타 방지).
- 공통 키: `gamma`, `hidden_sizes`, `activation`. MARL 공통: `agent_id`(관측에 에이전트 one-hot 추가).
- 값 기반 MARL: `reward_scale`. PPO 계열 MARL·MAT: `value_norm`(리턴 running 정규화, MAPPO/MAT 논문 방식).

## 검증 결과 (CPU, seed 0, 기본 설정)

학습이 실제로 되는지 확인하기 위해 기본 설정으로 돌린 결과입니다(최종 결정적 정책 평가). 시드 하나라 분산이 크므로 참고용입니다.

| 환경 (무작위 정책) | 알고리즘 → 최종 평가 리턴 |
|---|---|
| CartPole-v1 (≈20, 최대 500) | dueling_dqn·rainbow·reinforce·a2c·a3c·trpo 500 / qr_dqn 266 · ppo 250 · c51 227 · dqn 224 · double_dqn 193 · per_dqn 142 |
| Pendulum-v1 (≈−1200) | ddpg −169 · td3 −170 · sac −170 |
| spread, 3 agents (≈−57) | matd3 −27 · happo −32 · g2anet −33 · mappo −33 · commnet −33 · vdn −33 · qplex −33 · ippo −34 · qmix −36 (3시드 −31~−36) · wqmix −37 · mat −38 · maddpg −39 · qtran −41 · iql −43 · coma −47 |
| matrix_game `qtran` (최적 8) | qtran 3/3 · wqmix 3/3 · qplex 2/3 시드가 최적 도달, iql·vdn·qmix는 0에 수렴 (논문과 동일한 경향) |

- 값 기반 MARL·MADDPG·MATD3의 spread 결과는 60k~100k 스텝, 온폴리시 계열은 300k 스텝 기준입니다.
- COMA는 spread 같은 dense-reward 과제에서 약한 것으로 알려져 있어(EPyMARL 벤치마크와 같은 경향), 기본 설정을 `critic_epochs: 10`, `normalize_advantage: true`로 두었습니다.
- QMIX는 높은 replay ratio(환경 1스텝당 업데이트 1회)에서 spread에서 발산해, 기본 설정을 pymarl처럼 `batch_size: 512`, `train_freq: 25`(에피소드당 약 1회 업데이트)로 두었습니다.

## 프로젝트 구조

```
train.py / evaluate.py / plot.py
configs/rl/*.yaml   configs/marl/*.yaml      # 알고리즘별 기본 설정 30개
rlzoo/
  registry.py        # 알고리즘 이름 → 클래스
  runner.py          # 설정 → 환경/알고리즘 생성, 학습, 평가, 저장
  config.py          # YAML 로드, --set 덮어쓰기
  common/            # Algorithm 베이스, 네트워크, 리플레이 버퍼(PER/n-step), GAE, 로거
  envs/              # gymnasium VecEnv, MARL 환경 + PettingZoo/SMAC 어댑터
  rl/                # dqn.py, actor_critic.py, trpo.py, ppo.py, continuous.py
  marl/              # value_based.py, mixers.py, coma.py, maddpg.py, ppo_family.py, mat.py, networks.py
tests/test_smoke.py
```

## 새 알고리즘 추가하기

1. `rlzoo/rl/` 또는 `rlzoo/marl/`에 클래스를 만듭니다. `OffPolicyAlgo` / `OnPolicyAlgo` / `MAOffPolicyAlgo` / `MAOnPolicyAlgo`를 상속하면 학습 루프가 제공되므로 `act`, `update`(on-policy MARL은 `policy_step`, `bootstrap_value`도)만 구현하면 됩니다.
2. `default_hparams`에 하이퍼파라미터 기본값을 선언합니다(부모 클래스 기본값과 자동 병합).
3. `rlzoo/registry.py`의 `_ENTRIES`에 한 줄 추가하고, `configs/<rl|marl>/<이름>.yaml`을 만듭니다.

## 구현상 단순화

교육·실험용으로 읽기 쉬운 구현을 목표로 했으며, 논문 원본과 다른 점은 다음과 같습니다.

- MARL 에이전트 네트워크는 RNN(GRU) 대신 MLP + 에이전트 ID를 사용하고, 리플레이 버퍼는 에피소드가 아닌 전이(transition) 단위입니다.
- QTRAN의 joint Q는 에이전트 은닉 상태 대신 전역 상태 + joint action을 입력으로 받습니다.
- G2ANet의 hard attention은 Bi-LSTM 대신 쌍(pair)별 MLP 게이트(straight-through)이며, CommNet/G2ANet은 MAPPO 방식으로 학습합니다.
- MADDPG/MATD3는 이산 행동을 위해 Gumbel-Softmax를 쓰고, 협력 설정이므로 critic 파라미터를 에이전트 간 공유합니다(에이전트 ID 입력).
- A3C는 Python 스레드로 구현되어 GIL의 영향을 받습니다(알고리즘 구조는 동일).
- 기본 설정은 CPU에서 수 분 내에 돌아가도록 작게 잡았습니다. Atari·MuJoCo·SMAC 같은 큰 벤치마크에는 `total_steps`, 네트워크 크기, 병렬 환경 수를 늘려야 합니다(Atari 픽셀 입력용 CNN 인코더는 포함되어 있지 않습니다).

# rlzoo — RL 40종 + MARL 20종을 YAML 설정으로 구성하는 PyTorch 프로젝트

단일 에이전트 강화학습(RL) 알고리즘 40개(온라인·오프라인·모방·모델 기반·탐험·순환·목표 조건)와 멀티에이전트 강화학습(MARL)
알고리즘 20개, 총 60개를 **PyTorch로 직접 구현**했습니다. 외부 RL 라이브러리(SB3, RLlib, PyMARL 등)에 의존하지 않습니다.
알고리즘·환경·하이퍼파라미터는 모두 YAML 설정 파일 또는 커맨드라인 `--set`으로 바꿀 수 있습니다.

- 필수 의존성: `torch`, `numpy`, `gymnasium`, `pyyaml`, `matplotlib`, `pillow` (시각화)
- 학습이 끝난 뒤 `--watch` 한 줄로 시뮬레이션을 직접 볼 수 있습니다 (아래 "시뮬레이션 시각화").
- CV 허브(Mutli-Tool4Vision)와 **같은 하드웨어 가속 모듈**(`rlzoo/common/hwaccel.py`)을 써서 `--device`, `--parallel dp|ddp` 옵션이 동일합니다.
- MARL 기본 환경(Spread, Predator-Prey, Matrix game)은 numpy로 내장되어 있어 추가 설치가 필요 없습니다.
- 선택: PettingZoo(MPE 등), SMAC(StarCraft II) 어댑터 포함

## 설치

```bash
pip install -r requirements.txt
```

또는 `bash setup.sh` (conda가 있으면 conda 환경 `rlzoo`, 없으면 `.venv`를 만듭니다. `--extras`로 PettingZoo/Box2D 추가 설치).

## 빠른 시작

```bash
python train.py --list                          # 60개 알고리즘 목록
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
python watch.py runs/<run_dir>                           # 시뮬레이션 뷰어 (아래 참고)
python plot.py runs/dqn_* runs/rainbow_* --metric eval/return --out compare.png
python tests/test_smoke.py                               # 60개 전부 짧게 실행해 동작 확인
python benchmark.py --kind rl                            # 여러 알고리즘 연속 실행 + 결과 표
python benchmark.py --algos qmix vdn qplex --seeds 0 1 2 --env matrix_game --steps 10000
```

## 시뮬레이션 시각화 (학습 후 직접 보기)

학습이 끝난 모델이 환경에서 어떻게 움직이는지 창을 띄워 보거나, GIF / MP4 / HTML로 저장할 수 있습니다.

```bash
python train.py --algo ppo --watch                       # 학습이 끝나면 바로 뷰어 열기
python train.py --algo qmix --watch-save qmix.html       # 학습 후 에피소드를 HTML 플레이어로 저장
python watch.py runs/<run_dir>                           # 저장된 run 열기 (라이브 창)
python watch.py runs/<run_dir> --save ep.gif --no-window --episodes 3
python watch.py runs/<run_dir> --save ep.html --stochastic --fps 20
python evaluate.py runs/<run_dir> --watch                # 평가 후 뷰어
```

- **라이브 창** (matplotlib): 왼쪽에 환경 화면, 오른쪽에 에피소드/스텝/행동/보상/리턴 HUD, 알고리즘 진단(DQN 계열은 Q값, 정책 계열은 행동 확률·가치, MARL은 에이전트×행동 Q/확률 히트맵), 에피소드 리턴 그래프.
  키: `space` 일시정지 · `n`/`→` 한 스텝 · `r` 리셋 · `d` 결정적/확률적 토글 · `+`/`-` 속도 · `s` 스크린샷 · `q` 종료
- **HTML 플레이어**: 파일 하나에 프레임과 진단 정보가 들어 있어 브라우저에서 재생·스크럽할 수 있습니다 (공유용).
- **GIF**(pillow) / **MP4**(ffmpeg 필요).
- 렌더링: 환경 자체의 `rgb_array` 렌더러를 쓰고(gymnasium classic control은 `pip install pygame` 필요), 없으면 rlzoo가 상태를 직접 그립니다
  (CartPole, MountainCar, Pendulum, Acrobot, BitFlip, PointReach, PixelGrid, Spread, Predator-Prey, Matrix game).
  새 환경은 `rlzoo/viz/renderers.py`의 `_BY_CLASS`에 그리기 함수를 추가하면 됩니다.

## 하드웨어 가속: DP / DDP (CV 허브와 공통)

`rlzoo/common/hwaccel.py`는 CV 허브의 `utils/hwaccel.py`와 **바이트 단위로 같은 파일**이며, 옵션과 동작이 동일합니다.
YAML의 `accel:` 섹션 또는 커맨드라인 플래그로 설정합니다 (플래그가 YAML보다 우선).

| 모드 | 실행 | 의미 |
|---|---|---|
| `none` | `python train.py --algo ppo --device cuda:0` | 단일 디바이스 (`--device auto`가 기본: CUDA 있으면 CUDA) |
| `dp` | `python train.py --algo sac --parallel dp --gpu-ids 0,1` | 한 프로세스, `nn.DataParallel`. forward가 있는 모든 서브네트워크를 자동으로 감쌉니다 (GPU 2장 미만이면 `none`으로 내려감) |
| `ddp` | `torchrun --nproc_per_node=N train.py --algo ppo` 또는 `python launch_ddp.py --nproc N train.py --algo ppo` | 프로세스(=rank)마다 **자기 환경**을 돌리고 매 업데이트마다 그래디언트를 평균합니다. `total_steps`·`log_interval` 등은 모든 rank의 환경 스텝 합계 기준 |

```bash
python launch_ddp.py --nproc 2 train.py --algo ppo --device cpu     # CPU + gloo 로도 동작 (테스트용)
python launch_ddp.py --nproc 4 train.py --algo redq --set env.num_envs=4
```

- DDP에서는 rank 0만 로그·평가·저장을 하고, 학습이 끝나면 모든 rank의 파라미터가 같은지 검사해 출력합니다 (`DDP parameter sync check: OK`).
- 환경 시드는 rank마다 다르게(`seed + 7919 * rank`), 네트워크 초기화는 같은 시드 + rank 0 브로드캐스트로 맞춥니다.
- `a3c`는 자체적으로 비동기이므로 `ddp`를 지원하지 않습니다 (`none`/`dp`).
- Windows용 PyTorch 휠은 libuv 없이 빌드되어 `torchrun`이 실패하는 경우가 있습니다. 그때는 `launch_ddp.py`를 쓰면 됩니다.

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

## 관측 인코더와 내장 환경

- `hparams.encoder: auto | mlp | nature_cnn | small_cnn | impala_cnn` — 이미지 관측(Box uint8 HxWxC)이면 `auto`가 CNN을 고릅니다. 모든 단일 에이전트 알고리즘에 적용되고 리플레이 버퍼는 uint8로 저장합니다.
- 내장 gymnasium 환경 (추가 설치 없음):

| `env.id` | 설명 | 용도 |
|---|---|---|
| `BitFlip-v0` | n비트 뒤집기, 목표 조건 (Dict 관측) | HER (`her_dqn`) |
| `PointReach-v0` | 2D 점 질량을 목표 위치로, 연속 행동 | HER (`her_td3`, `her_sac`) |
| `PixelGrid-v0` | RGB 이미지 관측의 격자 길찾기 | CNN 인코더, DRQN, PPO-RNN |

## 오프라인 데이터셋

오프라인 RL·모방학습(`bc`, `td3_bc`, `cql`, `iql_offline`, `awac`, `dt`, `gail`)은 설정의 `dataset` 키로 데이터를 지정합니다.

```bash
python collect.py --env CartPole-v1 --steps 20000                        # 무작위 정책 -> datasets/CartPole-v1_random.npz
python collect.py --run runs/ppo_CartPole-v1_s0_... --steps 50000        # 학습된 정책(전문가) -> datasets/CartPole-v1_<run>.npz
python collect.py --run runs/sac_Pendulum-v1_... --noise 0.3 --out datasets/pendulum_medium.npz
python train.py --algo dt --set dataset=datasets/CartPole-v1_ppo_CartPole-v1_s0_....npz
python train.py --algo gail --set dataset=runs/ppo_CartPole-v1_s0_...     # run 디렉터리를 주면 자동으로 수집
```

`dataset: auto`(기본)는 파일이 없으면 무작위 정책으로 `dataset_steps`만큼 수집합니다(동작 확인용). 실제 모방 학습에는 전문가 run을 지정하세요.

## 알고리즘

### 단일 에이전트 RL (40)

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
| `noisy_dqn` | Value-based (NoisyNet 탐험) | CartPole-v1 | Fortunato et al., 2018 |
| `m_dqn` | Value-based (Munchausen) | CartPole-v1 | Vieillard et al., 2020 |
| `iqn` | Distributional (implicit quantile) | CartPole-v1 | Dabney et al., 2018 |
| `drqn` | Recurrent (GRU Q-network, 에피소드 리플레이) | CartPole-v1 | Hausknecht & Stone, 2015 |
| `ppo_rnn` | Recurrent (GRU PPO) | CartPole-v1 | Schulman et al., 2017 |
| `sac_discrete` | Actor-critic (이산 SAC) | CartPole-v1 | Christodoulou, 2019 |
| `tqc` | 연속 제어 (truncated quantile critics) | Pendulum-v1 | Kuznetsov et al., 2020 |
| `redq` | 연속 제어 (critic 앙상블 10, UTD 20) | Pendulum-v1 | Chen et al., 2021 |
| `droq` | 연속 제어 (dropout + LayerNorm critic, UTD 20) | Pendulum-v1 | Hiraoka et al., 2022 |
| `crossq` | 연속 제어 (BatchRenorm, 타깃 네트워크 없음) | Pendulum-v1 | Bhatt et al., 2024 |
| `ppg` | Trust region (phasic policy gradient) | CartPole-v1 | Cobbe et al., 2020 |
| `impala` | Actor-critic (V-trace, 동기 버전) | CartPole-v1 | Espeholt et al., 2018 |
| `rnd` | 탐험 (PPO + random network distillation) | MountainCar-v0 | Burda et al., 2019 |
| `icm` | 탐험 (PPO + intrinsic curiosity) | MountainCar-v0 | Pathak et al., 2017 |
| `her_dqn` | 목표 조건 (Double DQN + HER) | BitFlip-v0 | Andrychowicz et al., 2017 |
| `her_td3` / `her_sac` | 목표 조건 (TD3 / SAC + HER) | PointReach-v0 | Andrychowicz et al., 2017 |
| `mbpo` | 모델 기반 (앙상블 다이내믹스 + SAC) | Pendulum-v1 | Janner et al., 2019 |
| `bc` | 오프라인 / 모방 (behaviour cloning) | CartPole-v1 | Pomerleau, 1991 |
| `td3_bc` | 오프라인 (TD3 + BC 정규화) | Pendulum-v1 | Fujimoto & Gu, 2021 |
| `cql` | 오프라인 (conservative Q; 이산·연속) | CartPole-v1 | Kumar et al., 2020 |
| `iql_offline` | 오프라인 (implicit Q-learning) | CartPole-v1 | Kostrikov et al., 2022 |
| `awac` | 오프라인 (advantage-weighted actor-critic) | CartPole-v1 | Nair et al., 2020 |
| `dt` | 오프라인 (Decision Transformer) | CartPole-v1 | Chen et al., 2021 |
| `gail` | 모방 (PPO + discriminator 보상) | CartPole-v1 | Ho & Ermon, 2016 |

- DQN 계열(`dqn`~`rainbow`)은 하나의 구현을 공유하므로 `double`, `dueling`, `noisy`, `prioritized`, `n_step`을 어떤 조합으로든 켜고 끌 수 있습니다.
- `reinforce`, `a2c`, `a3c`, `trpo`, `ppo`는 이산·연속 행동 공간 모두 지원합니다(`--env Pendulum-v1` 등).
- `ddpg`, `td3`, `sac`, `tqc`, `redq`, `droq`, `crossq`, `mbpo`, `td3_bc`는 연속 행동(Box) 전용, DQN 계열·`drqn`·`sac_discrete`는 이산 행동 전용입니다.
- `her: true` 하이퍼파라미터는 모든 off-policy 알고리즘에서 쓸 수 있습니다 (목표 조건 Dict 관측 환경 필요). `her_dqn`/`her_td3`/`her_sac`는 이를 켠 설정입니다.
- `redq`, `droq`, `crossq`는 `sac`의 하이퍼파라미터 변형(`target_subset`, `actor_q`, `policy_delay`, `critic_dropout`, `critic_layer_norm`)으로 구현되어 있어 자유롭게 섞을 수 있습니다.
- 오프라인 알고리즘의 `train.total_steps`는 환경 스텝이 아니라 그래디언트 스텝 수입니다.

### 멀티에이전트 RL (20)

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
| `qatten` | Value decomposition | 멀티헤드 어텐션 가중 단조 믹싱 | Yang et al., 2020 |
| `maa2c` | Policy gradient | 중앙 critic A2C (클리핑 없음) | Papoudakis et al., 2021 |
| `ia2c` | Policy gradient | 독립 A2C (로컬 critic) | Papoudakis et al., 2021 |
| `masac` | Centralized critic | 이산 SAC + 에이전트별 중앙 critic | Haarnoja et al., 2018 (MA) |
| `facmac` | Centralized critic | 분해 critic + QMIX 믹서, 중앙 정책 그래디언트 | Peng et al., 2021 |

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
device: auto          # auto | cpu | cuda  (accel.device 의 축약)
accel:                # 하드웨어 가속 (선택, 위 "하드웨어 가속" 참고)
  parallel: none
dataset: auto         # 오프라인/모방 알고리즘만: auto | path.npz | runs/<run_dir>
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
train.py / evaluate.py / watch.py / collect.py / plot.py / benchmark.py
launch_ddp.py        # torchrun 대체 런처
configs/rl/*.yaml   configs/marl/*.yaml      # 알고리즘별 기본 설정 60개
rlzoo/
  registry.py        # 알고리즘 이름 → 클래스
  runner.py          # 설정 → 환경/알고리즘 생성, 학습, 평가, 저장 (Accelerator / DDP rank 처리)
  config.py          # YAML 로드, --set 덮어쓰기, accel 기본값
  common/
    hwaccel.py       # 하드웨어 가속 공통 모듈 (CV 허브와 동일 파일)
    base.py          # Algorithm 베이스 (DDP 그래디언트 동기화, explain() 진단 훅)
    nets.py          # MLP/CNN 인코더, Q/IQN/recurrent 네트워크, actor-critic, critic 변형(BatchRenorm, quantile, dropout)
    datasets.py      # 오프라인 데이터셋 수집/로드
    buffers.py, utils.py, logger.py
  envs/              # gymnasium VecEnv(이미지·목표 조건 관측), builtin.py(BitFlip/PointReach/PixelGrid), MARL 환경 + 어댑터
  rl/                # dqn.py, actor_critic.py, trpo.py, ppo.py, continuous.py, recurrent.py, sac_discrete.py, ppg.py,
                     # impala.py, curiosity.py, offline.py, imitation.py, mbpo.py
  marl/              # value_based.py, mixers.py, coma.py, maddpg.py, ppo_family.py, mat.py, masac.py, networks.py
  viz/               # renderers.py(프레임 소스 + 상태 그리기), player.py(라이브 뷰어, GIF/MP4/HTML 내보내기)
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
- RND/ICM은 외재·내재 보상을 하나의 가치 헤드로 학습하는 단순화 버전이고, IMPALA는 비동기 액터 대신 `policy_lag`만큼 뒤처진 행동 정책을 쓰는 동기 버전입니다. MBPO는 환경 종료 함수 대신 모델 롤아웃을 비종료로 취급합니다.
- 기본 설정은 CPU에서 수 분 내에 돌아가도록 작게 잡았습니다. Atari·MuJoCo·SMAC 같은 큰 벤치마크에는 `total_steps`, 네트워크 크기, 병렬 환경 수를 늘리고 `hparams.encoder`(nature_cnn/impala_cnn)를 지정하세요.

# RL + MARL 예제 프로젝트

이 저장소는 단일 에이전트 RL과 멀티에이전트(MARL)를 로컬에서 쉽게 실행할 수 있도록 예제 스크립트와 환경을 제공합니다.

요약:
- 단일 에이전트: `stable-baselines3` (PPO)
- 멀티 에이전트: `Ray RLlib` + `PettingZoo` (공유/독립 정책 구성 가능)
- Atari-57 등 Gymnasium Atari 환경 지원

환경 설정

1) Conda (권장):

```bash
conda env create -f environment.yml
conda activate rl-env
```

2) Virtualenv / pip:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

사용법

- 단일 에이전트 RL (예: CartPole):

```bash
python src/train_rl.py --env CartPole-v1 --algo ppo --total-timesteps 100000
```

- Atari 학습 (예: Pong):

```bash
python src/train_rl.py --env PongNoFrameskip-v4 --algo ppo --total-timesteps 500000 --atari
```

- 멀티에이전트 예제 (PettingZoo의 simple_spread):

```bash
python src/train_marl.py --env simple_spread_v2 --algo ppo --stop-timesteps 200000
```

파일 설명
- `environment.yml`: conda 환경 정의
- `requirements.txt`: pip 설치용 의존성
- `src/train_rl.py`: 단일 에이전트 학습 스크립트 (stable-baselines3)
- `src/train_marl.py`: MARL 예제 (RLlib + PettingZoo)
- `src/train_marl_smac.py`: SMAC(StarCraft Multi-Agent Challenge) 통합 스켈레톤
- `src/algos_registry.py`: 알고리즘 플러그인/레지스트리 (SB3 + 외부플러그인)
- `src/external_algos/`: 외부/고급 알고리즘 플러그인 위치(Decision Transformer, MuZero, Agent57 등)

지원 알고리즘(초기):
- 단일 에이전트(SB3 지원): `dqn`, `ppo`, `sac`, `td3`
- 추가 단일 에이전트(플러그인/외부 필요): `a3c`, `trpo`, `ddpg`, `rainbow`, `muzero`, `agent57`, `decision_transformer`
- 멀티에이전트(MARL): `maddpg`, `mappo`, `qmix`, `vdn`, `qplex`, `coma`, `mat`, `happo`, `g2anet`, `alphastar`

플러그인 방식 설명:
- `src/external_algos/<name>.py`에 `def train(env_id, **kwargs)`를 구현하면 `python src/train_rl.py --env <env> --algo <name>`로 호출됩니다.
- 기본적으로 많은 고급 알고리즘은 연구 레포/서드파티 구현을 필요로 하므로 `setup.sh --all`로 `third_party/`에 복제한 뒤 각 레포의 사용법에 맞게 래퍼를 구현하세요.

SMAC(StarCraft Multi-Agent Challenge) 사용 안내:
- SMAC은 StarCraft II가 필요합니다. 설치가 쉽지 않으므로 공식 SMAC / PyMARL 설치 지침을 따르세요.
- 간단한 실행 흐름은 `src/train_marl_smac.py`를 통해 스크립트 위치와 맵을 지정해 호출합니다.

통합 설치 스크립트
- `setup.sh` 하나로 Conda 환경 생성 또는 `venv` 설치를 선택하고, `--all` 옵션으로 MuZero/SMAC/기타 서드파티 레포를 `third_party/`에 복제합니다.

예:
```bash
# 기본: conda가 있으면 environment.yml로 env 생성
bash setup.sh

# venv 사용
bash setup.sh --venv

# 모든 서드파티 레포 복제 (MuZero, PyMARL/SMAC 등)
bash setup.sh --all
```

제약
- 일부 알고리즘(MuZero, Agent57, Agent57 등)은 외부 연구 코드/레포를 필요로 하며, 자동 통합이 어렵습니다. `third_party/`에 레포를 복제한 뒤 각 레포의 README에 따라 추가 설치 및 학습 스크립트를 실행하세요. 이 저장소는 플러그인 래퍼와 예시를 제공합니다.

플러그인 추가 예시
1) `src/external_algos/muzero.py`를 만들고 `def train(env_id, **kwargs):`를 구현
2) `python src/train_rl.py --env <env> --algo muzero`

제약 및 권장사항
- MuZero, Agent57, Decision Transformer 등은 외부 구현을 추가해야 합니다. 이 저장소는 플러그인 방식의 래퍼와 예시를 제공합니다.
- SMAC은 별도의 설치(StarCraft II + SMAC repo) 및 많은 디스크/메모리 리소스를 요구합니다.

노트
- Atari 사용 시 `gymnasium[accept-rom-license]` 설치 필요(라이선스 동의)
- GPU 사용 시 `torch`를 CUDA 빌드로 설치하세요 (conda 채널 권장)

문제가 있거나 추가 예제가 필요하면 알려주세요.

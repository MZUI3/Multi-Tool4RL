#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
ENV_FILE="$ROOT_DIR/environment.yml"
REQ_FILE="$ROOT_DIR/requirements.txt"
TP_DIR="$ROOT_DIR/third_party"

usage(){
  cat <<EOF
Usage: setup.sh [--conda|--venv] [--all]
  --conda    Create conda env from environment.yml (default if conda available)
  --venv     Use python -m venv and pip install -r requirements.txt
  --all      Also clone and prepare third-party repos for advanced algos and SMAC

After running, activate the environment and run examples in src/
EOF
}

MODE="conda"
CLONE_ALL=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv) MODE="venv"; shift ;;
    --conda) MODE="conda"; shift ;;
    --all) CLONE_ALL=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ "$MODE" == "conda" && $(command -v conda || true) ]]; then
  echo "Creating conda env from $ENV_FILE"
  conda env create -f "$ENV_FILE" || conda env update -f "$ENV_FILE"
  echo "Done. Activate with: conda activate rl-env"
else
  echo "Creating virtualenv and installing pip requirements"
  python -m venv "$ROOT_DIR/.venv"
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.venv/bin/activate"
  pip install -U pip
  pip install -r "$REQ_FILE"
  echo "Done. Activate with: source .venv/bin/activate (Windows: .venv\Scripts\activate)"
fi

if [[ "$CLONE_ALL" == "true" ]]; then
  mkdir -p "$TP_DIR"
  pushd "$TP_DIR" >/dev/null
  echo "Cloning third-party implementations (may take a while)"
  # MuZero (example implementation)
  if [[ ! -d muzero-general ]]; then
    git clone https://github.com/Auburns/FastMuZero muzero-general || true
  fi
  # PyMARL / SMAC
  if [[ ! -d pymarl ]]; then
    git clone https://github.com/oxwhirl/pymarl pymarl || true
  fi
  # Example QMIX implementations
  if [[ ! -d qmix ]]; then
    git clone https://github.com/oxwhirl/qmix qmix || true
  fi
  popd >/dev/null
  echo "Third-party repos cloned under $TP_DIR"
fi

echo "Setup complete. See README.md for next steps and limitations."

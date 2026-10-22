#!/usr/bin/env bash
# Create an environment for rlzoo.
#   bash setup.sh            # conda env "rlzoo" if conda exists, else .venv
#   bash setup.sh --venv     # force python venv
#   bash setup.sh --extras   # also install PettingZoo MPE + Box2D extras
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
MODE="conda"
EXTRAS=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv) MODE="venv"; shift ;;
    --conda) MODE="conda"; shift ;;
    --extras) EXTRAS=true; shift ;;
    -h|--help) sed -n '2,5p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ "$MODE" == "conda" ]] && command -v conda >/dev/null 2>&1; then
  conda env create -f "$ROOT_DIR/environment.yml" || conda env update -f "$ROOT_DIR/environment.yml"
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate rlzoo
  echo "Activate with: conda activate rlzoo"
else
  python -m venv "$ROOT_DIR/.venv"
  if [[ -f "$ROOT_DIR/.venv/Scripts/activate" ]]; then
    # shellcheck disable=SC1091
    source "$ROOT_DIR/.venv/Scripts/activate"
  else
    # shellcheck disable=SC1091
    source "$ROOT_DIR/.venv/bin/activate"
  fi
  pip install -U pip
  pip install -r "$ROOT_DIR/requirements.txt"
  echo "Activate with: source .venv/bin/activate  (Windows: .venv\Scripts\activate)"
fi

if [[ "$EXTRAS" == "true" ]]; then
  pip install "gymnasium[box2d]" pettingzoo mpe2 || echo "Some extras failed to install (optional)."
fi

python "$ROOT_DIR/train.py" --list

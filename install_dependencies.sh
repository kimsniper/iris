#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 -m venv "$PROJECT_ROOT/.venv"
source "$PROJECT_ROOT/.venv/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$PROJECT_ROOT/requirements.txt"
python -m pip install -e "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
python -c "import gymnasium, mujoco, numpy, osqp, stable_baselines3, yaml; print('Dependency import check passed')"
python "$PROJECT_ROOT/scripts/validate_model.py"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q "$PROJECT_ROOT/tests"

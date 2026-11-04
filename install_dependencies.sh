#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RIFT_ROOT="$(cd "$PROJECT_ROOT/../rift" && pwd)"

if [[ ! -f "$RIFT_ROOT/pyproject.toml" ]]; then
  echo "RIFT repository not found at: $RIFT_ROOT"
  exit 1
fi

python3 -m venv "$PROJECT_ROOT/.venv"
source "$PROJECT_ROOT/.venv/bin/activate"

python -m pip install --upgrade pip setuptools wheel

# Install the local RIFT source as the application's dependency.
python -m pip install -e "$RIFT_ROOT"

# Install IRIS and its remaining dependencies.
python -m pip install -e "$PROJECT_ROOT"

python - <<'PY'
import gymnasium
import mujoco
import numpy
import osqp
import rift
import stable_baselines3
import yaml

print("Dependency import check passed")
print(f"RIFT version: {rift.__version__}")
print(f"RIFT source: {rift.__file__}")
PY

python "$PROJECT_ROOT/scripts/validate_model.py"

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest -q "$RIFT_ROOT/tests"

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest -q "$PROJECT_ROOT/tests"

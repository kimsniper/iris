# IRIS

**Intelligent Robotics for Inertial Stability**

IRIS is a research platform for developing and evaluating classical,
model-based, and learning-based control for self-balancing robots.

The implementation uses MuJoCo, Gymnasium, and Stable-Baselines3 to
compare three controller approaches for a two-wheel self-balancing
robot:

1. PID with manually qualified gains
2. MPC using an identified linear dynamics model and manually qualified
   weights
3. Direct PPO motor control

PID and MPC run without learned-policy inference. The direct PPO
controller loads a qualified policy at runtime.

## Setup

```bash
./install_dependencies.sh
source activate.sh
```

## Validate

```bash
python scripts/validate_model.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests
```

## Visualize controllers

```bash
python scripts/view_pid.py
python scripts/view_mpc.py
python scripts/view_ppo.py \
  --model models/direct_control/best_model.zip
```

## Evaluate controllers

```bash
python scripts/evaluate_pid.py
python scripts/evaluate_mpc.py
python rl/direct_control/evaluate.py \
  --model models/direct_control/best_model.zip
```

PID results are written under `results/pid/`. MPC results are written
under `results/mpc/`.

## Train direct PPO

```bash
python rl/direct_control/train.py \
  --total-timesteps 200000 \
  --seed 0
```

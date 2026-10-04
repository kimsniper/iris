# IRIS

**Intelligent Robotics for Inertial Stability**

IRIS is a research platform for developing and evaluating classical, model-based, and learning-based control for self-balancing robots.

The implementation uses MuJoCo, Gymnasium, and Stable-Baselines3 to compare three controllers for a two-wheel self-balancing robot:

1. PID using gains selected by PPO
2. MPC using tuning values selected by PPO
3. Direct PPO motor control

PPO is used during PID and MPC tuning only. The exported PID gains and MPC parameters are consumed by the final classical controllers without PPO inference. The direct PPO controller loads a qualified policy at runtime.

## Setup

```bash
source activate.sh
./install_dependencies.sh
```

## Validate

```bash
python scripts/validate_model.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests
```

## Initial runs

```bash
python scripts/run_pid.py
python scripts/run_mpc.py
python rl/direct_control/train.py --total-timesteps 10000
python scripts/run_ppo.py --model models/direct_control/best_model.zip
python scripts/compare_controllers.py
```

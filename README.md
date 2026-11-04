# IRIS

**Intelligent Robotics for Inertial Stability**

IRIS is a research platform for developing and evaluating classical,
model-based, and learning-based control for self-balancing robots.

IRIS is built on [RIFT](https://github.com/kimsniper/rift), a reusable
robotics and Physical AI platform.

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

Clone IRIS:

```bash
git clone git@github.com:kimsniper/iris.git
cd iris
```

Create the project environment and install the dependencies:

```bash
./install_dependencies.sh
source activate.sh
```

The installer creates `.venv/` inside the IRIS repository and installs
RIFT and the other required packages into that environment.

## Validate

```bash
python scripts/validate_model.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests
```

## Visualize controllers

### PID

```bash
python scripts/view_pid.py
```

### MPC

```bash
python scripts/view_mpc.py
```

### PPO

```bash
python scripts/view_ppo.py \
  --model models/direct_control/best_model.zip
```

## Evaluate controllers

### PID

```bash
python scripts/evaluate_pid.py
```

PID results are written under `results/pid/`.

### MPC

```bash
python scripts/evaluate_mpc.py
```

MPC results are written under `results/mpc/`.

### PPO

```bash
python rl/direct_control/evaluate.py \
  --model models/direct_control/best_model.zip
```

## Train direct PPO

```bash
python rl/direct_control/train.py \
  --total-timesteps 200000 \
  --seed 0
```

Training outputs are written under `models/direct_control/` and
`logs/direct_control/`.

## MPC system identification

```bash
python scripts/identify_mpc_dynamics.py
python scripts/validate_mpc_multistep.py
```

## License

IRIS is licensed under the Apache License 2.0. See `LICENSE` for details.

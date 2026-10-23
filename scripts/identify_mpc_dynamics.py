# Copyright 2026 Mezael Docoy
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import numpy as np

from controllers.pid.pid_controller import PIDController
from rl.environments.balancing_env import BalancingEnv


OUTPUT_PATH = Path("controllers/mpc/identified_dynamics.npz")

TRAIN_SEEDS = list(range(5001, 5051))
VALIDATION_SEEDS = list(range(6001, 6011))

EXCITATION_LIMIT = 0.03
EXCITATION_SEED = 77


def extract_state(env, initial_position):
    return np.array(
        [
            env._truth_pitch(),
            float(env.data.qvel[4]),
            float(env.data.qpos[0]) - initial_position,
            float(env.data.qvel[0]),
        ],
        dtype=float,
    )


def make_controller():
    return PIDController(
        kp=10.0,
        ki=0.0,
        kd=0.2,
        position_kp=0.002,
        velocity_kd=0.002,
        max_pitch_reference=0.02,
        integral_limit=0.5,
        output_limit=0.5,
    )


def collect_transitions(seeds, rng):
    states = []
    inputs = []
    next_states = []

    for seed in seeds:
        env = BalancingEnv()
        observation, _ = env.reset(seed=seed)

        controller = make_controller()
        controller.reset()

        initial_position = float(env.data.qpos[0])
        terminated = False
        truncated = False

        while not (terminated or truncated):
            state = extract_state(
                env,
                initial_position,
            )

            nominal_command = float(
                controller.compute(
                    observation,
                    env.dt,
                )[0]
            )
            excitation = float(
                rng.uniform(
                    -EXCITATION_LIMIT,
                    EXCITATION_LIMIT,
                )
            )
            torque = float(
                np.clip(
                    nominal_command + excitation,
                    -controller.output_limit,
                    controller.output_limit,
                )
            )

            observation, _, terminated, truncated, _ = env.step(
                np.array([torque], dtype=np.float32)
            )

            next_state = extract_state(
                env,
                initial_position,
            )

            states.append(state)
            inputs.append(torque)
            next_states.append(next_state)

    return (
        np.asarray(states, dtype=float),
        np.asarray(inputs, dtype=float),
        np.asarray(next_states, dtype=float),
    )


def fit_dynamics(states, inputs, next_states):
    regressors = np.column_stack(
        [
            states,
            inputs,
        ]
    )

    coefficients, _, _, _ = np.linalg.lstsq(
        regressors,
        next_states,
        rcond=None,
    )

    a_matrix = coefficients[:4, :].T
    b_matrix = coefficients[4, :].reshape(4, 1)

    return a_matrix, b_matrix


def prediction_errors(
    a_matrix,
    b_matrix,
    states,
    inputs,
    next_states,
):
    predictions = (
        states @ a_matrix.T
        + inputs[:, None] @ b_matrix.T
    )
    errors = predictions - next_states

    return (
        np.mean(np.abs(errors), axis=0),
        np.sqrt(np.mean(np.square(errors), axis=0)),
        np.max(np.abs(errors), axis=0),
    )


def controllability_matrix(a_matrix, b_matrix):
    columns = []
    current = b_matrix.copy()

    for _ in range(a_matrix.shape[0]):
        columns.append(current)
        current = a_matrix @ current

    return np.hstack(columns)


def main():
    rng = np.random.default_rng(EXCITATION_SEED)

    train_states, train_inputs, train_next_states = (
        collect_transitions(TRAIN_SEEDS, rng)
    )
    validation_states, validation_inputs, validation_next_states = (
        collect_transitions(VALIDATION_SEEDS, rng)
    )

    a_matrix, b_matrix = fit_dynamics(
        train_states,
        train_inputs,
        train_next_states,
    )

    train_mae, train_rms, train_max = prediction_errors(
        a_matrix,
        b_matrix,
        train_states,
        train_inputs,
        train_next_states,
    )
    validation_mae, validation_rms, validation_max = (
        prediction_errors(
            a_matrix,
            b_matrix,
            validation_states,
            validation_inputs,
            validation_next_states,
        )
    )

    controllability = controllability_matrix(
        a_matrix,
        b_matrix,
    )
    controllability_rank = int(
        np.linalg.matrix_rank(controllability)
    )
    eigenvalues = np.linalg.eigvals(a_matrix)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    np.savez(
        OUTPUT_PATH,
        A=a_matrix,
        B=b_matrix,
        equilibrium_state=np.zeros(4, dtype=float),
        equilibrium_input=np.zeros(1, dtype=float),
        control_dt=np.array([0.02], dtype=float),
        state_order=np.array(
            [
                "pitch",
                "pitch_rate",
                "chassis_position",
                "chassis_velocity",
            ]
        ),
    )

    np.set_printoptions(
        precision=8,
        suppress=True,
    )

    print("IRIS trajectory-based MPC identification")
    print(f"Training transitions: {len(train_states)}")
    print(
        f"Validation transitions: "
        f"{len(validation_states)}"
    )

    print("\nA matrix")
    print(a_matrix)

    print("\nB matrix")
    print(b_matrix)

    print("\nOpen-loop eigenvalues")
    print(eigenvalues)

    print(
        "\nControllability rank: "
        f"{controllability_rank}/4"
    )

    names = [
        "pitch",
        "pitch_rate",
        "chassis_position",
        "chassis_velocity",
    ]

    print("\nTraining prediction errors")
    for index, name in enumerate(names):
        print(
            f"  {name}: "
            f"MAE={train_mae[index]:.8f}, "
            f"RMS={train_rms[index]:.8f}, "
            f"MAX={train_max[index]:.8f}"
        )

    print("\nHeld-out prediction errors")
    for index, name in enumerate(names):
        print(
            f"  {name}: "
            f"MAE={validation_mae[index]:.8f}, "
            f"RMS={validation_rms[index]:.8f}, "
            f"MAX={validation_max[index]:.8f}"
        )

    print(f"\nSaved model: {OUTPUT_PATH}")

    if controllability_rank != 4:
        raise SystemExit(
            "Identified model is not fully controllable."
        )

    if not np.isfinite(a_matrix).all():
        raise SystemExit("A matrix contains invalid values.")

    if not np.isfinite(b_matrix).all():
        raise SystemExit("B matrix contains invalid values.")


if __name__ == "__main__":
    main()

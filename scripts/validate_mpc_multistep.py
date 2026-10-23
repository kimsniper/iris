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

import numpy as np

from scripts.identify_mpc_dynamics import (
    EXCITATION_SEED,
    VALIDATION_SEEDS,
    collect_transitions,
)


DYNAMICS_PATH = "controllers/mpc/identified_dynamics.npz"
HORIZONS = [1, 5, 10, 20]
EPISODE_STEPS = 500


def main():
    identified = np.load(DYNAMICS_PATH)
    a_matrix = identified["A"]
    b_matrix = identified["B"]

    rng = np.random.default_rng(EXCITATION_SEED + 1)
    states, inputs, next_states = collect_transitions(
        VALIDATION_SEEDS,
        rng,
    )

    expected_transitions = (
        len(VALIDATION_SEEDS) * EPISODE_STEPS
    )

    if len(states) != expected_transitions:
        raise RuntimeError(
            f"Expected {expected_transitions} transitions, "
            f"received {len(states)}."
        )

    state_names = [
        "pitch",
        "pitch_rate",
        "chassis_position",
        "chassis_velocity",
    ]

    print("IRIS MPC multi-step prediction validation")
    print(f"Episodes: {len(VALIDATION_SEEDS)}")
    print(f"Steps per episode: {EPISODE_STEPS}")
    print(f"Transitions: {len(states)}")

    for horizon in HORIZONS:
        errors = []

        for episode_index in range(len(VALIDATION_SEEDS)):
            episode_start = episode_index * EPISODE_STEPS
            episode_end = episode_start + EPISODE_STEPS

            for start in range(
                episode_start,
                episode_end - horizon + 1,
            ):
                predicted = states[start].copy()

                for offset in range(horizon):
                    predicted = (
                        a_matrix @ predicted
                        + b_matrix[:, 0]
                        * inputs[start + offset]
                    )

                actual = next_states[
                    start + horizon - 1
                ]
                errors.append(predicted - actual)

        errors = np.asarray(errors, dtype=float)

        rms_error = np.sqrt(
            np.mean(np.square(errors), axis=0)
        )
        max_error = np.max(
            np.abs(errors),
            axis=0,
        )

        print(
            f"\nHorizon: {horizon} steps "
            f"({horizon * 0.02:.2f} s)"
        )

        for index, name in enumerate(state_names):
            print(
                f"  {name}: "
                f"RMS={rms_error[index]:.8f}, "
                f"MAX={max_error[index]:.8f}"
            )


if __name__ == "__main__":
    main()

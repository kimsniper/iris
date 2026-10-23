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


DEFAULT_DYNAMICS_PATH = (
    Path(__file__).with_name("identified_dynamics.npz")
)


def load_dynamics(path=DEFAULT_DYNAMICS_PATH):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Identified MPC dynamics not found: {path}"
        )

    with np.load(path) as dynamics:
        a_matrix = dynamics["A"].copy()
        b_matrix = dynamics["B"].copy()
        equilibrium_state = (
            dynamics["equilibrium_state"].copy()
        )
        equilibrium_input = float(
            dynamics["equilibrium_input"][0]
        )
        control_dt = float(
            dynamics["control_dt"][0]
        )
        state_order = tuple(
            str(name)
            for name in dynamics["state_order"]
        )

    if a_matrix.shape != (4, 4):
        raise ValueError(
            f"Expected A shape (4, 4), got {a_matrix.shape}."
        )

    if b_matrix.shape != (4, 1):
        raise ValueError(
            f"Expected B shape (4, 1), got {b_matrix.shape}."
        )

    if not np.isfinite(a_matrix).all():
        raise ValueError("A matrix contains invalid values.")

    if not np.isfinite(b_matrix).all():
        raise ValueError("B matrix contains invalid values.")

    return {
        "A": a_matrix,
        "B": b_matrix,
        "equilibrium_state": equilibrium_state,
        "equilibrium_input": equilibrium_input,
        "control_dt": control_dt,
        "state_order": state_order,
    }


def controllability_matrix(a_matrix, b_matrix):
    columns = []
    current = b_matrix.copy()

    for _ in range(a_matrix.shape[0]):
        columns.append(current)
        current = a_matrix @ current

    return np.hstack(columns)


def linear_model(dt=None):
    dynamics = load_dynamics()

    if dt is not None and not np.isclose(
        float(dt),
        dynamics["control_dt"],
    ):
        raise ValueError(
            "Controller timestep does not match identified dynamics: "
            f"{dt} != {dynamics['control_dt']}."
        )

    return dynamics["A"], dynamics["B"]

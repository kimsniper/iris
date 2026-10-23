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

from controllers.mpc.dynamics_model import (
    controllability_matrix,
    load_dynamics,
)


EXPECTED_STATE_ORDER = (
    "pitch",
    "pitch_rate",
    "chassis_position",
    "chassis_velocity",
)


def test_identified_dynamics_shapes():
    dynamics = load_dynamics()

    assert dynamics["A"].shape == (4, 4)
    assert dynamics["B"].shape == (4, 1)
    assert dynamics["equilibrium_state"].shape == (4,)


def test_identified_dynamics_are_finite():
    dynamics = load_dynamics()

    assert np.isfinite(dynamics["A"]).all()
    assert np.isfinite(dynamics["B"]).all()


def test_identified_dynamics_metadata():
    dynamics = load_dynamics()

    assert abs(dynamics["control_dt"] - 0.02) < 1e-12
    assert dynamics["state_order"] == EXPECTED_STATE_ORDER
    assert dynamics["equilibrium_input"] == 0.0


def test_identified_dynamics_are_controllable():
    dynamics = load_dynamics()

    controllability = controllability_matrix(
        dynamics["A"],
        dynamics["B"],
    )

    assert np.linalg.matrix_rank(controllability) == 4


def test_identified_dynamics_have_unstable_mode():
    dynamics = load_dynamics()
    eigenvalues = np.linalg.eigvals(dynamics["A"])

    assert np.any(np.abs(eigenvalues) > 1.0)


def test_positive_torque_has_expected_signs():
    dynamics = load_dynamics()
    input_response = dynamics["B"][:, 0]

    assert input_response[0] < 0.0
    assert input_response[1] < 0.0
    assert input_response[2] > 0.0
    assert input_response[3] > 0.0

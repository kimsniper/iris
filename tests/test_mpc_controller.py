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
import pytest

from controllers.mpc.mpc_controller import MPCController


def make_observation(
    pitch=0.0,
    pitch_rate=0.0,
    wheel_position=0.0,
    wheel_velocity=0.0,
):
    return np.array(
        [
            pitch,
            pitch_rate,
            wheel_velocity,
            wheel_velocity,
            0.0,
            9.81,
            0.0,
            wheel_position,
            wheel_position,
        ],
        dtype=np.float32,
    )


def test_mpc_returns_bounded_pair():
    controller = MPCController(
        horizon=10,
        output_limit=0.5,
    )

    command = controller.compute(
        make_observation(pitch=0.1),
        0.02,
    )

    assert command.shape == (2,)
    assert command[0] == command[1]
    assert np.max(np.abs(command)) <= 0.5


def test_mpc_corrects_positive_pitch():
    controller = MPCController(
        horizon=10,
        output_limit=0.5,
    )

    command = controller.compute(
        make_observation(pitch=0.05),
        0.02,
    )

    assert command[0] > 0.0


def test_mpc_reset_clears_position_target():
    controller = MPCController()

    controller.compute(
        make_observation(wheel_position=1.0),
        0.02,
    )

    assert controller.target_position is not None

    controller.reset()

    assert controller.target_position is None


def test_mpc_rejects_wrong_timestep():
    controller = MPCController()

    with pytest.raises(
        ValueError,
        match="does not match MPC timestep",
    ):
        controller.compute(
            make_observation(),
            0.01,
        )


def test_mpc_requires_full_observation():
    controller = MPCController()

    with pytest.raises(
        ValueError,
        match="nine-element IRIS observation",
    ):
        controller.compute(
            np.zeros(7, dtype=np.float32),
            0.02,
        )


@pytest.mark.parametrize(
    "observation",
    [
        np.zeros(8, dtype=np.float32),
        np.zeros(10, dtype=np.float32),
        np.zeros((9, 1), dtype=np.float32),
    ],
)
def test_mpc_rejects_wrong_observation_shape(observation):
    controller = MPCController()

    with pytest.raises(
        ValueError,
        match="nine-element IRIS observation",
    ):
        controller.compute(observation, 0.02)


@pytest.mark.parametrize(
    "invalid_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_mpc_rejects_invalid_observation_values(
    invalid_value,
):
    controller = MPCController()
    observation = make_observation()
    observation[0] = invalid_value

    with pytest.raises(
        ValueError,
        match="observation contains invalid values",
    ):
        controller.compute(observation, 0.02)


@pytest.mark.parametrize(
    "dt",
    [
        0.0,
        -0.02,
        np.nan,
        np.inf,
    ],
)
def test_mpc_rejects_invalid_runtime_dt(dt):
    controller = MPCController()

    with pytest.raises(
        ValueError,
        match="timestep must be finite and greater than zero",
    ):
        controller.compute(make_observation(), dt)


@pytest.mark.parametrize(
    "dt",
    [
        0.0,
        -0.02,
        np.nan,
        np.inf,
    ],
)
def test_mpc_rejects_invalid_constructor_dt(dt):
    with pytest.raises(
        ValueError,
        match="timestep must be finite and greater than zero",
    ):
        MPCController(dt=dt)


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("horizon", 0),
        ("horizon", -1),
        ("output_limit", 0.0),
        ("output_limit", -0.5),
        ("wheel_radius", 0.0),
        ("wheel_radius", -0.08),
        ("q_pitch", -1.0),
        ("q_pitch_rate", -1.0),
        ("q_position", -1.0),
        ("q_velocity", -1.0),
        ("r_input", 0.0),
        ("r_input", -0.1),
    ],
)
def test_mpc_rejects_invalid_parameters(parameter, value):
    arguments = {
        parameter: value,
    }

    with pytest.raises(ValueError):
        MPCController(**arguments)


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("q_pitch", np.nan),
        ("q_pitch_rate", np.inf),
        ("q_position", -np.inf),
        ("q_velocity", np.nan),
        ("r_input", np.inf),
        ("output_limit", np.nan),
        ("wheel_radius", np.inf),
    ],
)
def test_mpc_rejects_non_finite_parameters(
    parameter,
    value,
):
    arguments = {
        parameter: value,
    }

    with pytest.raises(
        ValueError,
        match="MPC parameters must be finite",
    ):
        MPCController(**arguments)

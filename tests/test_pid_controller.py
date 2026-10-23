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

from controllers.pid.pid_controller import PIDController


def make_observation(
    pitch=0.0,
    pitch_rate=0.0,
    wheel_velocity=0.0,
    wheel_position=0.0,
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


def test_pid_corrects_positive_pitch():
    controller = PIDController(20.0, 0.0, 1.0)

    command = controller.compute(
        make_observation(pitch=0.1),
        0.02,
    )

    assert command[0] > 0.0
    assert command[0] == command[1]


def test_outer_loop_corrects_positive_position():
    controller = PIDController(20.0, 0.0, 1.0)

    controller.compute(
        make_observation(wheel_position=0.0),
        0.02,
    )
    command = controller.compute(
        make_observation(wheel_position=1.0),
        0.02,
    )

    assert command[0] > 0.0
    assert command[0] == command[1]


def test_pid_output_is_bounded():
    controller = PIDController(
        20.0,
        0.0,
        1.0,
        output_limit=2.0,
    )

    command = controller.compute(
        make_observation(pitch=1.0),
        0.02,
    )

    assert np.all(np.abs(command) <= 2.0)


@pytest.mark.parametrize(
    "observation",
    [
        np.zeros(8, dtype=np.float32),
        np.zeros(10, dtype=np.float32),
        np.zeros((9, 1), dtype=np.float32),
    ],
)
def test_pid_rejects_wrong_observation_shape(observation):
    controller = PIDController(20.0, 0.0, 1.0)

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
def test_pid_rejects_invalid_observation_values(
    invalid_value,
):
    controller = PIDController(20.0, 0.0, 1.0)
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
def test_pid_rejects_invalid_dt(dt):
    controller = PIDController(20.0, 0.0, 1.0)

    with pytest.raises(
        ValueError,
        match="dt must be finite and greater than zero",
    ):
        controller.compute(make_observation(), dt)


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("max_pitch_reference", -0.01),
        ("integral_limit", -0.01),
        ("output_limit", 0.0),
        ("output_limit", -0.5),
    ],
)
def test_pid_rejects_invalid_limits(parameter, value):
    arguments = {
        "kp": 20.0,
        "ki": 0.0,
        "kd": 1.0,
        parameter: value,
    }

    with pytest.raises(ValueError):
        PIDController(**arguments)


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("kp", np.nan),
        ("ki", np.inf),
        ("kd", -np.inf),
        ("position_kp", np.nan),
        ("velocity_kd", np.inf),
        ("max_pitch_reference", np.nan),
        ("integral_limit", np.inf),
        ("output_limit", np.nan),
    ],
)
def test_pid_rejects_non_finite_parameters(
    parameter,
    value,
):
    arguments = {
        "kp": 20.0,
        "ki": 0.0,
        "kd": 1.0,
        parameter: value,
    }

    with pytest.raises(
        ValueError,
        match="PID parameters must be finite",
    ):
        PIDController(**arguments)

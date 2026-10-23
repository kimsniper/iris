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

import mujoco
import numpy as np
import pytest

from simulation.reset_manager import ResetManager
from simulation.robot_interface import RobotInterface


MODEL_PATH = "description/mjcf/two_wheel_robot.xml"


def make_model_and_data():
    model = mujoco.MjModel.from_xml_path(
        MODEL_PATH
    )
    data = mujoco.MjData(model)

    return model, data


def test_reset_pitch_stays_within_requested_range():
    model, data = make_model_and_data()
    rng = np.random.default_rng(1001)

    pitch = ResetManager.reset(
        model,
        data,
        rng,
        0.08,
    )

    assert abs(pitch) <= 0.08
    assert np.isfinite(data.qpos).all()


def test_reset_zero_pitch_is_identity_orientation():
    model, data = make_model_and_data()
    rng = np.random.default_rng(1001)

    pitch = ResetManager.reset(
        model,
        data,
        rng,
        0.0,
    )

    assert pitch == 0.0
    assert np.allclose(
        data.qpos[3:7],
        [1.0, 0.0, 0.0, 0.0],
    )


@pytest.mark.parametrize(
    "max_pitch",
    [
        -0.01,
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_reset_rejects_invalid_pitch_range(max_pitch):
    model, data = make_model_and_data()
    rng = np.random.default_rng(1001)

    with pytest.raises(ValueError):
        ResetManager.reset(
            model,
            data,
            rng,
            max_pitch,
        )


class InvalidResetGenerator:
    def __init__(self, value):
        self.value = value

    def uniform(self, low, high):
        return self.value


@pytest.mark.parametrize(
    "sample",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_reset_rejects_non_finite_sample(sample):
    model, data = make_model_and_data()

    with pytest.raises(
        ValueError,
        match="Sampled reset pitch must be finite",
    ):
        ResetManager.reset(
            model,
            data,
            InvalidResetGenerator(sample),
            0.08,
        )


@pytest.mark.parametrize(
    "sample",
    [
        -0.081,
        0.081,
    ],
)
def test_reset_rejects_sample_outside_range(sample):
    model, data = make_model_and_data()

    with pytest.raises(
        ValueError,
        match="exceeds the requested range",
    ):
        ResetManager.reset(
            model,
            data,
            InvalidResetGenerator(sample),
            0.08,
        )


def test_robot_interface_reads_expected_sensor_shapes():
    model, data = make_model_and_data()
    robot = RobotInterface(model, data)

    mujoco.mj_forward(model, data)

    gyro, acceleration = robot.imu()

    assert gyro.shape == (3,)
    assert acceleration.shape == (3,)
    assert len(robot.wheel_positions()) == 2
    assert len(robot.wheel_velocities()) == 2


def test_robot_interface_applies_motor_command():
    model, data = make_model_and_data()
    robot = RobotInterface(model, data)

    robot.set_torque([0.25, -0.25])

    assert np.allclose(
        data.ctrl,
        [0.25, -0.25],
    )


def test_robot_interface_clips_motor_command():
    model, data = make_model_and_data()
    robot = RobotInterface(model, data)

    robot.set_torque([3.0, -3.0])

    assert np.allclose(
        data.ctrl,
        [2.0, -2.0],
    )


@pytest.mark.parametrize(
    "action",
    [
        [],
        [0.0],
        [0.0, 0.0, 0.0],
        [[0.0], [0.0]],
    ],
)
def test_robot_interface_rejects_wrong_command_shape(
    action,
):
    model, data = make_model_and_data()
    robot = RobotInterface(model, data)

    with pytest.raises(
        ValueError,
        match="must contain two values",
    ):
        robot.set_torque(action)


@pytest.mark.parametrize(
    "invalid_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_robot_interface_rejects_invalid_command(
    invalid_value,
):
    model, data = make_model_and_data()
    robot = RobotInterface(model, data)

    with pytest.raises(
        ValueError,
        match="finite values",
    ):
        robot.set_torque(
            [invalid_value, 0.0]
        )

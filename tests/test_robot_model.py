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


MODEL_PATH = "description/mjcf/two_wheel_robot.xml"


def load_model():
    return mujoco.MjModel.from_xml_path(MODEL_PATH)


def test_required_robot_components_exist():
    model = load_model()

    for body_name in (
        "base_link",
        "left_wheel",
        "right_wheel",
        "imu_link",
    ):
        model.body(body_name)

    for joint_name in (
        "left_wheel_joint",
        "right_wheel_joint",
    ):
        model.joint(joint_name)

    for actuator_name in (
        "left_motor",
        "right_motor",
    ):
        model.actuator(actuator_name)

    for sensor_name in (
        "imu_gyro",
        "imu_accelerometer",
        "left_wheel_position",
        "right_wheel_position",
        "left_wheel_velocity",
        "right_wheel_velocity",
        "base_orientation",
    ):
        model.sensor(sensor_name)


def test_robot_has_two_motor_actuators():
    model = load_model()

    assert model.nu == 2
    assert model.actuator("left_motor").id != (
        model.actuator("right_motor").id
    )


def test_motor_control_ranges_match():
    model = load_model()

    expected_range = np.array(
        [-2.0, 2.0],
        dtype=float,
    )

    assert np.allclose(
        model.actuator_ctrlrange[
            model.actuator("left_motor").id
        ],
        expected_range,
    )
    assert np.allclose(
        model.actuator_ctrlrange[
            model.actuator("right_motor").id
        ],
        expected_range,
    )


def test_imu_sensor_dimensions():
    model = load_model()

    assert model.sensor_dim[
        model.sensor("imu_gyro").id
    ] == 3
    assert model.sensor_dim[
        model.sensor("imu_accelerometer").id
    ] == 3


def test_body_masses_and_inertias_are_valid():
    model = load_model()

    dynamic_body_ids = [
        model.body("base_link").id,
        model.body("left_wheel").id,
        model.body("right_wheel").id,
        model.body("imu_link").id,
    ]

    for body_id in dynamic_body_ids:
        assert np.isfinite(model.body_mass[body_id])
        assert model.body_mass[body_id] > 0.0

        inertia = model.body_inertia[body_id]
        assert np.isfinite(inertia).all()
        assert np.all(inertia > 0.0)


def test_equal_motor_torque_produces_symmetric_wheel_response():
    model = load_model()
    data = mujoco.MjData(model)

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    data.ctrl[:] = [0.05, 0.05]

    steps = round(0.02 / model.opt.timestep)

    for _ in range(steps):
        mujoco.mj_step(model, data)

    left_velocity = float(
        data.sensordata[
            model.sensor_adr[
                model.sensor(
                    "left_wheel_velocity"
                ).id
            ]
        ]
    )
    right_velocity = float(
        data.sensordata[
            model.sensor_adr[
                model.sensor(
                    "right_wheel_velocity"
                ).id
            ]
        ]
    )

    assert left_velocity > 0.0
    assert right_velocity > 0.0
    assert abs(left_velocity - right_velocity) <= 1.0e-6

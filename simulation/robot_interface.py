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


class RobotInterface:
    def __init__(self, model, data):
        self.model = model
        self.data = data

        self.gyro_adr = model.sensor_adr[
            model.sensor("imu_gyro").id
        ]
        self.accel_adr = model.sensor_adr[
            model.sensor("imu_accelerometer").id
        ]
        self.left_vel_adr = model.sensor_adr[
            model.sensor("left_wheel_velocity").id
        ]
        self.right_vel_adr = model.sensor_adr[
            model.sensor("right_wheel_velocity").id
        ]
        self.left_pos_adr = model.sensor_adr[
            model.sensor("left_wheel_position").id
        ]
        self.right_pos_adr = model.sensor_adr[
            model.sensor("right_wheel_position").id
        ]

        if model.nu != 2:
            raise ValueError(
                "IRIS requires exactly two motor actuators."
            )

        if model.actuator_ctrlrange.shape != (2, 2):
            raise ValueError(
                "IRIS requires two actuator control ranges."
            )

        if not np.isfinite(
            model.actuator_ctrlrange
        ).all():
            raise ValueError(
                "Actuator control ranges must be finite."
            )

    def imu(self):
        gyro = self.data.sensordata[
            self.gyro_adr:
            self.gyro_adr + 3
        ].copy()

        acceleration = self.data.sensordata[
            self.accel_adr:
            self.accel_adr + 3
        ].copy()

        return gyro, acceleration

    def wheel_velocities(self):
        return (
            float(
                self.data.sensordata[
                    self.left_vel_adr
                ]
            ),
            float(
                self.data.sensordata[
                    self.right_vel_adr
                ]
            ),
        )

    def wheel_positions(self):
        return (
            float(
                self.data.sensordata[
                    self.left_pos_adr
                ]
            ),
            float(
                self.data.sensordata[
                    self.right_pos_adr
                ]
            ),
        )

    def set_torque(self, action):
        action = np.asarray(
            action,
            dtype=float,
        )

        if action.shape != (2,):
            raise ValueError(
                "Motor command must contain two values."
            )

        if not np.isfinite(action).all():
            raise ValueError(
                "Motor command must contain finite values."
            )

        self.data.ctrl[:] = np.clip(
            action,
            self.model.actuator_ctrlrange[:, 0],
            self.model.actuator_ctrlrange[:, 1],
        )

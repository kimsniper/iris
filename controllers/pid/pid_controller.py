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

from controllers.base_controller import BaseController


class PIDController(BaseController):
    def __init__(
        self,
        kp,
        ki,
        kd,
        position_kp=0.02,
        velocity_kd=0.01,
        max_pitch_reference=0.15,
        integral_limit=0.5,
        output_limit=2.0,
    ):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.position_kp = float(position_kp)
        self.velocity_kd = float(velocity_kd)
        self.max_pitch_reference = float(
            max_pitch_reference
        )
        self.integral_limit = float(integral_limit)
        self.output_limit = float(output_limit)

        parameters = np.asarray(
            [
                self.kp,
                self.ki,
                self.kd,
                self.position_kp,
                self.velocity_kd,
                self.max_pitch_reference,
                self.integral_limit,
                self.output_limit,
            ],
            dtype=float,
        )

        if not np.isfinite(parameters).all():
            raise ValueError(
                "PID parameters must be finite."
            )

        if self.max_pitch_reference < 0.0:
            raise ValueError(
                "Maximum pitch reference must not be negative."
            )

        if self.integral_limit < 0.0:
            raise ValueError(
                "Integral limit must not be negative."
            )

        if self.output_limit <= 0.0:
            raise ValueError(
                "Output limit must be greater than zero."
            )

        self.reset()

    def reset(self):
        self.integral = 0.0
        self.target_wheel_position = None

    def compute(self, observation, dt):
        observation = np.asarray(
            observation,
            dtype=np.float32,
        )

        if observation.shape != (9,):
            raise ValueError(
                "PID requires the nine-element IRIS observation."
            )

        if not np.isfinite(observation).all():
            raise ValueError(
                "PID observation contains invalid values."
            )

        dt = float(dt)

        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError(
                "PID dt must be finite and greater than zero."
            )

        pitch = float(observation[0])
        pitch_rate = float(observation[1])

        average_wheel_velocity = 0.5 * (
            float(observation[2])
            + float(observation[3])
        )
        average_wheel_position = 0.5 * (
            float(observation[7])
            + float(observation[8])
        )

        if self.target_wheel_position is None:
            self.target_wheel_position = (
                average_wheel_position
            )

        position_error = (
            average_wheel_position
            - self.target_wheel_position
        )

        # Request a corrective lean toward the target wheel position.
        pitch_reference = -(
            self.position_kp * position_error
            + self.velocity_kd * average_wheel_velocity
        )
        pitch_reference = float(
            np.clip(
                pitch_reference,
                -self.max_pitch_reference,
                self.max_pitch_reference,
            )
        )

        balance_error = pitch - pitch_reference
        self.integral = np.clip(
            self.integral + balance_error * dt,
            -self.integral_limit,
            self.integral_limit,
        )

        torque = (
            self.kp * balance_error
            + self.ki * self.integral
            + self.kd * pitch_rate
        )
        torque = float(
            np.clip(
                torque,
                -self.output_limit,
                self.output_limit,
            )
        )

        return np.array(
            [torque, torque],
            dtype=np.float32,
        )

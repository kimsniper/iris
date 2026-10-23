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

import math


class ComplementaryFilter:
    def __init__(
        self,
        alpha=0.98,
        max_accel_correction=0.5,
        gravity=9.81,
        accel_norm_tolerance=2.0,
    ):
        self.alpha = float(alpha)
        self.max_accel_correction = float(max_accel_correction)
        self.gravity = float(gravity)
        self.accel_norm_tolerance = float(
            accel_norm_tolerance
        )

        parameters = (
            self.alpha,
            self.max_accel_correction,
            self.gravity,
            self.accel_norm_tolerance,
        )

        if not all(
            math.isfinite(value)
            for value in parameters
        ):
            raise ValueError(
                "Filter parameters must be finite."
            )

        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(
                "Filter alpha must be between zero and one."
            )

        if self.max_accel_correction < 0.0:
            raise ValueError(
                "Maximum acceleration correction "
                "must not be negative."
            )

        if self.gravity <= 0.0:
            raise ValueError(
                "Gravity must be greater than zero."
            )

        if self.accel_norm_tolerance < 0.0:
            raise ValueError(
                "Acceleration tolerance must not be negative."
            )

        self.pitch = 0.0
        self.previous_gyro_y = None

    def reset(self, pitch=0.0):
        pitch = float(pitch)

        if not math.isfinite(pitch):
            raise ValueError(
                "Reset pitch must be finite."
            )

        self.pitch = pitch
        self.previous_gyro_y = None

    @staticmethod
    def _wrap_angle(angle):
        return math.atan2(
            math.sin(angle),
            math.cos(angle),
        )

    def update(self, gyro_y, accel_x, accel_z, dt):
        gyro_y = float(gyro_y)
        accel_x = float(accel_x)
        accel_z = float(accel_z)
        dt = float(dt)

        if not all(
            math.isfinite(value)
            for value in (
                gyro_y,
                accel_x,
                accel_z,
            )
        ):
            raise ValueError(
                "IMU values must be finite."
            )

        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError(
                "Filter dt must be finite and greater than zero."
            )

        if self.previous_gyro_y is None:
            average_gyro_y = gyro_y
        else:
            average_gyro_y = 0.5 * (
                self.previous_gyro_y + gyro_y
            )

        self.previous_gyro_y = gyro_y

        predicted_pitch = self._wrap_angle(
            self.pitch + average_gyro_y * dt
        )

        accel_norm = math.hypot(
            accel_x,
            accel_z,
        )

        # Use acceleration only when gravity is the dominant component.
        if (
            abs(accel_norm - self.gravity)
            > self.accel_norm_tolerance
        ):
            self.pitch = predicted_pitch
            return self.pitch

        accel_pitch = math.atan2(
            -accel_x,
            accel_z,
        )
        correction = self._wrap_angle(
            accel_pitch - predicted_pitch
        )

        if abs(correction) > self.max_accel_correction:
            self.pitch = predicted_pitch
            return self.pitch

        # Preserve the configured filter response across update rates.
        effective_alpha = self.alpha ** (dt / 0.02)

        self.pitch = self._wrap_angle(
            predicted_pitch
            + (1.0 - effective_alpha) * correction
        )
        return self.pitch

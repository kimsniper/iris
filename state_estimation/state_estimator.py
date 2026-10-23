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

from state_estimation.complementary_filter import (
    ComplementaryFilter,
)


class StateEstimator:
    def __init__(self, alpha=0.98):
        self.filter = ComplementaryFilter(alpha)

    def reset(self, pitch=0.0):
        self.filter.reset(pitch)

    def update(self, gyro, acceleration, dt):
        gyro = np.asarray(
            gyro,
            dtype=float,
        )
        acceleration = np.asarray(
            acceleration,
            dtype=float,
        )

        if gyro.shape != (3,):
            raise ValueError(
                "Gyroscope input must contain three values."
            )

        if acceleration.shape != (3,):
            raise ValueError(
                "Acceleration input must contain three values."
            )

        if not np.isfinite(gyro).all():
            raise ValueError(
                "Gyroscope input contains invalid values."
            )

        if not np.isfinite(acceleration).all():
            raise ValueError(
                "Acceleration input contains invalid values."
            )

        pitch = self.filter.update(
            gyro[1],
            acceleration[0],
            acceleration[2],
            dt,
        )

        return pitch, float(gyro[1])

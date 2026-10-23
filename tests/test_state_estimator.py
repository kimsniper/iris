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

from state_estimation.state_estimator import StateEstimator


def test_estimator_returns_pitch_and_gyro_y():
    estimator = StateEstimator()
    estimator.reset(0.05)

    pitch, pitch_rate = estimator.update(
        gyro=np.array(
            [0.0, 0.1, 0.0],
            dtype=float,
        ),
        acceleration=np.array(
            [0.0, 0.0, 20.0],
            dtype=float,
        ),
        dt=0.02,
    )

    assert abs(pitch - 0.052) < 1.0e-6
    assert pitch_rate == 0.1


def test_estimator_accepts_array_like_inputs():
    estimator = StateEstimator()

    pitch, pitch_rate = estimator.update(
        gyro=[0.0, 0.0, 0.0],
        acceleration=[0.0, 0.0, 9.81],
        dt=0.02,
    )

    assert pitch == 0.0
    assert pitch_rate == 0.0


@pytest.mark.parametrize(
    "gyro",
    [
        [],
        [0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [[0.0], [0.0], [0.0]],
    ],
)
def test_estimator_rejects_invalid_gyro_shape(gyro):
    estimator = StateEstimator()

    with pytest.raises(
        ValueError,
        match="Gyroscope input must contain three values",
    ):
        estimator.update(
            gyro=gyro,
            acceleration=[0.0, 0.0, 9.81],
            dt=0.02,
        )


@pytest.mark.parametrize(
    "acceleration",
    [
        [],
        [0.0, 9.81],
        [0.0, 0.0, 9.81, 0.0],
        [[0.0], [0.0], [9.81]],
    ],
)
def test_estimator_rejects_invalid_acceleration_shape(
    acceleration,
):
    estimator = StateEstimator()

    with pytest.raises(
        ValueError,
        match="Acceleration input must contain three values",
    ):
        estimator.update(
            gyro=[0.0, 0.0, 0.0],
            acceleration=acceleration,
            dt=0.02,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_estimator_rejects_invalid_gyro_values(
    invalid_value,
):
    estimator = StateEstimator()
    gyro = np.array(
        [0.0, invalid_value, 0.0],
        dtype=float,
    )

    with pytest.raises(
        ValueError,
        match="Gyroscope input contains invalid values",
    ):
        estimator.update(
            gyro=gyro,
            acceleration=[0.0, 0.0, 9.81],
            dt=0.02,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_estimator_rejects_invalid_acceleration_values(
    invalid_value,
):
    estimator = StateEstimator()
    acceleration = np.array(
        [invalid_value, 0.0, 9.81],
        dtype=float,
    )

    with pytest.raises(
        ValueError,
        match="Acceleration input contains invalid values",
    ):
        estimator.update(
            gyro=[0.0, 0.0, 0.0],
            acceleration=acceleration,
            dt=0.02,
        )


@pytest.mark.parametrize(
    "dt",
    [
        0.0,
        -0.02,
        np.nan,
        np.inf,
    ],
)
def test_estimator_rejects_invalid_dt(dt):
    estimator = StateEstimator()

    with pytest.raises(
        ValueError,
        match="dt must be finite and greater than zero",
    ):
        estimator.update(
            gyro=[0.0, 0.0, 0.0],
            acceleration=[0.0, 0.0, 9.81],
            dt=dt,
        )

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

import pytest

from state_estimation.complementary_filter import ComplementaryFilter


def test_filter_accepts_consistent_accelerometer_pitch():
    pitch = 0.05
    filter_ = ComplementaryFilter()
    filter_.reset(pitch)

    estimate = filter_.update(
        gyro_y=0.0,
        accel_x=-9.81 * math.sin(pitch),
        accel_z=9.81 * math.cos(pitch),
        dt=0.02,
    )

    assert abs(estimate - pitch) < 1e-6


def test_filter_rejects_opposite_accelerometer_sample():
    filter_ = ComplementaryFilter()
    filter_.reset(0.05)

    estimate = filter_.update(
        gyro_y=0.0,
        accel_x=0.49,
        accel_z=-9.80,
        dt=0.02,
    )

    assert abs(estimate - 0.05) < 1e-6


def test_filter_rejects_high_linear_acceleration():
    filter_ = ComplementaryFilter()
    filter_.reset(0.05)

    estimate = filter_.update(
        gyro_y=0.1,
        accel_x=20.0,
        accel_z=20.0,
        dt=0.02,
    )

    expected = 0.05 + 0.1 * 0.02

    assert abs(estimate - expected) < 1e-6


def test_filter_uses_average_gyro_rate():
    filter_ = ComplementaryFilter(alpha=1.0)
    filter_.reset(0.0)

    filter_.update(
        gyro_y=0.0,
        accel_x=0.0,
        accel_z=9.81,
        dt=0.02,
    )
    estimate = filter_.update(
        gyro_y=1.0,
        accel_x=0.0,
        accel_z=20.0,
        dt=0.02,
    )

    assert abs(estimate - 0.01) < 1e-6


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("alpha", -0.01),
        ("alpha", 1.01),
        ("max_accel_correction", -0.01),
        ("gravity", 0.0),
        ("gravity", -9.81),
        ("accel_norm_tolerance", -0.01),
    ],
)
def test_filter_rejects_invalid_parameters(
    parameter,
    value,
):
    arguments = {
        parameter: value,
    }

    with pytest.raises(ValueError):
        ComplementaryFilter(**arguments)


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("alpha", math.nan),
        ("max_accel_correction", math.inf),
        ("gravity", -math.inf),
        ("accel_norm_tolerance", math.nan),
    ],
)
def test_filter_rejects_non_finite_parameters(
    parameter,
    value,
):
    arguments = {
        parameter: value,
    }

    with pytest.raises(
        ValueError,
        match="Filter parameters must be finite",
    ):
        ComplementaryFilter(**arguments)


@pytest.mark.parametrize(
    "pitch",
    [
        math.nan,
        math.inf,
        -math.inf,
    ],
)
def test_filter_rejects_invalid_reset_pitch(pitch):
    filter_ = ComplementaryFilter()

    with pytest.raises(
        ValueError,
        match="Reset pitch must be finite",
    ):
        filter_.reset(pitch)


@pytest.mark.parametrize(
    ("gyro_y", "accel_x", "accel_z"),
    [
        (math.nan, 0.0, 9.81),
        (0.0, math.inf, 9.81),
        (0.0, 0.0, -math.inf),
    ],
)
def test_filter_rejects_invalid_imu_values(
    gyro_y,
    accel_x,
    accel_z,
):
    filter_ = ComplementaryFilter()

    with pytest.raises(
        ValueError,
        match="IMU values must be finite",
    ):
        filter_.update(
            gyro_y,
            accel_x,
            accel_z,
            0.002,
        )


@pytest.mark.parametrize(
    "dt",
    [
        0.0,
        -0.002,
        math.nan,
        math.inf,
    ],
)
def test_filter_rejects_invalid_dt(dt):
    filter_ = ComplementaryFilter()

    with pytest.raises(
        ValueError,
        match="dt must be finite and greater than zero",
    ):
        filter_.update(
            gyro_y=0.0,
            accel_x=0.0,
            accel_z=9.81,
            dt=dt,
        )

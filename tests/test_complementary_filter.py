import math

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

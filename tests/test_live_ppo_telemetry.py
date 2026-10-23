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

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from scripts.live_ppo_telemetry import TelemetryViewer


def close_viewer(viewer):
    import matplotlib.pyplot as plt

    viewer.socket.close()
    plt.close(viewer.figure)


@pytest.mark.parametrize(
    "window_seconds",
    [
        0.0,
        -1.0,
        np.nan,
        np.inf,
    ],
)
def test_viewer_rejects_invalid_window(
    window_seconds,
):
    with pytest.raises(
        ValueError,
        match="window must be finite",
    ):
        TelemetryViewer(
            "127.0.0.1",
            8765,
            window_seconds,
        )


@pytest.mark.parametrize(
    "message",
    [
        None,
        [],
        {},
        {"type": "unknown"},
        {"type": "sample"},
    ],
)
def test_viewer_ignores_invalid_messages(
    message,
):
    viewer = TelemetryViewer(
        "127.0.0.1",
        0,
        10.0,
    )

    try:
        assert not viewer.process_message(
            message
        )
        assert viewer.times == []
    finally:
        close_viewer(viewer)


def test_viewer_processes_episode_messages():
    viewer = TelemetryViewer(
        "127.0.0.1",
        0,
        10.0,
    )

    try:
        assert viewer.process_message(
            {
                "type": "episode_start",
                "checkpoint": "checkpoint_0_steps.zip",
                "episode": 1,
                "seed": 12001,
            }
        )

        assert "checkpoint_0_steps.zip" in viewer.title

        assert viewer.process_message(
            {
                "type": "episode_end",
                "status": "FELL",
                "seconds": 0.76,
                "return": -76.13,
            }
        )

        assert "FELL" in viewer.title
    finally:
        close_viewer(viewer)


def test_viewer_processes_finite_sample():
    viewer = TelemetryViewer(
        "127.0.0.1",
        0,
        10.0,
    )

    try:
        accepted = viewer.process_message(
            {
                "type": "sample",
                "time_s": 0.02,
                "truth_pitch_rad": 0.01,
                "estimated_pitch_rad": 0.011,
                "left_wheel_velocity_rad_s": 0.1,
                "right_wheel_velocity_rad_s": 0.1,
                "average_wheel_position_rad": 0.01,
                "normalized_action": 0.2,
                "motor_torque_nm": 0.1,
            }
        )

        assert accepted
        assert viewer.times == [0.02]
        assert viewer.truth_pitch == [0.01]
    finally:
        close_viewer(viewer)


def test_viewer_rejects_non_finite_sample():
    viewer = TelemetryViewer(
        "127.0.0.1",
        0,
        10.0,
    )

    try:
        accepted = viewer.process_message(
            {
                "type": "sample",
                "time_s": 0.02,
                "truth_pitch_rad": np.nan,
                "estimated_pitch_rad": 0.0,
                "left_wheel_velocity_rad_s": 0.0,
                "right_wheel_velocity_rad_s": 0.0,
                "average_wheel_position_rad": 0.0,
                "normalized_action": 0.0,
                "motor_torque_nm": 0.0,
            }
        )

        assert not accepted
        assert viewer.times == []
    finally:
        close_viewer(viewer)


def test_viewer_discards_samples_outside_window():
    viewer = TelemetryViewer(
        "127.0.0.1",
        0,
        1.0,
    )

    try:
        for time_s in (
            0.0,
            0.5,
            1.0,
            1.5,
        ):
            assert viewer.process_message(
                {
                    "type": "sample",
                    "time_s": time_s,
                    "truth_pitch_rad": 0.0,
                    "estimated_pitch_rad": 0.0,
                    "left_wheel_velocity_rad_s": 0.0,
                    "right_wheel_velocity_rad_s": 0.0,
                    "average_wheel_position_rad": 0.0,
                    "normalized_action": 0.0,
                    "motor_torque_nm": 0.0,
                }
            )

        assert viewer.times == [
            0.5,
            1.0,
            1.5,
        ]
    finally:
        close_viewer(viewer)

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

from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

from scripts.plot_ppo_telemetry import (
    plot_telemetry,
    save_csv,
)


def make_record(time_s=0.02):
    return {
        "time_s": time_s,
        "truth_pitch_rad": 0.01,
        "estimated_pitch_rad": 0.011,
        "pitch_rate_rad_s": 0.02,
        "left_wheel_velocity_rad_s": 0.1,
        "right_wheel_velocity_rad_s": 0.1,
        "accel_x_m_s2": 0.0,
        "accel_z_m_s2": 9.81,
        "normalized_action": 0.2,
        "left_wheel_position_rad": 0.01,
        "right_wheel_position_rad": 0.01,
        "motor_torque_nm": 0.1,
        "reward": 0.99,
    }


@pytest.mark.parametrize(
    "function",
    [
        save_csv,
        plot_telemetry,
    ],
)
def test_telemetry_outputs_reject_empty_records(
    function,
    tmp_path,
):
    output = tmp_path / "output"

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        if function is save_csv:
            function([], output)
        else:
            function(
                [],
                output,
                1001,
                {
                    "completed": False,
                },
            )


def test_save_csv_preserves_record_columns(
    tmp_path,
):
    output = tmp_path / "telemetry.csv"

    save_csv(
        [make_record()],
        output,
    )

    text = output.read_text()

    assert output.is_file()
    assert "truth_pitch_rad" in text
    assert "normalized_action" in text
    assert "motor_torque_nm" in text


@pytest.mark.parametrize(
    "completed",
    [
        True,
        False,
    ],
)
def test_plot_telemetry_uses_explicit_status(
    tmp_path,
    completed,
):
    output = tmp_path / "telemetry.png"

    figure = plot_telemetry(
        [make_record()],
        output,
        seed=1001,
        episode={
            "completed": completed,
        },
    )

    try:
        assert output.is_file()

        title = figure._suptitle.get_text()

        expected = (
            "completed"
            if completed
            else "fell"
        )

        assert expected in title
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)

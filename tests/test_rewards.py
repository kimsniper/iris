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

import pytest

from rl.common.rewards import (
    balancing_reward,
    direct_control_reward,
)


def test_balancing_reward_is_one_at_equilibrium():
    reward = balancing_reward(
        pitch=0.0,
        pitch_rate=0.0,
        wheel_velocity=0.0,
        action=0.0,
        previous_action=0.0,
    )

    assert reward == 1.0


def test_balancing_reward_matches_equation():
    reward = balancing_reward(
        pitch=0.1,
        pitch_rate=0.2,
        wheel_velocity=0.3,
        action=0.4,
        previous_action=0.1,
    )

    expected = (
        1.0
        - 8.0 * 0.1**2
        - 0.2 * 0.2**2
        - 0.02 * 0.3**2
        - 0.01 * 0.4**2
        - 0.02 * (0.4 - 0.1) ** 2
    )

    assert reward == pytest.approx(expected)


def test_direct_control_reward_is_one_at_equilibrium():
    reward = direct_control_reward(
        pitch=0.0,
        pitch_rate=0.0,
        wheel_position=0.0,
        wheel_velocity=0.0,
        normalized_action=0.0,
        previous_action=0.0,
        terminated=False,
    )

    assert reward == 1.0


def test_direct_control_reward_matches_equation():
    reward = direct_control_reward(
        pitch=0.1,
        pitch_rate=0.2,
        wheel_position=0.3,
        wheel_velocity=0.4,
        normalized_action=0.5,
        previous_action=0.2,
        terminated=False,
    )

    expected = (
        1.0
        - 12.0 * 0.1**2
        - 0.5 * 0.2**2
        - 0.05 * 0.3**2
        - 0.02 * 0.4**2
        - 0.01 * 0.5**2
        - 0.02 * (0.5 - 0.2) ** 2
    )

    assert reward == pytest.approx(expected)


def test_direct_control_terminal_penalty_is_exact():
    arguments = {
        "pitch": 0.1,
        "pitch_rate": 0.2,
        "wheel_position": 0.3,
        "wheel_velocity": 0.4,
        "normalized_action": 0.5,
        "previous_action": 0.2,
    }

    running_reward = direct_control_reward(
        **arguments,
        terminated=False,
    )
    terminal_reward = direct_control_reward(
        **arguments,
        terminated=True,
    )

    assert terminal_reward == pytest.approx(
        running_reward - 25.0
    )

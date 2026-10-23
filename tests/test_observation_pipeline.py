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

from rl.environments.balancing_env import BalancingEnv


def test_reset_observation_matches_initial_pitch():
    env = BalancingEnv(max_initial_pitch=0.05)
    observation, _ = env.reset(seed=0)

    truth_pitch = env._truth_pitch()
    estimated_pitch = float(observation[0])

    assert np.isfinite(observation).all()
    assert abs(estimated_pitch - truth_pitch) <= 0.01


def test_zero_torque_observation_remains_finite():
    env = BalancingEnv(
        max_initial_pitch=0.05,
        max_seconds=0.2,
    )
    env.reset(seed=0)

    for _ in range(5):
        observation, _, terminated, truncated, _ = env.step(
            np.array([0.0], dtype=np.float32)
        )

        assert np.isfinite(observation).all()

        if terminated or truncated:
            break


def test_equal_motor_commands_keep_wheel_speeds_symmetric():
    env = BalancingEnv(
        max_initial_pitch=0.0,
        max_seconds=0.2,
    )
    env.reset(seed=0)

    observation, _, _, _, _ = env.step(
        np.array([0.05], dtype=np.float32)
    )

    left_wheel_velocity = float(observation[2])
    right_wheel_velocity = float(observation[3])

    assert abs(
        left_wheel_velocity - right_wheel_velocity
    ) <= 1e-6


def test_fall_angle_terminates_episode():
    env = BalancingEnv(
        max_initial_pitch=0.0,
        max_seconds=2.0,
        fall_angle=0.10,
    )
    env.reset(seed=0)

    terminated = False

    for _ in range(100):
        _, _, terminated, truncated, _ = env.step(
            np.array([-0.05], dtype=np.float32)
        )

        if terminated or truncated:
            break

    assert terminated

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
from gymnasium.utils.env_checker import check_env

from rl.environments.direct_control_env import DirectControlEnv


def test_direct_control_environment_passes_checker():
    env = DirectControlEnv()
    check_env(env, skip_render_check=True)


def test_direct_control_action_space_is_normalized():
    env = DirectControlEnv()

    assert env.action_space.shape == (1,)
    assert np.all(env.action_space.low == -1.0)
    assert np.all(env.action_space.high == 1.0)


def test_direct_control_reset_uses_relative_wheel_positions():
    env = DirectControlEnv()
    observation, _ = env.reset(seed=1001)

    assert observation.shape == (9,)
    assert abs(float(observation[7])) < 1e-6
    assert abs(float(observation[8])) < 1e-6


def test_direct_control_scales_action_to_torque():
    env = DirectControlEnv(torque_limit=0.5)
    env.reset(seed=1001)

    _, _, _, _, info = env.step(
        np.array([0.5], dtype=np.float32)
    )

    assert abs(info["motor_torque"] - 0.25) < 1e-6


def test_direct_control_clips_normalized_action():
    env = DirectControlEnv(torque_limit=0.5)
    env.reset(seed=1001)

    observation, _, _, _, info = env.step(
        np.array([2.0], dtype=np.float32)
    )

    assert info["normalized_action"] == 1.0
    assert info["motor_torque"] == 0.5
    assert observation[6] == 1.0


def test_direct_control_reward_is_finite():
    env = DirectControlEnv()
    env.reset(seed=1001)

    _, reward, _, _, _ = env.step(
        np.array([0.0], dtype=np.float32)
    )

    assert np.isfinite(reward)


import pytest


@pytest.mark.parametrize(
    "torque_limit",
    [
        0.0,
        -0.5,
        np.nan,
        np.inf,
    ],
)
def test_direct_control_rejects_invalid_torque_limit(
    torque_limit,
):
    with pytest.raises(
        ValueError,
        match="finite and greater than zero",
    ):
        DirectControlEnv(
            torque_limit=torque_limit
        )


@pytest.mark.parametrize(
    "action",
    [
        [],
        [0.0, 0.0],
        np.zeros((2, 1), dtype=float),
    ],
)
def test_direct_control_rejects_wrong_action_size(
    action,
):
    env = DirectControlEnv(
        max_seconds=0.1
    )
    env.reset(seed=0)

    with pytest.raises(
        ValueError,
        match="Environment action must contain one value",
    ):
        env.step(action)


@pytest.mark.parametrize(
    "action",
    [
        [np.nan],
        [np.inf],
        [-np.inf],
    ],
)
def test_direct_control_rejects_non_finite_action(
    action,
):
    env = DirectControlEnv(
        max_seconds=0.1
    )
    env.reset(seed=0)

    with pytest.raises(
        ValueError,
        match="Environment action must be finite",
    ):
        env.step(action)

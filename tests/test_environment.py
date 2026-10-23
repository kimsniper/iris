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

from rl.environments.balancing_env import BalancingEnv

def test_environment_reset_and_step():
    env=BalancingEnv(max_seconds=0.1); obs,_=env.reset(seed=0); assert obs.shape==(9,)
    obs,reward,terminated,truncated,info=env.step([0.0]); assert obs.shape==(9,) and isinstance(reward,float)


def test_reset_applies_requested_pitch_range():
    env = BalancingEnv(max_initial_pitch=0.08)
    env.reset(seed=0)

    assert abs(env._truth_pitch()) <= 0.08 + 1e-6


import numpy as np
import pytest


@pytest.mark.parametrize(
    "frame_skip",
    [
        0,
        -1,
        1.5,
        True,
    ],
)
def test_environment_rejects_invalid_frame_skip(
    frame_skip,
):
    with pytest.raises(
        ValueError,
        match="positive integer",
    ):
        BalancingEnv(
            frame_skip=frame_skip
        )


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("max_seconds", 0.0),
        ("max_seconds", -1.0),
        ("max_initial_pitch", -0.01),
        ("fall_angle", 0.0),
        ("fall_angle", -0.1),
    ],
)
def test_environment_rejects_invalid_parameters(
    parameter,
    value,
):
    arguments = {
        parameter: value,
    }

    with pytest.raises(ValueError):
        BalancingEnv(**arguments)


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("max_seconds", np.nan),
        ("max_initial_pitch", np.inf),
        ("fall_angle", -np.inf),
    ],
)
def test_environment_rejects_non_finite_parameters(
    parameter,
    value,
):
    arguments = {
        parameter: value,
    }

    with pytest.raises(
        ValueError,
        match="parameters must be finite",
    ):
        BalancingEnv(**arguments)


@pytest.mark.parametrize(
    "action",
    [
        [],
        [0.0, 0.0],
        np.zeros((2, 1), dtype=float),
    ],
)
def test_environment_rejects_wrong_action_size(
    action,
):
    env = BalancingEnv(
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
def test_environment_rejects_non_finite_action(
    action,
):
    env = BalancingEnv(
        max_seconds=0.1
    )
    env.reset(seed=0)

    with pytest.raises(
        ValueError,
        match="Environment action must be finite",
    ):
        env.step(action)

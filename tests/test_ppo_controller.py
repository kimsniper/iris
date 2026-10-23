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

from controllers.ppo.ppo_controller import PPOController


class FakePolicy:
    def __init__(self, action):
        self.action = action

    def predict(self, observation, deterministic):
        assert deterministic
        return (
            np.array([self.action], dtype=np.float32),
            None,
        )


def make_controller(action, torque_limit=0.5):
    controller = PPOController.__new__(PPOController)
    controller.model = FakePolicy(action)
    controller.torque_limit = torque_limit
    controller.reset()
    return controller


def test_ppo_controller_scales_normalized_action():
    controller = make_controller(0.5)

    command = controller.compute(
        np.zeros(9, dtype=np.float32),
        0.02,
    )

    assert np.allclose(command, [0.25, 0.25])


def test_ppo_controller_clips_action():
    controller = make_controller(2.0)

    command = controller.compute(
        np.zeros(9, dtype=np.float32),
        0.02,
    )

    assert np.allclose(command, [0.5, 0.5])


def test_ppo_controller_returns_equal_wheel_torque():
    controller = make_controller(-0.4)

    command = controller.compute(
        np.zeros(9, dtype=np.float32),
        0.02,
    )

    assert command.shape == (2,)
    assert command[0] == command[1]
    assert command[0] < 0.0


class RecordingPolicy:
    def __init__(self):
        self.observations = []

    def predict(self, observation, deterministic):
        self.observations.append(
            np.asarray(observation).copy()
        )
        return np.array([0.0], dtype=np.float32), None


def test_ppo_controller_uses_relative_wheel_positions():
    controller = PPOController.__new__(PPOController)
    controller.model = RecordingPolicy()
    controller.torque_limit = 0.5
    controller.reset()

    first = np.zeros(9, dtype=np.float32)
    first[7] = 5.0
    first[8] = 5.0

    second = first.copy()
    second[7] = 5.25
    second[8] = 5.25

    controller.compute(first, 0.02)
    controller.compute(second, 0.02)

    assert np.allclose(
        controller.model.observations[0][7:9],
        [0.0, 0.0],
    )
    assert np.allclose(
        controller.model.observations[1][7:9],
        [0.25, 0.25],
    )


def test_ppo_controller_reset_clears_wheel_reference():
    controller = make_controller(0.0)

    observation = np.zeros(9, dtype=np.float32)
    observation[7:9] = 2.0

    controller.compute(observation, 0.02)

    assert controller.initial_left_wheel_position == 2.0
    assert controller.initial_right_wheel_position == 2.0

    controller.reset()

    assert controller.initial_left_wheel_position is None
    assert controller.initial_right_wheel_position is None


def test_ppo_controller_tracks_previous_normalized_action():
    controller = PPOController.__new__(PPOController)
    controller.model = RecordingPolicy()
    controller.torque_limit = 0.5
    controller.reset()

    observation = np.zeros(9, dtype=np.float32)
    observation[6] = 0.25

    controller.compute(observation, 0.02)

    first_policy_observation = (
        controller.model.observations[0]
    )

    assert first_policy_observation[6] == 0.0

    controller.previous_normalized_action = 0.5
    controller.compute(observation, 0.02)

    second_policy_observation = (
        controller.model.observations[1]
    )

    assert second_policy_observation[6] == 0.5


def test_ppo_controller_saves_current_normalized_action():
    controller = make_controller(0.4)

    controller.compute(
        np.zeros(9, dtype=np.float32),
        0.02,
    )

    assert abs(
        controller.previous_normalized_action - 0.4
    ) < 1e-6


@pytest.mark.parametrize(
    "observation",
    [
        np.zeros(8, dtype=np.float32),
        np.zeros(10, dtype=np.float32),
        np.zeros((9, 1), dtype=np.float32),
    ],
)
def test_ppo_controller_rejects_wrong_observation_shape(
    observation,
):
    controller = make_controller(0.0)

    with pytest.raises(
        ValueError,
        match="nine-element IRIS observation",
    ):
        controller.compute(observation, 0.02)


@pytest.mark.parametrize(
    "invalid_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_ppo_controller_rejects_invalid_observation_values(
    invalid_value,
):
    controller = make_controller(0.0)
    observation = np.zeros(9, dtype=np.float32)
    observation[0] = invalid_value

    with pytest.raises(
        ValueError,
        match="observation contains invalid values",
    ):
        controller.compute(observation, 0.02)


@pytest.mark.parametrize(
    "dt",
    [
        0.0,
        -0.02,
        np.nan,
        np.inf,
    ],
)
def test_ppo_controller_rejects_invalid_dt(dt):
    controller = make_controller(0.0)

    with pytest.raises(
        ValueError,
        match="dt must be finite and greater than zero",
    ):
        controller.compute(
            np.zeros(9, dtype=np.float32),
            dt,
        )


@pytest.mark.parametrize(
    "action",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_ppo_controller_rejects_invalid_policy_action(
    action,
):
    controller = make_controller(action)

    with pytest.raises(
        RuntimeError,
        match="policy returned an invalid action",
    ):
        controller.compute(
            np.zeros(9, dtype=np.float32),
            0.02,
        )


@pytest.mark.parametrize(
    "action",
    [
        [],
        [0.0, 0.0],
    ],
)
def test_ppo_controller_rejects_wrong_action_size(action):
    controller = make_controller(0.0)
    controller.model.action = action

    with pytest.raises(
        RuntimeError,
        match="policy must return one action",
    ):
        controller.compute(
            np.zeros(9, dtype=np.float32),
            0.02,
        )


@pytest.mark.parametrize(
    "torque_limit",
    [
        0.0,
        -0.5,
        np.nan,
        np.inf,
    ],
)
def test_ppo_controller_rejects_invalid_torque_limit(
    torque_limit,
    monkeypatch,
):
    load_called = False

    def fake_load(*args, **kwargs):
        nonlocal load_called
        load_called = True
        return FakePolicy(0.0)

    monkeypatch.setattr(
        "controllers.ppo.ppo_controller.PPO.load",
        fake_load,
    )

    with pytest.raises(
        ValueError,
        match="finite and greater than zero",
    ):
        PPOController(
            "unused-model.zip",
            torque_limit=torque_limit,
        )

    assert not load_called

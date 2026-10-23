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
from stable_baselines3 import PPO

from controllers.base_controller import BaseController


class PPOController(BaseController):
    def __init__(
        self,
        model_path,
        torque_limit=0.5,
    ):
        self.torque_limit = float(torque_limit)

        if (
            not np.isfinite(self.torque_limit)
            or self.torque_limit <= 0.0
        ):
            raise ValueError(
                "Torque limit must be finite and greater than zero."
            )

        self.model = PPO.load(
            model_path,
            device="cpu",
        )

        self.reset()

    def reset(self):
        self.initial_left_wheel_position = None
        self.initial_right_wheel_position = None
        self.previous_normalized_action = 0.0

    def compute(self, observation, dt):
        dt = float(dt)

        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError(
                "PPO dt must be finite and greater than zero."
            )

        policy_observation = np.asarray(
            observation,
            dtype=np.float32,
        ).copy()

        if policy_observation.shape != (9,):
            raise ValueError(
                "PPO requires the nine-element IRIS observation."
            )

        if not np.isfinite(policy_observation).all():
            raise ValueError(
                "PPO observation contains invalid values."
            )

        if self.initial_left_wheel_position is None:
            self.initial_left_wheel_position = float(
                policy_observation[7]
            )
            self.initial_right_wheel_position = float(
                policy_observation[8]
            )

        policy_observation[7] -= (
            self.initial_left_wheel_position
        )
        policy_observation[8] -= (
            self.initial_right_wheel_position
        )
        policy_observation[6] = (
            self.previous_normalized_action
        )

        action, _ = self.model.predict(
            policy_observation,
            deterministic=True,
        )

        action = np.asarray(
            action,
            dtype=float,
        ).reshape(-1)

        if action.size != 1:
            raise RuntimeError(
                "PPO policy must return one action."
            )

        if not np.isfinite(action[0]):
            raise RuntimeError(
                "PPO policy returned an invalid action."
            )

        normalized_action = float(
            np.clip(
                action[0],
                -1.0,
                1.0,
            )
        )
        self.previous_normalized_action = (
            normalized_action
        )
        torque = normalized_action * self.torque_limit

        return np.array(
            [torque, torque],
            dtype=np.float32,
        )

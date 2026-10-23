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

import mujoco
import numpy as np
from gymnasium import spaces

from rl.common.rewards import direct_control_reward
from rl.environments.balancing_env import BalancingEnv


class DirectControlEnv(BalancingEnv):
    def __init__(
        self,
        torque_limit=0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.torque_limit = float(torque_limit)

        if (
            not np.isfinite(self.torque_limit)
            or self.torque_limit <= 0.0
        ):
            raise ValueError(
                "Torque limit must be finite "
                "and greater than zero."
            )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

        self.initial_left_wheel_position = 0.0
        self.initial_right_wheel_position = 0.0

    def reset(self, seed=None, options=None):
        observation, info = super().reset(
            seed=seed,
            options=options,
        )

        (
            self.initial_left_wheel_position,
            self.initial_right_wheel_position,
        ) = self.robot.wheel_positions()

        self.previous_action = 0.0

        return self._direct_observation(), info

    def _direct_observation(self):
        observation = self._observation().copy()

        observation[6] = self.previous_action
        observation[7] -= self.initial_left_wheel_position
        observation[8] -= self.initial_right_wheel_position

        return observation

    def step(self, action):
        normalized_action = float(
            np.clip(
                self._scalar_action(action),
                -1.0,
                1.0,
            )
        )
        torque = normalized_action * self.torque_limit

        self.robot.set_torque([torque, torque])

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            self._update_estimator()

        self.steps += 1

        observation = self._direct_observation()

        average_wheel_velocity = 0.5 * (
            float(observation[2])
            + float(observation[3])
        )
        average_wheel_position = 0.5 * (
            float(observation[7])
            + float(observation[8])
        )

        truth_pitch = self._truth_pitch()
        terminated = abs(truth_pitch) > self.fall_angle
        truncated = self.steps >= self.max_steps

        reward = direct_control_reward(
            pitch=float(observation[0]),
            pitch_rate=float(observation[1]),
            wheel_position=average_wheel_position,
            wheel_velocity=average_wheel_velocity,
            normalized_action=normalized_action,
            previous_action=self.previous_action,
            terminated=terminated,
        )

        self.previous_action = normalized_action
        observation[6] = normalized_action

        info = {
            "truth_pitch": truth_pitch,
            "motor_torque": torque,
            "normalized_action": normalized_action,
            "wheel_position": average_wheel_position,
            "wheel_velocity": average_wheel_velocity,
        }

        return (
            observation,
            float(reward),
            terminated,
            truncated,
            info,
        )

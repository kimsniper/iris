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

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from controllers.ppo.observation_builder import (
    build_observation,
)
from rl.common.rewards import balancing_reward
from simulation.reset_manager import ResetManager
from simulation.robot_interface import RobotInterface
from state_estimation.state_estimator import StateEstimator


class BalancingEnv(gym.Env):
    metadata = {
        "render_modes": [],
    }

    def __init__(
        self,
        model_path="description/mjcf/two_wheel_robot.xml",
        frame_skip=10,
        max_seconds=10.0,
        max_initial_pitch=0.08,
        fall_angle=0.70,
    ):
        if (
            isinstance(frame_skip, bool)
            or not isinstance(frame_skip, (int, np.integer))
            or frame_skip <= 0
        ):
            raise ValueError(
                "Frame skip must be a positive integer."
            )

        max_seconds = float(max_seconds)
        max_initial_pitch = float(max_initial_pitch)
        fall_angle = float(fall_angle)

        parameters = np.asarray(
            [
                max_seconds,
                max_initial_pitch,
                fall_angle,
            ],
            dtype=float,
        )

        if not np.isfinite(parameters).all():
            raise ValueError(
                "Environment parameters must be finite."
            )

        if max_seconds <= 0.0:
            raise ValueError(
                "Maximum duration must be greater than zero."
            )

        if max_initial_pitch < 0.0:
            raise ValueError(
                "Maximum initial pitch must not be negative."
            )

        if fall_angle <= 0.0:
            raise ValueError(
                "Fall angle must be greater than zero."
            )

        self.model = mujoco.MjModel.from_xml_path(
            str(Path(model_path))
        )
        self.data = mujoco.MjData(self.model)
        self.frame_skip = int(frame_skip)

        self.dt = (
            self.model.opt.timestep
            * self.frame_skip
        )

        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError(
                "Environment timestep must be positive."
            )

        self.max_steps = int(max_seconds / self.dt)

        if self.max_steps <= 0:
            raise ValueError(
                "Maximum duration must include at least one step."
            )

        self.max_initial_pitch = max_initial_pitch
        self.fall_angle = fall_angle

        self.robot = RobotInterface(
            self.model,
            self.data,
        )
        self.estimator = StateEstimator()

        self.action_space = spaces.Box(
            -2.0,
            2.0,
            shape=(1,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=(9,),
            dtype=np.float32,
        )

        self.previous_action = 0.0
        self.steps = 0

    def _truth_pitch(self):
        quaternion_wxyz = self.data.qpos[3:7]
        quaternion_xyzw = [
            quaternion_wxyz[1],
            quaternion_wxyz[2],
            quaternion_wxyz[3],
            quaternion_wxyz[0],
        ]

        return float(
            Rotation.from_quat(
                quaternion_xyzw
            ).as_euler("xyz")[1]
        )

    def _update_estimator(self):
        gyro, acceleration = self.robot.imu()

        (
            self.estimated_pitch,
            self.estimated_pitch_rate,
        ) = self.estimator.update(
            gyro,
            acceleration,
            self.model.opt.timestep,
        )

    def _observation(self):
        _, acceleration = self.robot.imu()

        (
            left_velocity,
            right_velocity,
        ) = self.robot.wheel_velocities()

        (
            left_position,
            right_position,
        ) = self.robot.wheel_positions()

        return build_observation(
            self.estimated_pitch,
            self.estimated_pitch_rate,
            left_velocity,
            right_velocity,
            acceleration[0],
            acceleration[2],
            self.previous_action,
            left_position,
            right_position,
        )

    @staticmethod
    def _scalar_action(action):
        action = np.asarray(
            action,
            dtype=float,
        )

        if action.size != 1:
            raise ValueError(
                "Environment action must contain one value."
            )

        torque = float(
            action.reshape(-1)[0]
        )

        if not np.isfinite(torque):
            raise ValueError(
                "Environment action must be finite."
            )

        return torque

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        initial_pitch = ResetManager.reset(
            self.model,
            self.data,
            self.np_random,
            self.max_initial_pitch,
        )

        self.estimator.reset(initial_pitch)
        self.estimated_pitch = initial_pitch
        self.estimated_pitch_rate = 0.0
        self.previous_action = 0.0
        self.steps = 0

        return self._observation(), {}

    def step(self, action):
        torque = float(
            np.clip(
                self._scalar_action(action),
                -2.0,
                2.0,
            )
        )

        self.robot.set_torque(
            [torque, torque]
        )

        for _ in range(self.frame_skip):
            mujoco.mj_step(
                self.model,
                self.data,
            )
            self._update_estimator()

        observation = self._observation()
        self.steps += 1

        wheel_velocity = 0.5 * (
            observation[2]
            + observation[3]
        )

        reward = balancing_reward(
            observation[0],
            observation[1],
            wheel_velocity,
            torque,
            self.previous_action,
        )

        self.previous_action = torque
        truth_pitch = self._truth_pitch()

        terminated = (
            abs(truth_pitch)
            > self.fall_angle
        )
        truncated = (
            self.steps
            >= self.max_steps
        )

        return (
            observation,
            float(reward),
            terminated,
            truncated,
            {
                "truth_pitch": truth_pitch,
            },
        )

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
import osqp
from scipy import sparse

from controllers.base_controller import BaseController
from controllers.mpc.dynamics_model import load_dynamics


class MPCController(BaseController):
    def __init__(
        self,
        dt=0.02,
        horizon=10,
        q_pitch=100.0,
        q_pitch_rate=5.0,
        q_position=1.0,
        q_velocity=1.0,
        r_input=0.1,
        output_limit=0.5,
        wheel_radius=0.08,
        **_,
    ):
        dynamics = load_dynamics()

        dt = float(dt)

        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError(
                "MPC timestep must be finite and greater than zero."
            )

        if not np.isclose(
            dt,
            dynamics["control_dt"],
        ):
            raise ValueError(
                "Controller timestep does not match "
                "identified dynamics."
            )

        self.A = dynamics["A"]
        self.B = dynamics["B"]

        self.dt = dt
        self.horizon = int(horizon)
        self.output_limit = float(output_limit)
        self.wheel_radius = float(wheel_radius)

        parameters = np.asarray(
            [
                q_pitch,
                q_pitch_rate,
                q_position,
                q_velocity,
                r_input,
                self.output_limit,
                self.wheel_radius,
            ],
            dtype=float,
        )

        if not np.isfinite(parameters).all():
            raise ValueError(
                "MPC parameters must be finite."
            )

        self.Q = np.diag(
            [
                float(q_pitch),
                float(q_pitch_rate),
                float(q_position),
                float(q_velocity),
            ]
        )
        self.R = np.array(
            [[float(r_input)]],
            dtype=float,
        )

        if self.horizon <= 0:
            raise ValueError("MPC horizon must be positive.")

        if self.output_limit <= 0.0:
            raise ValueError(
                "MPC output limit must be positive."
            )

        if self.wheel_radius <= 0.0:
            raise ValueError(
                "MPC wheel radius must be positive."
            )

        if np.any(np.diag(self.Q) < 0.0):
            raise ValueError(
                "MPC state weights must not be negative."
            )

        if self.R[0, 0] <= 0.0:
            raise ValueError(
                "MPC input weight must be positive."
            )

        self._build_prediction_matrices()
        self._build_solver()
        self.reset()

    def reset(self):
        self.target_position = None
        self.last_solution = np.zeros(
            self.horizon,
            dtype=float,
        )

    def _build_prediction_matrices(self):
        state_size = self.A.shape[0]
        input_size = self.B.shape[1]

        self.state_prediction = np.zeros(
            (
                self.horizon * state_size,
                state_size,
            ),
            dtype=float,
        )
        self.input_prediction = np.zeros(
            (
                self.horizon * state_size,
                self.horizon * input_size,
            ),
            dtype=float,
        )

        for row in range(self.horizon):
            a_power = np.linalg.matrix_power(
                self.A,
                row + 1,
            )

            row_slice = slice(
                row * state_size,
                (row + 1) * state_size,
            )
            self.state_prediction[
                row_slice,
                :,
            ] = a_power

            for column in range(row + 1):
                column_slice = slice(
                    column * input_size,
                    (column + 1) * input_size,
                )
                transition = np.linalg.matrix_power(
                    self.A,
                    row - column,
                )

                self.input_prediction[
                    row_slice,
                    column_slice,
                ] = transition @ self.B

        self.state_cost = sparse.block_diag(
            [self.Q] * self.horizon,
            format="csc",
        )
        self.input_cost = sparse.block_diag(
            [self.R] * self.horizon,
            format="csc",
        )

    def _build_solver(self):
        hessian = (
            self.input_prediction.T
            @ self.state_cost
            @ self.input_prediction
            + self.input_cost
        )
        hessian = 0.5 * (
            hessian + hessian.T
        )

        constraint_matrix = sparse.eye(
            self.horizon,
            format="csc",
        )
        lower_bound = np.full(
            self.horizon,
            -self.output_limit,
            dtype=float,
        )
        upper_bound = np.full(
            self.horizon,
            self.output_limit,
            dtype=float,
        )

        self.solver = osqp.OSQP()
        self.solver.setup(
            P=sparse.csc_matrix(2.0 * hessian),
            q=np.zeros(self.horizon, dtype=float),
            A=constraint_matrix,
            l=lower_bound,
            u=upper_bound,
            verbose=False,
            warm_starting=True,
            polishing=False,
        )

    def _state_from_observation(self, observation):
        observation = np.asarray(
            observation,
            dtype=np.float32,
        )

        if observation.shape != (9,):
            raise ValueError(
                "MPC requires the nine-element IRIS observation."
            )

        if not np.isfinite(observation).all():
            raise ValueError(
                "MPC observation contains invalid values."
            )

        pitch = float(observation[0])
        pitch_rate = float(observation[1])

        average_wheel_velocity = 0.5 * (
            float(observation[2])
            + float(observation[3])
        )
        average_wheel_position = 0.5 * (
            float(observation[7])
            + float(observation[8])
        )

        chassis_position = self.wheel_radius * (
            average_wheel_position + pitch
        )
        chassis_velocity = self.wheel_radius * (
            average_wheel_velocity + pitch_rate
        )

        if self.target_position is None:
            self.target_position = chassis_position

        return np.array(
            [
                pitch,
                pitch_rate,
                chassis_position - self.target_position,
                chassis_velocity,
            ],
            dtype=float,
        )

    def compute(self, observation, dt):
        dt = float(dt)

        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError(
                "Runtime timestep must be finite and greater than zero."
            )

        if not np.isclose(dt, self.dt):
            raise ValueError(
                "Runtime timestep does not match MPC timestep."
            )

        state = self._state_from_observation(
            observation
        )

        free_response = (
            self.state_prediction @ state
        )
        gradient = (
            2.0
            * self.input_prediction.T
            @ self.state_cost
            @ free_response
        )

        self.solver.update(q=gradient)
        self.solver.warm_start(x=self.last_solution)

        result = self.solver.solve(raise_error=False)

        if result.info.status not in (
            "solved",
            "solved inaccurate",
        ):
            raise RuntimeError(
                "MPC optimization failed: "
                f"{result.info.status}"
            )

        if result.x is None:
            raise RuntimeError(
                "MPC optimization returned no solution."
            )

        self.last_solution = np.asarray(
            result.x,
            dtype=float,
        )

        if (
            self.last_solution.shape
            != (self.horizon,)
            or not np.isfinite(
                self.last_solution
            ).all()
        ):
            raise RuntimeError(
                "MPC optimization returned an invalid solution."
            )

        torque = float(
            np.clip(
                self.last_solution[0],
                -self.output_limit,
                self.output_limit,
            )
        )

        return np.array(
            [torque, torque],
            dtype=np.float32,
        )

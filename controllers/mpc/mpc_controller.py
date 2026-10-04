import numpy as np

from controllers.base_controller import BaseController
from controllers.mpc.dynamics_model import linear_model


class MPCController(BaseController):
    def __init__(
        self,
        dt=0.02,
        horizon=20,
        q_pitch=100.0,
        q_pitch_rate=5.0,
        q_position=1.0,
        q_velocity=1.0,
        r_input=0.1,
        output_limit=2.0,
        **_,
    ):
        self.output_limit = float(output_limit)
        self.A, self.B = linear_model(dt)

        self.Q = np.diag(
            [
                q_pitch,
                q_pitch_rate,
                q_position,
                q_velocity,
            ]
        )
        self.R = np.array([[r_input]], dtype=float)
        self.horizon = int(horizon)

    def reset(self):
        pass

    def _first_control_gain(self):
        # Work backward through the finite prediction horizon.
        cost_to_go = self.Q.copy()
        first_gain = None

        for _ in range(self.horizon):
            control_cost = self.R + self.B.T @ cost_to_go @ self.B
            gain = np.linalg.solve(
                control_cost,
                self.B.T @ cost_to_go @ self.A,
            )

            cost_to_go = (
                self.Q
                + self.A.T @ cost_to_go @ self.A
                - self.A.T @ cost_to_go @ self.B @ gain
            )
            first_gain = gain

        return first_gain

    def compute(self, observation, dt):
        pitch = float(observation[0])
        pitch_rate = float(observation[1])
        wheel_velocity = 0.5 * (
            float(observation[2]) + float(observation[3])
        )

        state = np.array(
            [
                pitch,
                pitch_rate,
                0.0,
                wheel_velocity,
            ],
            dtype=float,
        )

        gain = self._first_control_gain()
        torque = float(-(gain @ state)[0])
        torque = np.clip(
            torque,
            -self.output_limit,
            self.output_limit,
        )

        return np.array([torque, torque], dtype=np.float32)

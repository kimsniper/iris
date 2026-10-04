import numpy as np
from controllers.base_controller import BaseController

class PIDController(BaseController):
    def __init__(self, kp, ki, kd, integral_limit=0.5, output_limit=2.0):
        self.kp, self.ki, self.kd = float(kp), float(ki), float(kd)
        self.integral_limit, self.output_limit = float(integral_limit), float(output_limit)
        self.reset()

    def reset(self):
        self.integral = 0.0

    def compute(self, observation, dt):
        pitch, pitch_rate = float(observation[0]), float(observation[1])
        self.integral = np.clip(self.integral + pitch * dt, -self.integral_limit, self.integral_limit)
        torque = -(self.kp * pitch + self.ki * self.integral + self.kd * pitch_rate)
        torque = float(np.clip(torque, -self.output_limit, self.output_limit))
        return np.array([torque, torque], dtype=np.float32)

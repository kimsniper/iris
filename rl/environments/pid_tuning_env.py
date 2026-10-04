import numpy as np
from rl.environments.balancing_env import BalancingEnv
from controllers.pid.pid_controller import PIDController

class PIDTuningEnv(BalancingEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs); self.action_space.low[:] = [0.0]; self.action_space.high[:] = [1.0]
    def gains_from_action(self, action):
        a = np.clip(np.asarray(action, dtype=float), 0.0, 1.0)
        return 80.0*a[0], 0.0, 1.0

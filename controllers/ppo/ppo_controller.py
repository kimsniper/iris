import numpy as np
from stable_baselines3 import PPO
from controllers.base_controller import BaseController

class PPOController(BaseController):
    def __init__(self, model_path):
        self.model = PPO.load(model_path)

    def reset(self): pass

    def compute(self, observation, dt):
        action, _ = self.model.predict(observation, deterministic=True)
        torque = float(np.asarray(action).reshape(-1)[0])
        return np.array([torque, torque], dtype=np.float32)

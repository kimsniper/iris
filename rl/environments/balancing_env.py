from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from simulation.robot_interface import RobotInterface
from simulation.reset_manager import ResetManager
from state_estimation.state_estimator import StateEstimator
from controllers.ppo.observation_builder import build_observation
from rl.common.rewards import balancing_reward

class BalancingEnv(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self, model_path="description/mjcf/two_wheel_robot.xml", frame_skip=10, max_seconds=10.0, max_initial_pitch=0.08, fall_angle=0.70):
        self.model = mujoco.MjModel.from_xml_path(str(Path(model_path)))
        self.data = mujoco.MjData(self.model); self.frame_skip = frame_skip
        self.dt = self.model.opt.timestep * frame_skip; self.max_steps = int(max_seconds / self.dt)
        self.max_initial_pitch = max_initial_pitch; self.fall_angle = fall_angle
        self.robot = RobotInterface(self.model, self.data); self.estimator = StateEstimator()
        self.action_space = spaces.Box(-2.0, 2.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32)
        self.previous_action = 0.0; self.steps = 0

    def _truth_pitch(self):
        quat_wxyz = self.data.qpos[3:7]; quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
        return float(Rotation.from_quat(quat_xyzw).as_euler("xyz")[1])

    def _observation(self):
        gyro, accel = self.robot.imu(); pitch, pitch_rate = self.estimator.update(gyro, accel, self.dt)
        left, right = self.robot.wheel_velocities()
        return build_observation(pitch, pitch_rate, left, right, accel[0], accel[2], self.previous_action)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed); initial = ResetManager.reset(self.model, self.data, self.np_random, self.max_initial_pitch)
        self.estimator.reset(initial); self.previous_action = 0.0; self.steps = 0
        return self._observation(), {}

    def step(self, action):
        torque = float(np.clip(np.asarray(action).reshape(-1)[0], -2.0, 2.0)); self.robot.set_torque([torque, torque])
        for _ in range(self.frame_skip): mujoco.mj_step(self.model, self.data)
        obs = self._observation(); self.steps += 1
        wheel_velocity = 0.5 * (obs[2] + obs[3]); reward = balancing_reward(obs[0], obs[1], wheel_velocity, torque, self.previous_action)
        self.previous_action = torque; terminated = abs(self._truth_pitch()) > self.fall_angle; truncated = self.steps >= self.max_steps
        return obs, float(reward), terminated, truncated, {"truth_pitch": self._truth_pitch()}

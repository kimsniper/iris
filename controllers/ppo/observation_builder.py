import numpy as np

def build_observation(pitch, pitch_rate, left_wheel_velocity, right_wheel_velocity, accel_x, accel_z, previous_action):
    return np.asarray([pitch, pitch_rate, left_wheel_velocity, right_wheel_velocity, accel_x, accel_z, previous_action], dtype=np.float32)

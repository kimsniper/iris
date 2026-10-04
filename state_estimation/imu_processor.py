import numpy as np

class IMUProcessor:
    def __init__(self, gyro_bias=0.0): self.gyro_bias = float(gyro_bias)
    def process(self, gyro, acceleration):
        gyro = np.asarray(gyro, dtype=float).copy(); acceleration = np.asarray(acceleration, dtype=float).copy()
        gyro[1] -= self.gyro_bias
        return gyro, acceleration

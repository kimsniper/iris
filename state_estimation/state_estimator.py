from state_estimation.complementary_filter import ComplementaryFilter

class StateEstimator:
    def __init__(self, alpha=0.98): self.filter = ComplementaryFilter(alpha)
    def reset(self, pitch=0.0): self.filter.reset(pitch)
    def update(self, gyro, acceleration, dt):
        pitch = self.filter.update(gyro[1], acceleration[0], acceleration[2], dt)
        return pitch, float(gyro[1])

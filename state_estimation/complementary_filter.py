import math


class ComplementaryFilter:
    def __init__(self, alpha=0.98, max_accel_correction=0.5):
        self.alpha = float(alpha)
        self.max_accel_correction = float(max_accel_correction)
        self.pitch = 0.0

    def reset(self, pitch=0.0):
        self.pitch = float(pitch)

    @staticmethod
    def _wrap_angle(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def update(self, gyro_y, accel_x, accel_z, dt):
        predicted_pitch = self._wrap_angle(
            self.pitch + gyro_y * dt
        )
        accel_pitch = math.atan2(-accel_x, accel_z)

        correction = self._wrap_angle(
            accel_pitch - predicted_pitch
        )

        # Ignore startup transients and large non-gravity accelerations.
        if abs(correction) > self.max_accel_correction:
            self.pitch = predicted_pitch
            return self.pitch

        self.pitch = self._wrap_angle(
            predicted_pitch + (1.0 - self.alpha) * correction
        )
        return self.pitch

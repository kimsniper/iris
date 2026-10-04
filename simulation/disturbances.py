class DisturbanceConfig:
    def __init__(self, gyro_bias=0.0, motor_scale_left=1.0, motor_scale_right=1.0):
        self.gyro_bias = gyro_bias
        self.motor_scale_left = motor_scale_left
        self.motor_scale_right = motor_scale_right

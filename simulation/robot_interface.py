import numpy as np

class RobotInterface:
    def __init__(self, model, data):
        self.model, self.data = model, data
        self.gyro_adr = model.sensor_adr[model.sensor('imu_gyro').id]
        self.accel_adr = model.sensor_adr[model.sensor('imu_accelerometer').id]
        self.left_vel_adr = model.sensor_adr[model.sensor('left_wheel_velocity').id]
        self.right_vel_adr = model.sensor_adr[model.sensor('right_wheel_velocity').id]

    def imu(self):
        return self.data.sensordata[self.gyro_adr:self.gyro_adr+3].copy(), self.data.sensordata[self.accel_adr:self.accel_adr+3].copy()

    def wheel_velocities(self): return float(self.data.sensordata[self.left_vel_adr]), float(self.data.sensordata[self.right_vel_adr])
    def set_torque(self, action): self.data.ctrl[:] = np.clip(np.asarray(action, dtype=float), self.model.actuator_ctrlrange[:,0], self.model.actuator_ctrlrange[:,1])

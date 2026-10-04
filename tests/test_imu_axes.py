import mujoco

def test_imu_sensor_dimensions():
    m=mujoco.MjModel.from_xml_path('description/mjcf/two_wheel_robot.xml')
    assert m.sensor_dim[m.sensor('imu_gyro').id] == 3
    assert m.sensor_dim[m.sensor('imu_accelerometer').id] == 3

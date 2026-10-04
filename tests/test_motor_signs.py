import mujoco

def test_two_motor_actuators():
    m=mujoco.MjModel.from_xml_path('description/mjcf/two_wheel_robot.xml')
    assert m.nu == 2
    assert m.actuator('left_motor').id != m.actuator('right_motor').id

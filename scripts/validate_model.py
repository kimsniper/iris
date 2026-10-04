from pathlib import Path
import mujoco

path = Path(__file__).resolve().parents[1] / 'description/mjcf/two_wheel_robot.xml'
model = mujoco.MjModel.from_xml_path(str(path))
required = ['left_motor','right_motor','imu_gyro','imu_accelerometer','left_wheel_velocity','right_wheel_velocity']
for name in required:
    if name.endswith('motor'): model.actuator(name)
    else: model.sensor(name)
print(f'Model validation passed: nq={model.nq}, nv={model.nv}, nu={model.nu}, nsensor={model.nsensor}')

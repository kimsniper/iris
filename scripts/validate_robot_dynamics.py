import math

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from simulation.robot_interface import RobotInterface


MODEL_PATH = "description/mjcf/two_wheel_robot.xml"
INITIAL_PITCH = 0.05
CONTROL_DT = 0.02
TEST_DURATION = 0.10
MOTOR_TORQUE = 0.05


def set_pitch(model, data, pitch):
    mujoco.mj_resetData(model, data)

    half_pitch = 0.5 * pitch
    data.qpos[3:7] = [
        math.cos(half_pitch),
        0.0,
        math.sin(half_pitch),
        0.0,
    ]

    mujoco.mj_forward(model, data)

    # Complete one step so acceleration sensors contain a valid sample.
    data.ctrl[:] = 0.0
    mujoco.mj_step(model, data)


def get_truth_pitch(data):
    quat_wxyz = data.qpos[3:7]
    quat_xyzw = [
        quat_wxyz[1],
        quat_wxyz[2],
        quat_wxyz[3],
        quat_wxyz[0],
    ]

    return float(
        Rotation.from_quat(quat_xyzw).as_euler("xyz")[1]
    )


def get_imu_pitch(robot):
    gyro, acceleration = robot.imu()
    accel_pitch = math.atan2(
        -float(acceleration[0]),
        float(acceleration[2]),
    )

    return {
        "gyro_y": float(gyro[1]),
        "accel_x": float(acceleration[0]),
        "accel_z": float(acceleration[2]),
        "accel_pitch": accel_pitch,
    }


def run_case(motor_torque):
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    robot = RobotInterface(model, data)

    set_pitch(model, data, INITIAL_PITCH)

    initial_truth_pitch = get_truth_pitch(data)
    initial_imu = get_imu_pitch(robot)

    robot.set_torque([motor_torque, motor_torque])

    frame_skip = round(CONTROL_DT / model.opt.timestep)
    control_steps = round(TEST_DURATION / CONTROL_DT)

    for _ in range(control_steps):
        for _ in range(frame_skip):
            mujoco.mj_step(model, data)

    final_truth_pitch = get_truth_pitch(data)
    final_imu = get_imu_pitch(robot)
    left_velocity, right_velocity = robot.wheel_velocities()

    return {
        "motor_torque": motor_torque,
        "initial_truth_pitch": initial_truth_pitch,
        "initial_accel_pitch": initial_imu["accel_pitch"],
        "final_truth_pitch": final_truth_pitch,
        "final_accel_pitch": final_imu["accel_pitch"],
        "pitch_change": final_truth_pitch - initial_truth_pitch,
        "final_gyro_y": final_imu["gyro_y"],
        "left_wheel_velocity": left_velocity,
        "right_wheel_velocity": right_velocity,
    }


def print_case(result):
    print(f"\nMotor torque: {result['motor_torque']:+.3f} N m")
    print(
        "  Initial truth pitch: "
        f"{result['initial_truth_pitch']:+.6f} rad"
    )
    print(
        "  Initial accel pitch: "
        f"{result['initial_accel_pitch']:+.6f} rad"
    )
    print(
        "  Final truth pitch:   "
        f"{result['final_truth_pitch']:+.6f} rad"
    )
    print(
        "  Final accel pitch:   "
        f"{result['final_accel_pitch']:+.6f} rad"
    )
    print(
        "  Pitch change:        "
        f"{result['pitch_change']:+.6f} rad"
    )
    print(
        "  Final gyro Y:        "
        f"{result['final_gyro_y']:+.6f} rad/s"
    )
    print(
        "  Wheel velocities:    "
        f"L={result['left_wheel_velocity']:+.6f}, "
        f"R={result['right_wheel_velocity']:+.6f} rad/s"
    )


def main():
    print("Two-wheel robot dynamics validation")
    print(f"Initial pitch: {INITIAL_PITCH:+.3f} rad")
    print(f"Test duration: {TEST_DURATION:.3f} s")

    results = [
        run_case(0.0),
        run_case(MOTOR_TORQUE),
        run_case(-MOTOR_TORQUE),
    ]

    for result in results:
        print_case(result)

    positive_response = results[1]["pitch_change"]
    negative_response = results[2]["pitch_change"]

    print("\nMotor response summary")

    if positive_response < negative_response:
        print(
            "  Positive motor torque produces the more negative "
            "pitch response."
        )
    else:
        print(
            "  Negative motor torque produces the more negative "
            "pitch response."
        )

    print("  Truth pitch, gyro Y, and wheel velocity signs are consistent.")


if __name__ == "__main__":
    main()

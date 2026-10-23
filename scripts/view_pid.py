# Copyright 2026 Mezael Docoy
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import time

import mujoco.viewer
import numpy as np

from controllers.pid.pid_controller import PIDController
from rl.environments.balancing_env import BalancingEnv


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Visualize an IRIS PID controller in MuJoCo."
    )
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--kp", type=float, default=10.0)
    parser.add_argument("--ki", type=float, default=0.0)
    parser.add_argument("--kd", type=float, default=0.2)
    parser.add_argument(
        "--position-kp",
        type=float,
        default=0.002,
    )
    parser.add_argument(
        "--velocity-kd",
        type=float,
        default=0.002,
    )
    parser.add_argument(
        "--pitch-limit",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--output-limit",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=1.0,
        help="Playback multiplier. Use 0.5 for slow motion.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.playback_speed <= 0.0:
        raise ValueError("Playback speed must be greater than zero.")

    env = BalancingEnv()
    observation, _ = env.reset(seed=args.seed)

    controller = PIDController(
        kp=args.kp,
        ki=args.ki,
        kd=args.kd,
        position_kp=args.position_kp,
        velocity_kd=args.velocity_kd,
        max_pitch_reference=args.pitch_limit,
        integral_limit=0.5,
        output_limit=args.output_limit,
    )
    controller.reset()

    terminated = False
    truncated = False
    steps = 0
    saturated_steps = 0

    print("IRIS visual PID replay")
    print(f"  Seed: {args.seed}")
    print(
        f"  Inner gains: "
        f"Kp={args.kp}, Ki={args.ki}, Kd={args.kd}"
    )
    print(
        f"  Outer gains: "
        f"position_kp={args.position_kp}, "
        f"velocity_kd={args.velocity_kd}"
    )
    print(f"  Pitch-reference limit: {args.pitch_limit} rad")
    print(f"  Output limit: {args.output_limit} N m")
    print(f"  Playback speed: {args.playback_speed}x")

    with mujoco.viewer.launch_passive(
        env.model,
        env.data,
    ) as viewer:
        while (
            viewer.is_running()
            and not terminated
            and not truncated
        ):
            cycle_start = time.perf_counter()

            command = controller.compute(
                observation,
                env.dt,
            )
            torque = float(command[0])

            observation, _, terminated, truncated, info = env.step(
                np.array([torque], dtype=np.float32)
            )

            steps += 1

            if abs(torque) >= 0.99 * controller.output_limit:
                saturated_steps += 1

            viewer.sync()

            elapsed = time.perf_counter() - cycle_start
            target_period = env.dt / args.playback_speed
            remaining = target_period - elapsed

            if remaining > 0.0:
                time.sleep(remaining)

            if steps % 25 == 0:
                average_wheel_position = 0.5 * (
                    float(observation[7])
                    + float(observation[8])
                )
                average_wheel_velocity = 0.5 * (
                    float(observation[2])
                    + float(observation[3])
                )

                print(
                    f"t={steps * env.dt:5.2f} s | "
                    f"pitch={info['truth_pitch']:+.3f} rad | "
                    f"estimated={observation[0]:+.3f} rad | "
                    f"torque={torque:+.3f} N m | "
                    f"wheel_pos={average_wheel_position:+.3f} rad | "
                    f"wheel_vel={average_wheel_velocity:+.3f} rad/s"
                )

        saturation_percentage = (
            100.0 * saturated_steps / steps
            if steps > 0
            else 0.0
        )

        if terminated:
            status = "FELL"
        elif truncated:
            status = "COMPLETED"
        else:
            status = "STOPPED"

        print("\nReplay result")
        print(f"  Status: {status}")
        print(f"  Balance time: {steps * env.dt:.2f} s")
        print(f"  Saturation: {saturation_percentage:.2f}%")

        if viewer.is_running():
            print("  Close the MuJoCo window to exit.")

        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)


if __name__ == "__main__":
    main()

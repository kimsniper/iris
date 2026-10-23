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
from stable_baselines3 import PPO

from rl.environments.direct_control_env import DirectControlEnv


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Visualize an IRIS direct PPO policy."
    )
    parser.add_argument(
        "--model",
        default="models/direct_control/best_model.zip",
    )
    parser.add_argument("--seed", type=int, default=11001)
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=1.0,
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.playback_speed <= 0.0:
        raise ValueError(
            "Playback speed must be greater than zero."
        )

    env = DirectControlEnv()
    observation, _ = env.reset(seed=args.seed)
    model = PPO.load(args.model, device="cpu")

    terminated = False
    truncated = False
    steps = 0
    saturated_steps = 0

    print("IRIS visual PPO replay")
    print(f"  Model: {args.model}")
    print(f"  Seed: {args.seed}")
    print(f"  Torque limit: {env.torque_limit} N m")
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

            action, _ = model.predict(
                observation,
                deterministic=True,
            )

            (
                observation,
                _,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            normalized_action = float(
                info["normalized_action"]
            )
            torque = float(info["motor_torque"])
            steps += 1

            if abs(normalized_action) >= 0.99:
                saturated_steps += 1

            viewer.sync()

            elapsed = time.perf_counter() - cycle_start
            target_period = (
                env.dt / args.playback_speed
            )
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
                    f"estimated={float(observation[0]):+.3f} rad | "
                    f"action={normalized_action:+.3f} | "
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
        print(
            f"  Saturation: "
            f"{saturation_percentage:.2f}%"
        )

        if viewer.is_running():
            print("  Close the MuJoCo window to exit.")

        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)


if __name__ == "__main__":
    main()

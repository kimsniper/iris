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
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from rl.environments.direct_control_env import DirectControlEnv


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Plot observation and action telemetry "
            "for an IRIS PPO episode."
        )
    )
    parser.add_argument(
        "--model",
        default="models/direct_control/best_model.zip",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=11001,
    )
    parser.add_argument(
        "--output",
        default="results/ppo/ppo_telemetry.png",
    )
    parser.add_argument(
        "--csv-output",
        default="results/ppo/ppo_telemetry.csv",
    )
    parser.add_argument(
        "--show",
        action="store_true",
    )
    return parser.parse_args()



def collect_episode(model, seed):
    env = DirectControlEnv()

    try:
        observation, _ = env.reset(seed=seed)

        records = []
        terminated = False
        truncated = False
        steps = 0

        while not (terminated or truncated):
            action, _ = model.predict(
                observation,
                deterministic=True,
            )

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            steps += 1

            records.append(
                {
                    "time_s": steps * env.dt,
                    "truth_pitch_rad": float(
                        info["truth_pitch"]
                    ),
                    "estimated_pitch_rad": float(
                        observation[0]
                    ),
                    "pitch_rate_rad_s": float(
                        observation[1]
                    ),
                    "left_wheel_velocity_rad_s": float(
                        observation[2]
                    ),
                    "right_wheel_velocity_rad_s": float(
                        observation[3]
                    ),
                    "accel_x_m_s2": float(
                        observation[4]
                    ),
                    "accel_z_m_s2": float(
                        observation[5]
                    ),
                    "normalized_action": float(
                        info["normalized_action"]
                    ),
                    "left_wheel_position_rad": float(
                        observation[7]
                    ),
                    "right_wheel_position_rad": float(
                        observation[8]
                    ),
                    "motor_torque_nm": float(
                        info["motor_torque"]
                    ),
                    "reward": float(reward),
                }
            )

        return records, {
            "completed": bool(
                truncated and not terminated
            ),
            "fell": bool(terminated),
            "steps": steps,
            "seconds": steps * env.dt,
        }
    finally:
        env.close()

def save_csv(records, output_path):
    if not records:
        raise ValueError(
            "Telemetry records must not be empty."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=records[0].keys(),
        )
        writer.writeheader()
        writer.writerows(records)


def plot_telemetry(
    records,
    output_path,
    seed,
    episode,
):
    if not records:
        raise ValueError(
            "Telemetry records must not be empty."
        )

    time_s = np.asarray(
        [record["time_s"] for record in records]
    )

    truth_pitch = np.asarray(
        [
            record["truth_pitch_rad"]
            for record in records
        ]
    )
    estimated_pitch = np.asarray(
        [
            record["estimated_pitch_rad"]
            for record in records
        ]
    )
    left_wheel_velocity = np.asarray(
        [
            record["left_wheel_velocity_rad_s"]
            for record in records
        ]
    )
    right_wheel_velocity = np.asarray(
        [
            record["right_wheel_velocity_rad_s"]
            for record in records
        ]
    )
    average_wheel_position = 0.5 * (
        np.asarray(
            [
                record["left_wheel_position_rad"]
                for record in records
            ]
        )
        + np.asarray(
            [
                record["right_wheel_position_rad"]
                for record in records
            ]
        )
    )
    normalized_action = np.asarray(
        [
            record["normalized_action"]
            for record in records
        ]
    )
    motor_torque = np.asarray(
        [
            record["motor_torque_nm"]
            for record in records
        ]
    )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(14, 8),
        sharex=True,
    )

    pitch_axis = axes[0, 0]
    pitch_axis.plot(
        time_s,
        truth_pitch,
        label="MuJoCo truth",
        linewidth=2.0,
    )
    pitch_axis.plot(
        time_s,
        estimated_pitch,
        label="Estimated pitch",
        linewidth=1.5,
    )
    pitch_axis.axhline(
        0.0,
        color="black",
        linewidth=0.8,
    )
    pitch_axis.set_title(
        "Body pitch"
    )
    pitch_axis.set_ylabel(
        "Pitch (rad)"
    )
    pitch_axis.legend()
    pitch_axis.grid(True, alpha=0.3)

    velocity_axis = axes[0, 1]
    velocity_axis.plot(
        time_s,
        left_wheel_velocity,
        label="Left wheel",
    )
    velocity_axis.plot(
        time_s,
        right_wheel_velocity,
        label="Right wheel",
    )
    velocity_axis.axhline(
        0.0,
        color="black",
        linewidth=0.8,
    )
    velocity_axis.set_title(
        "Wheel velocity observation"
    )
    velocity_axis.set_ylabel(
        "Velocity (rad/s)"
    )
    velocity_axis.legend()
    velocity_axis.grid(True, alpha=0.3)

    position_axis = axes[1, 0]
    position_axis.plot(
        time_s,
        average_wheel_position,
        linewidth=2.0,
    )
    position_axis.axhline(
        0.0,
        color="black",
        linewidth=0.8,
    )
    position_axis.set_title(
        "Relative wheel position"
    )
    position_axis.set_xlabel(
        "Time (s)"
    )
    position_axis.set_ylabel(
        "Position (rad)"
    )
    position_axis.grid(True, alpha=0.3)

    action_axis = axes[1, 1]
    action_axis.plot(
        time_s,
        normalized_action,
        label="Normalized PPO action",
        linewidth=1.5,
    )
    action_axis.plot(
        time_s,
        motor_torque,
        label="Motor torque (N m)",
        linewidth=1.5,
    )
    action_axis.axhline(
        0.0,
        color="black",
        linewidth=0.8,
    )
    action_axis.set_title(
        "Policy action and motor command"
    )
    action_axis.set_xlabel(
        "Time (s)"
    )
    action_axis.set_ylabel(
        "Action / torque"
    )
    action_axis.legend()
    action_axis.grid(True, alpha=0.3)

    status = (
        "completed"
        if episode["completed"]
        else "fell"
    )

    figure.suptitle(
        "IRIS Direct PPO Telemetry\n"
        f"Seed {seed} | "
        f"{time_s[-1]:.2f} s | "
        f"{status}",
        fontsize=16,
    )
    figure.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    return figure


def main():
    args = parse_arguments()

    model = PPO.load(
        args.model,
        device="cpu",
    )
    records, episode = collect_episode(
        model,
        args.seed,
    )

    image_path = Path(args.output)
    csv_path = Path(args.csv_output)

    save_csv(records, csv_path)
    figure = plot_telemetry(
        records,
        image_path,
        args.seed,
        episode,
    )

    status = (
        "COMPLETED"
        if episode["completed"]
        else "FELL"
    )

    print(f"Episode status: {status}")
    print(f"Episode duration: {episode['seconds']:.2f} s")
    print(f"Telemetry image: {image_path}")
    print(f"Telemetry CSV: {csv_path}")

    if args.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()

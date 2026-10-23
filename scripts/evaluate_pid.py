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

import csv
import json
from pathlib import Path

import numpy as np
import yaml

from controllers.pid.pid_controller import PIDController
from rl.environments.balancing_env import BalancingEnv


PID_CONFIG = Path("controllers/pid/tuned_gains.yaml")
EVALUATION_CONFIG = Path("config/evaluation.yaml")
RESULTS_DIRECTORY = Path("results/pid")


def load_configuration():
    with PID_CONFIG.open() as stream:
        pid_config = yaml.safe_load(stream)["pid"]

    with EVALUATION_CONFIG.open() as stream:
        evaluation_config = yaml.safe_load(stream)["evaluation"]

    return pid_config, evaluation_config


def evaluate_episode(controller, seed):
    env = BalancingEnv()
    observation, _ = env.reset(seed=seed)
    controller.reset()

    initial_x = float(env.data.qpos[0])
    output_limit = controller.output_limit

    pitches = []
    torques = []
    wheel_velocities = []

    terminated = False
    truncated = False
    steps = 0

    while not (terminated or truncated):
        command = controller.compute(observation, env.dt)
        torque = float(command[0])

        observation, _, terminated, truncated, info = env.step(
            np.array([torque], dtype=np.float32)
        )

        pitches.append(float(info["truth_pitch"]))
        torques.append(torque)
        wheel_velocities.append(
            0.5 * (
                float(observation[2])
                + float(observation[3])
            )
        )
        steps += 1

    final_x = float(env.data.qpos[0])
    pitches = np.asarray(pitches, dtype=float)
    torques = np.asarray(torques, dtype=float)
    wheel_velocities = np.asarray(
        wheel_velocities,
        dtype=float,
    )

    saturation_threshold = 0.99 * output_limit
    saturation_percentage = 100.0 * np.mean(
        np.abs(torques) >= saturation_threshold
    )

    return {
        "seed": seed,
        "completed": bool(truncated and not terminated),
        "fell": bool(terminated),
        "balance_seconds": steps * env.dt,
        "rms_pitch_rad": float(
            np.sqrt(np.mean(np.square(pitches)))
        ),
        "peak_abs_pitch_rad": float(
            np.max(np.abs(pitches))
        ),
        "rms_torque_nm": float(
            np.sqrt(np.mean(np.square(torques)))
        ),
        "saturation_percent": float(
            saturation_percentage
        ),
        "rms_wheel_velocity_rad_s": float(
            np.sqrt(
                np.mean(np.square(wheel_velocities))
            )
        ),
        "position_drift_m": abs(final_x - initial_x),
    }


def summarize(results):
    completed = sum(
        result["completed"]
        for result in results
    )

    numeric_fields = [
        "balance_seconds",
        "rms_pitch_rad",
        "peak_abs_pitch_rad",
        "rms_torque_nm",
        "saturation_percent",
        "rms_wheel_velocity_rad_s",
        "position_drift_m",
    ]

    summary = {
        "episodes": len(results),
        "completed_episodes": completed,
        "completion_rate": completed / len(results),
    }

    for field in numeric_fields:
        values = [
            result[field]
            for result in results
        ]
        summary[f"mean_{field}"] = float(
            np.mean(values)
        )
        summary[f"max_{field}"] = float(
            np.max(values)
        )

    return summary


def save_results(results, summary):
    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = RESULTS_DIRECTORY / "nominal_pid.csv"
    json_path = RESULTS_DIRECTORY / "nominal_pid_summary.json"

    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=results[0].keys(),
        )
        writer.writeheader()
        writer.writerows(results)

    with json_path.open("w") as stream:
        json.dump(
            summary,
            stream,
            indent=2,
        )

    return csv_path, json_path


def main():
    pid_config, evaluation_config = load_configuration()

    controller_config = {
        key: value
        for key, value in pid_config.items()
        if key != "source"
    }

    controller = PIDController(**controller_config)
    seeds = evaluation_config["seeds"]

    results = [
        evaluate_episode(controller, seed)
        for seed in seeds
    ]
    summary = summarize(results)
    csv_path, json_path = save_results(
        results,
        summary,
    )

    print("IRIS nominal PID evaluation")
    print(
        f"PID gains: "
        f"Kp={controller.kp}, "
        f"Ki={controller.ki}, "
        f"Kd={controller.kd}"
    )

    for result in results:
        status = (
            "COMPLETED"
            if result["completed"]
            else "FELL"
        )

        print(
            f"  Seed {result['seed']}: "
            f"{status}, "
            f"{result['balance_seconds']:.2f} s, "
            f"RMS pitch={result['rms_pitch_rad']:.4f} rad, "
            f"peak={result['peak_abs_pitch_rad']:.4f} rad"
        )

    print("\nSummary")
    print(
        f"  Completed: "
        f"{summary['completed_episodes']}/"
        f"{summary['episodes']}"
    )
    print(
        f"  Completion rate: "
        f"{100.0 * summary['completion_rate']:.1f}%"
    )
    print(
        f"  Mean balance time: "
        f"{summary['mean_balance_seconds']:.2f} s"
    )
    print(
        f"  Mean RMS pitch: "
        f"{summary['mean_rms_pitch_rad']:.4f} rad"
    )
    print(
        f"  Mean RMS torque: "
        f"{summary['mean_rms_torque_nm']:.4f} N m"
    )
    print(
        f"  Mean saturation: "
        f"{summary['mean_saturation_percent']:.2f}%"
    )
    print(
        f"  Mean position drift: "
        f"{summary['mean_position_drift_m']:.4f} m"
    )
    print(f"\nDetailed results: {csv_path}")
    print(f"Summary results:  {json_path}")


if __name__ == "__main__":
    main()

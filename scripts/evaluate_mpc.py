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

from controllers.mpc.mpc_controller import MPCController
from rl.environments.balancing_env import BalancingEnv


MPC_CONFIG = Path("controllers/mpc/tuned_parameters.yaml")
EVALUATION_CONFIG = Path("config/evaluation.yaml")
RESULTS_DIRECTORY = Path("results/mpc")


def load_configuration():
    with MPC_CONFIG.open() as stream:
        mpc_config = yaml.safe_load(stream)["mpc"]

    with EVALUATION_CONFIG.open() as stream:
        evaluation_config = yaml.safe_load(stream)["evaluation"]

    return mpc_config, evaluation_config


def evaluate_episode(controller, seed):
    env = BalancingEnv()
    observation, _ = env.reset(seed=seed)
    controller.reset()

    initial_x = float(env.data.qpos[0])

    pitches = []
    torques = []

    terminated = False
    truncated = False
    steps = 0

    while not (terminated or truncated):
        command = controller.compute(
            observation,
            env.dt,
        )
        torque = float(command[0])

        observation, _, terminated, truncated, info = env.step(
            np.array([torque], dtype=np.float32)
        )

        pitches.append(float(info["truth_pitch"]))
        torques.append(torque)
        steps += 1

    pitches = np.asarray(pitches, dtype=float)
    torques = np.asarray(torques, dtype=float)

    final_x = float(env.data.qpos[0])

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
            100.0
            * np.mean(
                np.abs(torques)
                >= 0.99 * controller.output_limit
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


def main():
    mpc_config, evaluation_config = load_configuration()

    controller_config = {
        key: value
        for key, value in mpc_config.items()
        if key != "source"
    }
    controller = MPCController(**controller_config)

    results = [
        evaluate_episode(controller, seed)
        for seed in evaluation_config["seeds"]
    ]
    summary = summarize(results)

    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    csv_path = RESULTS_DIRECTORY / "nominal_mpc.csv"
    json_path = RESULTS_DIRECTORY / "nominal_mpc_summary.json"

    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=results[0].keys(),
        )
        writer.writeheader()
        writer.writerows(results)

    with json_path.open("w") as stream:
        json.dump(summary, stream, indent=2)

    print("IRIS nominal MPC evaluation")
    print(
        f"Horizon: {controller.horizon} steps "
        f"({controller.horizon * controller.dt:.2f} s)"
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
            f"RMS torque={result['rms_torque_nm']:.4f} N m, "
            f"drift={result['position_drift_m']:.4f} m"
        )

    print("\nSummary")
    print(
        f"  Completed: "
        f"{summary['completed_episodes']}/"
        f"{summary['episodes']}"
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


if __name__ == "__main__":
    main()

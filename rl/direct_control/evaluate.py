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

import numpy as np
from stable_baselines3 import PPO

from rl.environments.direct_control_env import (
    DirectControlEnv,
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate an IRIS direct PPO policy."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=1001,
    )
    return parser.parse_args()


def evaluate_episode(model, seed):
    env = DirectControlEnv()
    observation, _ = env.reset(seed=seed)

    initial_x = float(env.data.qpos[0])

    pitches = []
    torques = []
    total_return = 0.0
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

        pitches.append(float(info["truth_pitch"]))
        torques.append(float(info["motor_torque"]))
        total_return += float(reward)
        steps += 1

    pitches = np.asarray(pitches, dtype=float)
    torques = np.asarray(torques, dtype=float)
    final_x = float(env.data.qpos[0])

    env.close()

    return {
        "seed": seed,
        "completed": bool(truncated and not terminated),
        "balance_seconds": steps * env.dt,
        "return": total_return,
        "rms_pitch_rad": float(
            np.sqrt(np.mean(np.square(pitches)))
        ),
        "rms_torque_nm": float(
            np.sqrt(np.mean(np.square(torques)))
        ),
        "saturation_percent": float(
            100.0
            * np.mean(np.abs(torques) >= 0.99 * env.torque_limit)
        ),
        "position_drift_m": abs(final_x - initial_x),
    }


def main():
    args = parse_arguments()
    model = PPO.load(
        args.model,
        device="cpu",
    )

    results = [
        evaluate_episode(
            model,
            args.seed_start + episode,
        )
        for episode in range(args.episodes)
    ]

    for result in results:
        status = (
            "COMPLETED"
            if result["completed"]
            else "FELL"
        )

        print(
            f"Seed {result['seed']}: "
            f"{status}, "
            f"{result['balance_seconds']:.2f} s, "
            f"return={result['return']:.2f}, "
            f"RMS pitch={result['rms_pitch_rad']:.4f} rad, "
            f"RMS torque={result['rms_torque_nm']:.4f} N m, "
            f"saturation="
            f"{result['saturation_percent']:.2f}%, "
            f"drift={result['position_drift_m']:.4f} m"
        )

    print("\nSummary")
    print(
        f"  Completed: "
        f"{sum(result['completed'] for result in results)}/"
        f"{len(results)}"
    )
    print(
        f"  Mean balance time: "
        f"{np.mean([r['balance_seconds'] for r in results]):.2f} s"
    )
    print(
        f"  Mean return: "
        f"{np.mean([r['return'] for r in results]):.2f}"
    )
    print(
        f"  Mean RMS pitch: "
        f"{np.mean([r['rms_pitch_rad'] for r in results]):.4f} rad"
    )
    print(
        f"  Mean RMS torque: "
        f"{np.mean([r['rms_torque_nm'] for r in results]):.4f} N m"
    )
    print(
        f"  Mean position drift: "
        f"{np.mean([r['position_drift_m'] for r in results]):.4f} m"
    )


if __name__ == "__main__":
    main()

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
import math
import re
import time
from pathlib import Path

import mujoco.viewer
from stable_baselines3 import PPO

from rl.environments.direct_control_env import DirectControlEnv
from scripts.telemetry_publisher import TelemetryPublisher


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Replay PPO checkpoints in sequence to visualize "
            "learning progress."
        )
    )
    parser.add_argument(
        "--checkpoint-directory",
        default="models/direct_control/seed_0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12001,
    )
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=8.0,
    )
    parser.add_argument(
        "--episodes-per-checkpoint",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--telemetry-host",
        default="127.0.0.1",
    )
    parser.add_argument(
        "--telemetry-port",
        type=int,
        default=8765,
    )
    parser.add_argument(
        "--no-telemetry",
        action="store_true",
    )
    return parser.parse_args()


def checkpoint_step(path):
    match = re.search(
        r"checkpoint_(\d+)_steps",
        path.stem,
    )

    if match:
        return int(match.group(1))

    if path.stem == "best_model":
        return 10**18

    if path.stem == "final_model":
        return 10**18 + 1

    return 10**18 + 2


def find_checkpoints(directory):
    directory = Path(directory)

    checkpoints = [
        path
        for path in directory.glob(
            "checkpoint_*_steps.zip"
        )
        if re.fullmatch(
            r"checkpoint_\d+_steps",
            path.stem,
        )
    ]

    best_model = directory / "best_model.zip"
    final_model = directory / "final_model.zip"

    if best_model.is_file():
        checkpoints.append(best_model)

    if final_model.is_file():
        checkpoints.append(final_model)

    checkpoints = sorted(
        set(checkpoints),
        key=checkpoint_step,
    )

    if not checkpoints:
        raise FileNotFoundError(
            f"No PPO checkpoints found in {directory}."
        )

    return checkpoints


def replay_episode(
    viewer,
    env,
    model,
    seed,
    playback_speed,
    telemetry,
):
    observation, _ = env.reset(seed=seed)

    terminated = False
    truncated = False
    steps = 0
    reward_sum = 0.0

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
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        reward_sum += float(reward)
        steps += 1

        if telemetry is not None:
            telemetry.publish(
                {
                    "type": "sample",
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
                    "average_wheel_position_rad": 0.5
                    * (
                        float(observation[7])
                        + float(observation[8])
                    ),
                    "normalized_action": float(
                        info["normalized_action"]
                    ),
                    "motor_torque_nm": float(
                        info["motor_torque"]
                    ),
                    "reward": float(reward),
                }
            )

        viewer.sync()

        target_period = env.dt / playback_speed
        elapsed = time.perf_counter() - cycle_start
        remaining = target_period - elapsed

        if remaining > 0.0:
            time.sleep(remaining)

    return {
        "completed": bool(truncated and not terminated),
        "steps": steps,
        "seconds": steps * env.dt,
        "return": reward_sum,
    }



def replay_checkpoints(
    args,
    loaded_checkpoints,
    env,
    telemetry,
):
    print(
        f"  Checkpoints: "
        f"{len(loaded_checkpoints)}"
    )
    print(f"  Replay seed: {args.seed}")
    print(f"  Playback speed: {args.playback_speed}x")
    print(
        f"  Episodes per checkpoint: "
        f"{args.episodes_per_checkpoint}"
    )

    with mujoco.viewer.launch_passive(
        env.model,
        env.data,
    ) as viewer:
        for checkpoint_index, (
            checkpoint,
            model,
        ) in enumerate(loaded_checkpoints):
            if not viewer.is_running():
                break

            print(
                f"\n[{checkpoint_index + 1}/"
                f"{len(loaded_checkpoints)}] "
                f"{checkpoint.name}"
            )

            for episode in range(
                args.episodes_per_checkpoint
            ):
                replay_seed = (
                    args.seed + episode
                )

                if telemetry is not None:
                    telemetry.publish(
                        {
                            "type": "episode_start",
                            "checkpoint": checkpoint.name,
                            "checkpoint_index": (
                                checkpoint_index + 1
                            ),
                            "checkpoint_count": len(
                                loaded_checkpoints
                            ),
                            "episode": episode + 1,
                            "seed": replay_seed,
                        }
                    )

                result = replay_episode(
                    viewer=viewer,
                    env=env,
                    model=model,
                    seed=replay_seed,
                    playback_speed=args.playback_speed,
                    telemetry=telemetry,
                )

                status = (
                    "COMPLETED"
                    if result["completed"]
                    else "FELL"
                )

                print(
                    f"  Episode {episode + 1}: "
                    f"{status}, "
                    f"{result['seconds']:.2f} s, "
                    f"return={result['return']:.2f}"
                )

                if telemetry is not None:
                    telemetry.publish(
                        {
                            "type": "episode_end",
                            "checkpoint": checkpoint.name,
                            "status": status,
                            "seconds": result["seconds"],
                            "return": result["return"],
                        }
                    )

                if not viewer.is_running():
                    break

            if (
                viewer.is_running()
                and args.pause_seconds > 0.0
            ):
                time.sleep(args.pause_seconds)

        print("\nReplay finished.")

        if viewer.is_running():
            print(
                "Close the MuJoCo window to exit safely."
            )

        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)


def run(args):

    if (
        not math.isfinite(args.playback_speed)
        or args.playback_speed <= 0.0
    ):
        raise ValueError(
            "Playback speed must be finite "
            "and greater than zero."
        )

    if (
        isinstance(args.episodes_per_checkpoint, bool)
        or args.episodes_per_checkpoint <= 0
    ):
        raise ValueError(
            "Episodes per checkpoint must be positive."
        )

    if (
        not math.isfinite(args.pause_seconds)
        or args.pause_seconds < 0.0
    ):
        raise ValueError(
            "Pause duration must be finite "
            "and not be negative."
        )

    checkpoints = find_checkpoints(
        args.checkpoint_directory
    )

    print("Loading PPO checkpoints...")

    loaded_checkpoints = [
        (
            checkpoint,
            PPO.load(
                checkpoint,
                device="cpu",
            ),
        )
        for checkpoint in checkpoints
    ]

    print("IRIS PPO learning-progress replay")

    env = DirectControlEnv()
    telemetry = None

    try:
        if not args.no_telemetry:
            telemetry = TelemetryPublisher(
                host=args.telemetry_host,
                port=args.telemetry_port,
            )

        replay_checkpoints(
            args,
            loaded_checkpoints,
            env,
            telemetry,
        )
    finally:
        if telemetry is not None:
            telemetry.close()

        env.close()


def main():
    run(parse_arguments())


if __name__ == "__main__":
    main()

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
from pathlib import Path

import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from rl.direct_control.evaluate import evaluate_episode
from scripts.replay_ppo_learning import (
    checkpoint_step,
    find_checkpoints,
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Plot PPO checkpoint performance "
            "through training."
        )
    )
    parser.add_argument(
        "--checkpoint-directory",
        default="models/direct_control_demo/seed_3",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12001,
    )
    parser.add_argument(
        "--output",
        default="results/ppo/ppo_learning_progress.png",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=150000,
    )
    parser.add_argument(
        "--show",
        action="store_true",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    checkpoint_directory = Path(
        args.checkpoint_directory
    )

    if args.max_steps <= 0:
        raise ValueError(
            "Maximum training steps must be positive."
        )

    checkpoints = [
        (
            checkpoint_step(path),
            path,
        )
        for path in find_checkpoints(
            checkpoint_directory
        )
        if checkpoint_step(path) <= args.max_steps
    ]

    if not checkpoints:
        raise FileNotFoundError(
            "No matching checkpoints found."
        )

    training_steps = []
    survival_times = []
    returns = []
    completed = []

    for index, (step, path) in enumerate(
        checkpoints,
        start=1,
    ):
        model = PPO.load(
            path,
            device="cpu",
        )
        result = evaluate_episode(
            model,
            args.seed,
        )

        training_steps.append(step)
        survival_times.append(
            result["balance_seconds"]
        )
        returns.append(result["return"])
        completed.append(
            result["completed"]
        )

        status = (
            "COMPLETED"
            if result["completed"]
            else "FELL"
        )

        print(
            f"[{index}/{len(checkpoints)}] "
            f"{step} steps: "
            f"{status}, "
            f"{result['balance_seconds']:.2f} s, "
            f"return={result['return']:.2f}"
        )

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
    )

    survival_axis = axes[0]
    survival_axis.plot(
        training_steps,
        survival_times,
        marker="o",
        linewidth=2.0,
    )
    survival_axis.axhline(
        10.0,
        color="green",
        linestyle="--",
        label="Full nominal episode",
    )

    for step, time_s, is_completed in zip(
        training_steps,
        survival_times,
        completed,
    ):
        color = (
            "green"
            if is_completed
            else "red"
        )
        survival_axis.scatter(
            [step],
            [time_s],
            color=color,
            zorder=3,
        )

    survival_axis.set_ylabel(
        "Balance time (s)"
    )
    survival_axis.set_title(
        "PPO learning progression"
    )
    survival_axis.set_ylim(
        0.0,
        10.5,
    )
    survival_axis.grid(
        True,
        alpha=0.3,
    )
    survival_axis.legend()

    return_axis = axes[1]
    return_axis.plot(
        training_steps,
        returns,
        marker="o",
        linewidth=2.0,
    )
    return_axis.axhline(
        0.0,
        color="black",
        linewidth=0.8,
    )
    return_axis.set_xlabel(
        "Environment steps"
    )
    return_axis.set_ylabel(
        "Episode return"
    )
    return_axis.grid(
        True,
        alpha=0.3,
    )

    figure.suptitle(
        "IRIS Direct PPO Training Progress\n"
        f"Same replay seed: {args.seed}",
        fontsize=16,
    )
    figure.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    print(f"Saved plot: {output_path}")

    if args.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()

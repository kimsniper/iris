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

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor

from rl.common.direct_control_callback import (
    DirectControlEvaluationCallback,
)
from rl.environments.direct_control_env import (
    DirectControlEnv,
)


CONFIG_PATH = Path("config/ppo.yaml")
MODEL_DIRECTORY = Path("models/direct_control")
LOG_DIRECTORY = Path("logs/direct_control")


def load_configuration():
    with CONFIG_PATH.open() as stream:
        return yaml.safe_load(stream)["ppo"]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train direct PPO control for IRIS."
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    config = load_configuration()

    seed_directory = MODEL_DIRECTORY / f"seed_{args.seed}"
    log_directory = LOG_DIRECTORY / f"seed_{args.seed}"

    seed_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    log_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    env = DirectControlEnv(
        torque_limit=config["torque_limit"],
    )
    env = Monitor(
        env,
        filename=str(log_directory / "monitor.csv"),
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=int(config["checkpoint_frequency"]),
        save_path=str(seed_directory),
        name_prefix="checkpoint",
    )
    evaluation_callback = (
        DirectControlEvaluationCallback(
            evaluation_frequency=int(
                config["evaluation_frequency"]
            ),
            evaluation_seeds=config[
                "evaluation_seeds"
            ],
            output_directory=seed_directory,
            torque_limit=float(
                config["torque_limit"]
            ),
        )
    )
    callbacks = CallbackList(
        [
            checkpoint_callback,
            evaluation_callback,
        ]
    )

    model = PPO(
        policy=config["policy"],
        env=env,
        learning_rate=float(config["learning_rate"]),
        n_steps=int(config["n_steps"]),
        batch_size=int(config["batch_size"]),
        n_epochs=int(config["n_epochs"]),
        gamma=float(config["gamma"]),
        gae_lambda=float(config["gae_lambda"]),
        clip_range=float(config["clip_range"]),
        ent_coef=float(config["ent_coef"]),
        vf_coef=float(config["vf_coef"]),
        max_grad_norm=float(config["max_grad_norm"]),
        policy_kwargs=config["policy_kwargs"],
        verbose=1,
        seed=args.seed,
        device="cpu",
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        progress_bar=False,
    )

    output_path = seed_directory / "final_model"
    model.save(output_path)

    env.close()

    print(f"Saved model: {output_path}.zip")


if __name__ == "__main__":
    main()

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

from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from rl.environments.direct_control_env import DirectControlEnv


class DirectControlEvaluationCallback(BaseCallback):
    def __init__(
        self,
        evaluation_frequency,
        evaluation_seeds,
        output_directory,
        torque_limit,
        verbose=1,
    ):
        super().__init__(verbose=verbose)

        self.evaluation_frequency = int(
            evaluation_frequency
        )
        self.evaluation_seeds = list(
            evaluation_seeds
        )
        self.output_directory = Path(
            output_directory
        )
        self.torque_limit = float(torque_limit)

        self.best_score = None

    def _evaluate_policy(self):
        completed = 0
        balance_times = []
        returns = []

        for seed in self.evaluation_seeds:
            env = DirectControlEnv(
                torque_limit=self.torque_limit,
            )
            observation, _ = env.reset(seed=seed)

            terminated = False
            truncated = False
            steps = 0
            total_return = 0.0

            while not (terminated or truncated):
                action, _ = self.model.predict(
                    observation,
                    deterministic=True,
                )
                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    _,
                ) = env.step(action)

                total_return += float(reward)
                steps += 1

            if truncated and not terminated:
                completed += 1

            balance_times.append(steps * env.dt)
            returns.append(total_return)
            env.close()

        return (
            completed,
            float(np.mean(balance_times)),
            float(np.mean(returns)),
        )

    def _on_step(self):
        if (
            self.n_calls % self.evaluation_frequency
            != 0
        ):
            return True

        score = self._evaluate_policy()

        if self.verbose:
            print(
                "\nDirect PPO validation"
                f"\n  Timesteps: {self.num_timesteps}"
                f"\n  Completed: {score[0]}/"
                f"{len(self.evaluation_seeds)}"
                f"\n  Mean balance time: {score[1]:.2f} s"
                f"\n  Mean return: {score[2]:.2f}"
            )

        if (
            self.best_score is None
            or score > self.best_score
        ):
            self.best_score = score
            output_path = (
                self.output_directory
                / "best_model"
            )
            self.model.save(output_path)

            if self.verbose:
                print(
                    f"  Saved best model: "
                    f"{output_path}.zip"
                )

        return True

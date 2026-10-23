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

def balancing_reward(
    pitch,
    pitch_rate,
    wheel_velocity,
    action,
    previous_action,
):
    return (
        1.0
        - 8.0 * pitch * pitch
        - 0.2 * pitch_rate * pitch_rate
        - 0.02 * wheel_velocity * wheel_velocity
        - 0.01 * action * action
        - 0.02
        * (action - previous_action) ** 2
    )


def direct_control_reward(
    pitch,
    pitch_rate,
    wheel_position,
    wheel_velocity,
    normalized_action,
    previous_action,
    terminated,
):
    reward = (
        1.0
        - 12.0 * pitch * pitch
        - 0.5 * pitch_rate * pitch_rate
        - 0.05 * wheel_position * wheel_position
        - 0.02 * wheel_velocity * wheel_velocity
        - 0.01 * normalized_action * normalized_action
        - 0.02
        * (normalized_action - previous_action) ** 2
    )

    if terminated:
        reward -= 25.0

    return reward

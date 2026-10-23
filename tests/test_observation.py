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

import numpy as np

from controllers.ppo.observation_builder import build_observation


def test_observation_preserves_order_and_type():
    observation = build_observation(
        pitch=1.0,
        pitch_rate=2.0,
        left_wheel_velocity=3.0,
        right_wheel_velocity=4.0,
        accel_x=5.0,
        accel_z=6.0,
        previous_action=7.0,
        left_wheel_position=8.0,
        right_wheel_position=9.0,
    )

    assert observation.shape == (9,)
    assert observation.dtype == np.float32
    assert observation.tolist() == [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
    ]

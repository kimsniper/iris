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

import math

import mujoco


class ResetManager:
    @staticmethod
    def reset(model, data, rng, max_pitch):
        max_pitch = float(max_pitch)

        if not math.isfinite(max_pitch):
            raise ValueError(
                "Maximum reset pitch must be finite."
            )

        if max_pitch < 0.0:
            raise ValueError(
                "Maximum reset pitch must not be negative."
            )

        mujoco.mj_resetData(model, data)

        pitch = float(
            rng.uniform(
                -max_pitch,
                max_pitch,
            )
        )

        if not math.isfinite(pitch):
            raise ValueError(
                "Sampled reset pitch must be finite."
            )

        if abs(pitch) > max_pitch:
            raise ValueError(
                "Sampled reset pitch exceeds the requested range."
            )

        half_pitch = 0.5 * pitch

        # MuJoCo free-joint quaternions use the w, x, y, z order.
        data.qpos[3:7] = [
            math.cos(half_pitch),
            0.0,
            math.sin(half_pitch),
            0.0,
        ]

        mujoco.mj_forward(model, data)
        return pitch

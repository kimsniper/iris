import math

import mujoco


class ResetManager:
    @staticmethod
    def reset(model, data, rng, max_pitch):
        mujoco.mj_resetData(model, data)

        pitch = float(rng.uniform(-max_pitch, max_pitch))
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

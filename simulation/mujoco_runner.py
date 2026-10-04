from pathlib import Path
import mujoco

class MuJoCoRunner:
    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(str(Path(model_path)))
        self.data = mujoco.MjData(self.model)
    def step(self, n=1):
        for _ in range(n): mujoco.mj_step(self.model, self.data)

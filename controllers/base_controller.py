from abc import ABC, abstractmethod
import numpy as np

class BaseController(ABC):
    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def compute(self, observation: np.ndarray, dt: float) -> np.ndarray: ...

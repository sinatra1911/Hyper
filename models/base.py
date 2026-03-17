from abc import ABC, abstractmethod
import numpy as np

class BaseDetector(ABC):
    """
    Abstract Base Class for all anomaly detectors.
    Ensures a unified plug-and-play architecture.
    """
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def detect(self, cube: np.ndarray) -> np.ndarray:
        """
        Takes a 3D hyperspectral cube (H, W, B).
        Must return a 2D normalized numpy array (H, W) bounded [0, 1].
        """
        pass
from abc import ABC, abstractmethod
from ...types import ImageArray
from models import FaceLandmarks

class Aligner(ABC):
    """Abstract interface for face alignment."""
    
    @abstractmethod
    def align(self, image: ImageArray, landmarks: FaceLandmarks) -> ImageArray:
        """
        Takes the original image and the detected facial landmarks,
        and returns a cropped, transformed, and perfectly aligned face array.
        """
        raise NotImplementedError
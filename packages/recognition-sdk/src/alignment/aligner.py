from abc import ABC, abstractmethod
from sdk_types import ImageArray
from models import FaceLandmarks, AlignedFace


class Aligner(ABC):
    """Abstract interface for all face alignment models."""
    
    @abstractmethod
    def align(self, image: ImageArray, landmarks: FaceLandmarks) -> AlignedFace:
        """
        Takes the original image and the detected facial landmarks,
        and returns a wrapped AlignedFace model containing the transformed image.
        """
        raise NotImplementedError
from abc import ABC, abstractmethod
from ..types import ImageArray
from models import DetectionResult
from config import DetectorConfig
from enums import DetectorType

class Detector(ABC):
    """Abstract interface for all face detection models."""

    def __init__(self, config: DetectorConfig):
        self.config = config

    @property
    @abstractmethod
    def detector_type(self) -> DetectorType:
        """Returns the specific type of detector."""
        raise NotImplementedError

    @abstractmethod
    def detect(self, image: ImageArray) -> DetectionResult:
        """Detect faces in an image and return a structured DetectionResult."""
        raise NotImplementedError
from abc import ABC, abstractmethod

from models import AlignedFace, EmbeddingResult
from config import EmbedderConfig
from enums import EmbedderType

class Embedder(ABC):
    """Abstract interface for all face embedding models."""

    def __init__(self, config: EmbedderConfig):
        self.config = config

    @property
    @abstractmethod
    def embedder_type(self) -> EmbedderType:
        """Returns the specific type of embedder."""
        raise NotImplementedError

    @abstractmethod
    def embed(self, face: AlignedFace) -> EmbeddingResult:
        """Extracts a feature vector from a strictly aligned face."""
        raise NotImplementedError
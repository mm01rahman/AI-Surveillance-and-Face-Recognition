from abc import ABC, abstractmethod
from collections.abc import Sequence
from models import GalleryIdentity

class GalleryRepository(ABC):
    """Abstract interface for persistent gallery storage."""
    
    @abstractmethod
    def save_all(self, identities: Sequence[GalleryIdentity]) -> None:
        """Persists a collection of identities to storage."""
        raise NotImplementedError
        
    @abstractmethod
    def load_all(self) -> Sequence[GalleryIdentity]:
        """Loads all identities from storage."""
        raise NotImplementedError
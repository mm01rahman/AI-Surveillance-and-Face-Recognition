from abc import ABC, abstractmethod
import numpy as np
from sdk_types import PersonID

class SearchBackend(ABC):
    """Abstract interface for all vector storage and search backends."""
    
    @abstractmethod
    def add(self, person_id: PersonID, vectors: np.ndarray) -> None:
        """Adds vectors to the storage backend under the given PersonID."""
        raise NotImplementedError
        
    @abstractmethod
    def remove(self, person_id: PersonID) -> None:
        """Removes all vectors associated with the given PersonID."""
        raise NotImplementedError
        
    @abstractmethod
    def clear(self) -> None:
        """Empties the entire index."""
        raise NotImplementedError
        
    @abstractmethod
    def count(self) -> int:
        """Returns the total number of vectors in the index."""
        raise NotImplementedError

    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int) -> tuple[np.ndarray, list[PersonID | None]]:
        """Returns raw distances and IDs for the nearest neighbors."""
        raise NotImplementedError
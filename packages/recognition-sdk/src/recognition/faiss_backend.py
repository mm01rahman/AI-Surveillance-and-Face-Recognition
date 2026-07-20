import faiss
import numpy as np

from enums import DistanceMetric
from exceptions import MatchingError
from sdk_types import PersonID
from .backend import SearchBackend

class FAISSBackend(SearchBackend):
    """Local, in-memory vector storage using FAISS."""
    
    def __init__(self, dimension: int, metric: DistanceMetric):
        self.dimension = dimension
        
        # Explicitly declare the generic base type so Mypy accepts both branches
        base_index: faiss.Index
        
        # Inner Product equates to Cosine Similarity when vectors are L2 normalized
        if metric == DistanceMetric.COSINE:
            base_index = faiss.IndexFlatIP(dimension)
        else:
            base_index = faiss.IndexFlatL2(dimension)
            
        # IndexIDMap allows mapping custom integer IDs
        self.index = faiss.IndexIDMap(base_index)
        
        self._id_counter = 0
        self._int_to_str: dict[int, PersonID] = {}
        self._str_to_ints: dict[PersonID, set[int]] = {}

    def add(self, person_id: PersonID, vectors: np.ndarray) -> None:
        num_vectors = vectors.shape[0]
        ids = np.arange(self._id_counter, self._id_counter + num_vectors, dtype=np.int64)
        
        # Update mappings
        if person_id not in self._str_to_ints:
            self._str_to_ints[person_id] = set()
            
        for int_id in ids:
            # Convert the numpy.int64 scalar to a native Python int
            native_id = int(int_id)
            
            self._int_to_str[native_id] = person_id
            self._str_to_ints[person_id].add(native_id)
            
        try:
            self.index.add_with_ids(vectors, ids)
        except Exception as e:
            raise MatchingError(f"Failed to add vectors to FAISS index: {e}")
            
        self._id_counter += num_vectors

    def remove(self, person_id: PersonID) -> None:
        if person_id not in self._str_to_ints:
            return
            
        # Extract the integer IDs tied to this person
        ids_to_remove = list(self._str_to_ints[person_id])
        id_array = np.array(ids_to_remove, dtype=np.int64)
        
        try:
            # 1. Wrap the numpy array in a FAISS IDSelector
            sel = faiss.IDSelectorBatch(id_array)
            
            # 2. Pass the selector to remove_ids
            self.index.remove_ids(sel)
        except Exception as e:
            raise MatchingError(f"Failed to remove vectors from FAISS index: {e}")
            
        # Cleanup mappings
        for int_id in ids_to_remove:
            del self._int_to_str[int_id]
        del self._str_to_ints[person_id]
        
        
    def clear(self) -> None:
        self.index.reset()
        self._int_to_str.clear()
        self._str_to_ints.clear()
        self._id_counter = 0

    def count(self) -> int:
        return int(self.index.ntotal)

    def search(self, query_vector: np.ndarray, top_k: int) -> tuple[np.ndarray, list[PersonID | None]]:
        if self.count() == 0:
            return np.array([[]]), []
            
        try:
            distances, indices = self.index.search(query_vector, top_k)
        except Exception as e:
            raise MatchingError(f"FAISS search failed: {e}")
        
        # FAISS returns -1 if there aren't enough results to fill top_k
        mapped_ids = [self._int_to_str[int(idx)] if idx != -1 else None for idx in indices[0]]
                
        return distances, mapped_ids
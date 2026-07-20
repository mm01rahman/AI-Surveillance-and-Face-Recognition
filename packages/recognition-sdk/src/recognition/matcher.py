import time
import numpy as np

from models import Embedding, Match, MatchingResult, GalleryIdentity
from config import MatcherConfig
from sdk_types import PersonID

from .backend import SearchBackend
from .metrics import compute_similarity
from .threshold import is_match


class IdentityMatcher:
    """High-level API for identity resolution and gallery management."""
    
    def __init__(self, config: MatcherConfig, backend: SearchBackend):
        self.config = config
        self.backend = backend

    # --- Gallery Management ---
    
    def add(self, identity: GalleryIdentity) -> None:
        if not identity.face_embeddings:
            return
        vectors = np.array([emb.vector for emb in identity.face_embeddings], dtype=np.float32)
        self.backend.add(identity.person_id, vectors)
        
    def remove(self, person_id: PersonID) -> None:
        self.backend.remove(person_id)
        
    def clear(self) -> None:
        self.backend.clear()
        
    def count(self) -> int:
        return self.backend.count()

    # --- Search ---
    
    def search(self, embedding: Embedding) -> MatchingResult:
        start_time = time.perf_counter()
        
        # Backends expect 2D arrays (batch, dimension)
        query_vector = np.expand_dims(embedding.vector, axis=0)
        distances, person_ids = self.backend.search(query_vector, self.config.top_k)
        
        matches = []
        if len(person_ids) > 0:
            for dist, p_id in zip(distances[0], person_ids):
                if p_id is None:
                    continue
                    
                similarity = compute_similarity(float(dist), self.config.metric)
                
                if is_match(similarity, self.config):
                    matches.append(Match(
                        person_id=p_id,
                        similarity=similarity,
                        distance=float(dist)
                    ))
                    
        # Explicitly sort by similarity descending, never assume backend ordering
        matches.sort(key=lambda m: m.similarity, reverse=True)
        
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        
        return MatchingResult(
            top_matches=matches,
            processing_time_ms=duration_ms
        )
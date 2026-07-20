import numpy as np

from ..types import PersonID
from models import GalleryIdentity
from config import RepositoryConfig
from exceptions import RepositoryError

from .repository import GalleryRepository
from .gallery import GalleryRegistry
from recognition.backend import SearchBackend
class EnrollmentManager:
    """Orchestrates gallery CRUD operations with transactional rollback semantics."""
    
    def __init__(self, config: RepositoryConfig, repository: GalleryRepository, search_backend: SearchBackend):
        self.config = config
        self.repository = repository
        self.search_backend = search_backend
        self.registry = GalleryRegistry()

    def load_gallery(self) -> None:
        self.registry.clear()
        self.search_backend.clear()
        
        identities = self.repository.load_all()
        for identity in identities:
            self.registry.add(identity)
            
            if identity.face_embeddings:
                vectors = np.array([emb.vector for emb in identity.face_embeddings], dtype=np.float32)
                self.search_backend.add(identity.person_id, vectors)

    def save_gallery(self) -> None:
        self.repository.save_all(self.registry.get_all())

    def upsert(self, identity: GalleryIdentity) -> None:
        """Enrolls a new identity or cleanly replaces an existing one (Transaction)."""
        is_update = self.registry.exists(identity.person_id)
        backup_identity = self.registry.get(identity.person_id) if is_update else None

        try:
            # 1. Clear old vectors from the index if updating
            if is_update:
                self.search_backend.remove(identity.person_id)
                
            # 2. Insert new vectors to the index
            if identity.face_embeddings:
                vectors = np.array([emb.vector for emb in identity.face_embeddings], dtype=np.float32)
                self.search_backend.add(identity.person_id, vectors)
                
            # 3. Update memory state
            self.registry.add(identity)
            
            # 4. Attempt persistence
            if self.config.auto_save:
                self.save_gallery()
                
        except Exception as e:
            self._rollback_upsert(identity.person_id, backup_identity)
            raise RepositoryError(f"Enrollment transaction failed, state rolled back. Cause: {e}")

    def _rollback_upsert(self, person_id: PersonID, backup_identity: GalleryIdentity | None) -> None:
        """Internal safeguard to restore previous state after a failed transaction."""
        try:
            self.search_backend.remove(person_id)
            if backup_identity and backup_identity.face_embeddings:
                vectors = np.array([emb.vector for emb in backup_identity.face_embeddings], dtype=np.float32)
                self.search_backend.add(person_id, vectors)
        except Exception:
            pass  # Swallow secondary errors to avoid masking the primary crash
            
        if backup_identity:
            self.registry.add(backup_identity)
        else:
            self.registry.remove(person_id)

    def delete_identity(self, person_id: PersonID) -> None:
        # Deletion isn't easily rolled back from disk, so order dictates safety:
        # Backend -> Memory -> Disk
        self.search_backend.remove(person_id)
        self.registry.remove(person_id)
        
        if self.config.auto_save:
            self.save_gallery()

    def get_identity(self, person_id: PersonID) -> GalleryIdentity | None:
        return self.registry.get(person_id)
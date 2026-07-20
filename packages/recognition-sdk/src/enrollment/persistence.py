import os
import json
from collections.abc import Sequence

from models import GalleryIdentity
from config import RepositoryConfig
from exceptions import RepositoryError
from enrollment.repository import GalleryRepository
from enrollment.serializer import identity_to_dict, dict_to_identity

class JSONGalleryRepository(GalleryRepository):
    """A local file-based repository using atomic JSON writes."""
    
    def __init__(self, config: RepositoryConfig):
        self.gallery_path = config.gallery_path
        
        if not self.gallery_path.parent.exists():
            self.gallery_path.parent.mkdir(parents=True, exist_ok=True)

    def save_all(self, identities: Sequence[GalleryIdentity]) -> None:
        temp_path = self.gallery_path.with_suffix('.tmp')
        
        try:
            data = [identity_to_dict(identity) for identity in identities]
            
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
                f.flush()
                os.fsync(f.fileno())  # Force OS to write buffers to disk
                
            # Atomic rename (overwrites existing file safely)
            temp_path.replace(self.gallery_path)
            
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise RepositoryError(f"Failed to atomically save gallery: {e}")

    def load_all(self) -> Sequence[GalleryIdentity]:
        if not self.gallery_path.exists():
            return []
            
        try:
            with open(self.gallery_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return [dict_to_identity(item) for item in data]
        except Exception as e:
            raise RepositoryError(f"Failed to load gallery from {self.gallery_path}: {e}")
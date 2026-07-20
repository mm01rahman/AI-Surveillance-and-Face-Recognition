import numpy as np
from typing import Any

from models import GalleryIdentity, Embedding
from exceptions import RepositoryError

def identity_to_dict(identity: GalleryIdentity) -> dict[str, Any]:
    """Serializes a GalleryIdentity into a JSON-safe dictionary."""
    return {
        "person_id": identity.person_id,
        "metadata": identity.metadata,
        "face_embeddings": [
            {
                "vector": emb.vector.tolist(),
                "normalized": emb.normalized
            } 
            for emb in identity.face_embeddings
        ]
    }

def dict_to_identity(data: dict[str, Any]) -> GalleryIdentity:
    """Deserializes and validates a dictionary into a GalleryIdentity."""
    if not isinstance(data, dict):
        raise RepositoryError("Identity record must be a JSON object.")
        
    if "person_id" not in data or "face_embeddings" not in data:
        raise RepositoryError("Identity record missing required keys: 'person_id' or 'face_embeddings'.")
        
    if not isinstance(data["face_embeddings"], list):
        raise RepositoryError("'face_embeddings' must be a list.")

    try:
        embeddings = [
            Embedding(
                vector=np.array(emb_data["vector"], dtype=np.float32),
                normalized=bool(emb_data["normalized"])
            ) 
            for emb_data in data["face_embeddings"]
        ]
        
        return GalleryIdentity(
            person_id=str(data["person_id"]),
            face_embeddings=embeddings,
            metadata=data.get("metadata", {}) if isinstance(data.get("metadata"), dict) else {}
        )
    except Exception as e:
        raise RepositoryError(f"Data validation failed during deserialization: {e}")
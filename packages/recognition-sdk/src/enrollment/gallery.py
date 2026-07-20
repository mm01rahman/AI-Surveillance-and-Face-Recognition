from collections.abc import Sequence
from ..types import PersonID
from models import GalleryIdentity

class GalleryRegistry:
    """In-memory state manager for Gallery Identities."""
    
    def __init__(self):
        self._identities: dict[PersonID, GalleryIdentity] = {}

    def add(self, identity: GalleryIdentity) -> None:
        self._identities[identity.person_id] = identity

    def remove(self, person_id: PersonID) -> None:
        self._identities.pop(person_id, None)

    def get(self, person_id: PersonID) -> GalleryIdentity | None:
        return self._identities.get(person_id)
        
    def exists(self, person_id: PersonID) -> bool:
        """Explicitly checks for the presence of an identity."""
        return person_id in self._identities

    def get_all(self) -> Sequence[GalleryIdentity]:
        return list(self._identities.values())

    def clear(self) -> None:
        self._identities.clear()
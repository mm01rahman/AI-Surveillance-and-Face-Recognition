from dataclasses import dataclass, field
from collections.abc import Sequence
from typing import Optional, Tuple

from .types import (
    EmbeddingVector,
    ImageArray,
    LandmarkPoint,
    PersonID,
    Metadata
)
from .enums import RecognitionStatus, QualityStatus, DetectorType, EmbedderType


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class FaceLandmarks:
    left_eye: LandmarkPoint
    right_eye: LandmarkPoint
    nose: LandmarkPoint
    left_mouth: LandmarkPoint
    right_mouth: LandmarkPoint
    
    
@dataclass
class AlignedFace:
    image: ImageArray
    landmarks: FaceLandmarks
    # Future-proofed for: transform_matrix, original_bbox, quality_score, etc.


@dataclass
class DetectedFace:
    bounding_box: BoundingBox
    landmarks: FaceLandmarks
    quality_score: float = 0.0
    quality_status: QualityStatus = QualityStatus.PASS


@dataclass
class FaceCrop:
    image: ImageArray
    bounding_box: BoundingBox
    landmarks: FaceLandmarks


@dataclass(eq=False)
class Embedding:
    vector: EmbeddingVector
    normalized: bool = True
    
    @property
    def dimension(self) -> int:
        return len(self.vector)


@dataclass
class Match:
    person_id: PersonID
    similarity: float
    distance: float


# --- Stage-Specific Result Wrappers ---

@dataclass
class DetectionResult:
    faces: Sequence[DetectedFace]
    detector: DetectorType
    processing_time_ms: float = 0.0
    image_size: Tuple[int, int] = (0, 0)


@dataclass
class EmbeddingResult:
    embedding: Embedding
    processing_time_ms: float = 0.0


@dataclass
class MatchingResult:
    top_matches: Sequence[Match]
    processing_time_ms: float = 0.0


@dataclass
class RecognitionResult:
    status: RecognitionStatus
    detection: Optional[DetectionResult] = None
    embedding: Optional[EmbeddingResult] = None
    matching: Optional[MatchingResult] = None
    total_processing_time_ms: float = 0.0

    @property
    def best_match(self) -> Optional[Match]:
        if self.matching and self.matching.top_matches:
            return self.matching.top_matches[0]
        return None

    @property
    def detected_faces(self) -> Sequence[DetectedFace]:
        return self.detection.faces if self.detection else []

    @property
    def final_embedding(self) -> Optional[Embedding]:
        return self.embedding.embedding if self.embedding else None


@dataclass
class GalleryIdentity:
    person_id: PersonID
    face_embeddings: Sequence[Embedding]
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True)
class SDKInfo:
    version: str
    detector: DetectorType
    embedder: EmbedderType
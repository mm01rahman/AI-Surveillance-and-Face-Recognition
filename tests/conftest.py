from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from config import DetectorConfig, EmbedderConfig, MatcherConfig, QualityConfig, RecognitionConfig, RepositoryConfig
from detection.detector import Detector
from enums import DetectorType, QualityStatus
from exceptions import DetectionError
from models import AlignedFace, BoundingBox, DetectedFace, DetectionResult, Embedding, EmbeddingResult, FaceLandmarks, MatchingResult, Match
from pipeline.sdk import RecognitionSDK


@pytest.fixture
def sample_image() -> np.ndarray:
    return np.full((160, 160, 3), 127, dtype=np.uint8)


@pytest.fixture
def sample_face_image(sample_image: np.ndarray) -> np.ndarray:
    return sample_image


@pytest.fixture
def sample_landmarks() -> FaceLandmarks:
    return FaceLandmarks(
        left_eye=(55.0, 65.0),
        right_eye=(105.0, 65.0),
        nose=(80.0, 90.0),
        left_mouth=(60.0, 115.0),
        right_mouth=(100.0, 115.0),
    )


@pytest.fixture
def detected_face(sample_landmarks: FaceLandmarks) -> DetectedFace:
    return DetectedFace(
        bounding_box=BoundingBox(x1=40, y1=40, x2=120, y2=130, confidence=0.99),
        landmarks=sample_landmarks,
        quality_score=1.0,
        quality_status=QualityStatus.PASS,
    )


class DummyDetector(Detector):
    @property
    def detector_type(self) -> DetectorType:
        return DetectorType.SCRFD

    def detect(self, image: np.ndarray) -> DetectionResult:
        if not hasattr(image, "shape") or len(image.shape) != 3:
            raise DetectionError("Invalid image shape")
        return DetectionResult(
            faces=[DetectedFace(BoundingBox(40, 40, 120, 130, 0.99), FaceLandmarks((55, 65), (105, 65), (80, 90), (60, 115), (100, 115)))],
            detector=self.detector_type,
            processing_time_ms=1.0,
            image_size=(image.shape[1], image.shape[0]),
        )


@pytest.fixture
def valid_detector() -> Detector:
    return DummyDetector(DetectorConfig(model_path=Path("dummy.onnx")))


@pytest.fixture
def base_config(tmp_path: Path) -> RecognitionConfig:
    return RecognitionConfig(
        detector=DetectorConfig(model_path=Path("detector.onnx")),
        embedder=EmbedderConfig(model_path=Path("embedder.onnx"), embedding_size=4),
        matcher=MatcherConfig(threshold=0.6, top_k=1),
        quality=QualityConfig(),
        repository=RepositoryConfig(gallery_path=tmp_path / "gallery.json", index_path=tmp_path / "index.faiss", auto_save=False),
    )


@pytest.fixture
def mock_detector(detected_face: DetectedFace) -> MagicMock:
    detector = MagicMock()
    detector.detect.return_value = DetectionResult(faces=[detected_face], detector=DetectorType.SCRFD, processing_time_ms=1.0, image_size=(160, 160))
    return detector


@pytest.fixture
def mock_aligner(sample_image: np.ndarray, sample_landmarks: FaceLandmarks) -> MagicMock:
    aligner = MagicMock()
    aligner.align.return_value = AlignedFace(image=np.zeros((112, 112, 3), dtype=np.uint8), landmarks=sample_landmarks)
    return aligner


@pytest.fixture
def mock_embedder() -> MagicMock:
    embedder = MagicMock()
    embedder.embed.return_value = EmbeddingResult(embedding=Embedding(vector=np.ones(4, dtype=np.float32), normalized=True), processing_time_ms=1.0)
    return embedder


class InMemoryMatcher:
    def __init__(self):
        self.person_ids: list[str] = []

    def search(self, embedding: Embedding) -> MatchingResult:
        if not self.person_ids:
            return MatchingResult(top_matches=[], processing_time_ms=1.0)
        return MatchingResult(top_matches=[Match(person_id=self.person_ids[-1], similarity=1.0, distance=0.0)], processing_time_ms=1.0)


class InMemoryEnrollment:
    def __init__(self, matcher: InMemoryMatcher):
        self.matcher = matcher

    def upsert(self, identity):
        self.matcher.person_ids.append(identity.person_id)

    def delete_identity(self, person_id: str) -> None:
        self.matcher.person_ids = [pid for pid in self.matcher.person_ids if pid != person_id]

    def load_gallery(self) -> None:
        pass

    def save_gallery(self) -> None:
        pass


@pytest.fixture
def real_sdk(base_config: RecognitionConfig, mock_detector: MagicMock, mock_aligner: MagicMock, mock_embedder: MagicMock) -> RecognitionSDK:
    matcher = InMemoryMatcher()
    return RecognitionSDK(
        config=base_config,
        detector=mock_detector,
        aligner=mock_aligner,
        embedder=mock_embedder,
        matcher=matcher,
        enrollment=InMemoryEnrollment(matcher),
    )

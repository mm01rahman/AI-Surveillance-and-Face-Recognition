from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from .enums import Device, DistanceMetric, DetectorType, EmbedderType
from .exceptions import ConfigurationError


@dataclass(frozen=True)
class DetectorConfig:
    model_path: Path
    input_size: Tuple[int, int] = (640, 640)
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    device: Device = Device.AUTO
    detector_type: DetectorType = DetectorType.SCRFD
    
    def __post_init__(self):
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ConfigurationError("confidence_threshold must be between 0.0 and 1.0")
        if not (0.0 <= self.nms_threshold <= 1.0):
            raise ConfigurationError("nms_threshold must be between 0.0 and 1.0")


@dataclass(frozen=True)
class EmbedderConfig:
    model_path: Path
    device: Device = Device.AUTO
    embedding_size: int = 512
    normalize: bool = True
    embedder_type: EmbedderType = EmbedderType.ARCFACE
    
    def __post_init__(self):
        if self.embedding_size <= 0:
            raise ConfigurationError("embedding_size must be greater than 0")


@dataclass(frozen=True)
class MatcherConfig:
    metric: DistanceMetric = DistanceMetric.COSINE
    threshold: float = 0.6
    top_k: int = 1
    
    def __post_init__(self):
        if self.top_k < 1:
            raise ConfigurationError("top_k must be at least 1")
        if self.threshold < 0.0:
            raise ConfigurationError("threshold cannot be negative")


@dataclass(frozen=True)
class QualityConfig:
    minimum_face_size: int = 40 
    minimum_sharpness: float = 0.0
    minimum_pose_score: float = 0.0
    
    def __post_init__(self):
        if self.minimum_face_size <= 0:
            raise ConfigurationError("minimum_face_size must be greater than 0")


@dataclass(frozen=True)
class RepositoryConfig:
    gallery_path: Path
    index_path: Path
    auto_save: bool = True
    
    def __post_init__(self):
        # Fail fast if the parent directory doesn't even exist
        if not self.gallery_path.parent.exists():
             raise ConfigurationError(f"Gallery parent directory does not exist: {self.gallery_path.parent}")


@dataclass(frozen=True)
class RecognitionConfig:
    detector: DetectorConfig
    embedder: EmbedderConfig
    matcher: MatcherConfig
    quality: QualityConfig
    repository: RepositoryConfig
from enum import Enum

class Device(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    AUTO = "auto"

class DistanceMetric(str, Enum):
    COSINE = "cosine"
    L2 = "l2"

class ImageFormat(str, Enum):
    BGR = "bgr"
    RGB = "rgb"

class RecognitionStatus(str, Enum):
    MATCH = "match"
    UNKNOWN = "unknown"
    NO_FACE = "no_face"
    MULTIPLE_FACES = "multiple_faces"

class QualityStatus(str, Enum):
    PASS = "pass"
    TOO_SMALL = "too_small"
    BLURRY = "blurry"
    BAD_POSE = "bad_pose"

class DetectorType(str, Enum):
    SCRFD = "scrfd"

class EmbedderType(str, Enum):
    ARCFACE = "arcface"
    
class EnrollmentStrategy(str, Enum):
    LARGEST = "largest"
    HIGHEST_CONFIDENCE = "highest_confidence"
    SINGLE_ONLY = "single_only"
    
class ModelStatus(str, Enum):
    LOADED = "loaded"
    NOT_LOADED = "not_loaded"
    FAILED = "failed"
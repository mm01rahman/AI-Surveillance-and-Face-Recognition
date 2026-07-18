class RecognitionError(Exception):
    """Base exception for all Recognition SDK errors."""
    pass

class ConfigurationError(RecognitionError):
    """Raised when the SDK or a component is configured incorrectly."""
    pass

class DetectionError(RecognitionError):
    """Raised when face detection fails or encounters a runtime error."""
    pass

class AlignmentError(RecognitionError):
    """Raised when face alignment (landmark transformation) fails."""
    pass

class EmbeddingError(RecognitionError):
    """Raised when feature extraction (embedding) fails."""
    pass

class MatchingError(RecognitionError):
    """Raised when the matching process or index search fails."""
    pass

class RepositoryError(RecognitionError):
    """Raised when gallery storage, index I/O, or retrieval operations fail."""
    pass
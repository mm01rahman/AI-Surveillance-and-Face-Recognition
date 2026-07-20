import pytest
import numpy as np
from models import DetectionResult
from exceptions import DetectionError
from detection.detector import Detector

def test_detector_contract_returns_correct_model(valid_detector: Detector, sample_image: np.ndarray):
    """Any detector must return a strictly typed DetectionResult."""
    result = valid_detector.detect(sample_image)
    
    assert isinstance(result, DetectionResult)
    assert isinstance(result.faces, list)
    assert isinstance(result.processing_time_ms, float)

def test_detector_contract_raises_sdk_error_on_bad_input(valid_detector: Detector):
    """Any detector must trap 3rd-party errors and raise DetectionError."""
    bad_image = np.zeros((10, 10), dtype=np.uint8)  # Invalid shape/channels
    
    with pytest.raises(DetectionError):
        valid_detector.detect(bad_image)
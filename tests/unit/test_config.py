import pytest
from pydantic import ValidationError
from config import MatcherConfig

def test_matcher_config_rejects_invalid_threshold():
    # Thresholds must be between 0.0 and 1.0
    with pytest.raises(ValidationError):
        MatcherConfig(threshold=1.5)
        
    with pytest.raises(ValidationError):
        MatcherConfig(threshold=-0.1)
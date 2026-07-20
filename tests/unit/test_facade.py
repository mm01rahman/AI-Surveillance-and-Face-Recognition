import pytest
from unittest.mock import MagicMock
from pipeline.sdk import RecognitionSDK
from enums import RecognitionStatus
from models import MatchingResult

def test_recognize_routes_to_unknown_when_no_match(
    base_config, mock_detector, mock_aligner, mock_embedder
):
    # 1. Setup the Matcher mock to return an empty match list
    mock_matcher = MagicMock()
    mock_matcher.search.return_value = MatchingResult(
        top_matches=[], 
        processing_time_ms=1.5
    )
    
    # 2. Inject mocks into the SDK
    sdk = RecognitionSDK(
        config=base_config,
        detector=mock_detector,
        aligner=mock_aligner,
        embedder=mock_embedder,
        matcher=mock_matcher
    )
    
    # 3. Execute
    dummy_image = MagicMock()
    result = sdk.recognize(dummy_image)
    
    # 4. Assert Facade logic correctly translates empty matches to UNKNOWN
    assert result.status == RecognitionStatus.UNKNOWN
    
    # Prove to Pylance that the matcher actually ran and returned a result
    assert result.matching is not None
    
    assert result.matching.top_matches == []
import pytest
from pipeline.sdk import RecognitionSDK
from enums import RecognitionStatus

def test_end_to_end_enrollment_and_recognition(real_sdk: RecognitionSDK, sample_face_image):
    person_id = "test_user_01"
    
    # 1. State Check: Identity is unknown
    initial_result = real_sdk.recognize(sample_face_image)
    assert initial_result.status == RecognitionStatus.UNKNOWN
    
    # 2. Command: Enroll Identity
    real_sdk.enroll(image=sample_face_image, person_id=person_id)
    
    # 3. Query: Recognize Identity
    match_result = real_sdk.recognize(sample_face_image)
    
    # 4. Assert: Correct resolution
    assert match_result.status == RecognitionStatus.MATCH
    
    # Prove to Pylance that the matching result exists
    assert match_result.matching is not None 
    assert len(match_result.matching.top_matches) > 0 # Safe practice for arrays
    
    # Pylance now knows this is safe
    top_match = match_result.matching.top_matches[0]
    
    assert top_match.person_id == person_id
    assert top_match.similarity > 0.98
    
    # 5. Command: Delete Identity
    real_sdk.delete_identity(person_id)
    
    # 6. State Check: Identity is unknown again
    final_result = real_sdk.recognize(sample_face_image)
    assert final_result.status == RecognitionStatus.UNKNOWN
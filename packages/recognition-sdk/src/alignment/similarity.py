import cv2
import numpy as np
from typing import cast

from ..types import ImageArray
from exceptions import AlignmentError

def apply_similarity_transform(
    image: ImageArray, 
    source_landmarks: np.ndarray, 
    reference_landmarks: np.ndarray, 
    output_size: tuple[int, int]
) -> ImageArray:
    """Calculates and applies an affine similarity transformation."""
    
    # Never trust upstream modules blindly
    if source_landmarks.shape != (5, 2):
        raise AlignmentError(f"Expected shape (5, 2) for landmarks, got {source_landmarks.shape}")
    
    tform, _ = cv2.estimateAffinePartial2D(source_landmarks, reference_landmarks, method=cv2.LMEDS)

    if tform is None:
        raise AlignmentError("Failed to calculate affine transformation matrix.")

    try:
        aligned_face = cv2.warpAffine(
            image,
            tform,
            output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0
        )
    except Exception as e:
        raise AlignmentError(f"Failed to warp image: {e}")

    return cast(ImageArray, aligned_face)
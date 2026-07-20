import cv2
import numpy as np
from typing import cast  

from ...types import ImageArray
from models import FaceLandmarks
from exceptions import AlignmentError
from aligner import Aligner


# The strict pixel coordinates ArcFace expects for a 112x112 image.
# Do not alter these; they are mathematically tied to the ArcFace weights.
ARCFACE_REFERENCE_112 = np.array([
    [38.2946, 51.6963],  # Left eye
    [73.5318, 51.5014],  # Right eye
    [56.0252, 71.7366],  # Nose
    [41.5493, 92.3655],  # Left mouth
    [70.7299, 92.2041]   # Right mouth
], dtype=np.float32)


class ArcFaceAligner(Aligner):
    def __init__(self, output_size: tuple[int, int] = (112, 112)):
        self.output_size = output_size
        self.reference_landmarks = ARCFACE_REFERENCE_112

    def align(self, image: ImageArray, landmarks: FaceLandmarks) -> ImageArray:
        """Aligns the face using an Affine Similarity Transformation."""
        
        # 1. Convert our strict dataclass into a raw numpy array for OpenCV
        src_pts = np.array([
            landmarks.left_eye,
            landmarks.right_eye,
            landmarks.nose,
            landmarks.left_mouth,
            landmarks.right_mouth
        ], dtype=np.float32)

        # 2. Calculate the similarity transformation matrix.
        # cv2.LMEDS is a robust method that handles minor landmark noise well.
        tform, _ = cv2.estimateAffinePartial2D(
            src_pts, 
            self.reference_landmarks, 
            method=cv2.LMEDS
        )

        if tform is None:
            raise AlignmentError("Failed to calculate affine transformation matrix.")

        # 3. Apply the transformation to the original image
        try:
            aligned_face = cv2.warpAffine(
                image,
                tform,
                self.output_size,
                borderValue=0.0  # Pad out-of-bounds areas with black
            )
        except Exception as e:
            raise AlignmentError(f"Failed to warp image: {e}")

        # 4. Cast the OpenCV return type to satisfy Pylance's strict checks
        return cast(ImageArray, aligned_face)
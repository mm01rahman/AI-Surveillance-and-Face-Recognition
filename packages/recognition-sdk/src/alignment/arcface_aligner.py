import numpy as np
from dataclasses import dataclass, field

from ...types import ImageArray
from models import FaceLandmarks, AlignedFace
from aligner import Aligner
from alignment.template import get_reference_landmarks
from alignment.similarity import apply_similarity_transform

@dataclass(slots=True)
class ArcFaceAligner(Aligner):
    output_size: tuple[int, int] = (112, 112)
    reference_landmarks: np.ndarray = field(init=False)
    
    def __post_init__(self):
        # Cache the dynamically scaled template at initialization
        self.reference_landmarks = get_reference_landmarks(self.output_size)

    def align(self, image: ImageArray, landmarks: FaceLandmarks) -> AlignedFace:
        """Aligns the face for ArcFace inference."""
        
        source_landmarks = np.array([
            landmarks.left_eye,
            landmarks.right_eye,
            landmarks.nose,
            landmarks.left_mouth,
            landmarks.right_mouth
        ], dtype=np.float32)

        aligned_image = apply_similarity_transform(
            image, 
            source_landmarks, 
            self.reference_landmarks, 
            self.output_size
        )
        
        return AlignedFace(
            image=aligned_image, 
            landmarks=landmarks
        )
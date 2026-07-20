import numpy as np

# Base ArcFace coordinates for a 112x112 image
_ARCFACE_BASE_112 = np.array([
    [38.2946, 51.6963],  # Left eye
    [73.5318, 51.5014],  # Right eye
    [56.0252, 71.7366],  # Nose
    [41.5493, 92.3655],  # Left mouth
    [70.7299, 92.2041]   # Right mouth
], dtype=np.float32)


def get_reference_landmarks(output_size: tuple[int, int] = (112, 112)) -> np.ndarray:
    """Scales the 112x112 template to the target output size."""
    width, height = output_size
    
    # Calculate scale ratios relative to the 112x112 base
    scale_x = width / 112.0
    scale_y = height / 112.0
    
    scaled_landmarks = _ARCFACE_BASE_112.copy()
    scaled_landmarks[:, 0] *= scale_x
    scaled_landmarks[:, 1] *= scale_y
    
    return scaled_landmarks
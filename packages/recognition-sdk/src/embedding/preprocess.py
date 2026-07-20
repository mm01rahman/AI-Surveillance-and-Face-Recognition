import numpy as np
from ..types import ImageArray
from exceptions import EmbeddingError

def prepare_input_tensor(image: ImageArray, expected_size: tuple[int, int] = (112, 112)) -> np.ndarray:
    """Prepares an aligned face image for ArcFace ONNX inference."""
    
    # 1. Validate Shape
    if image.shape[:2] != expected_size:
        raise EmbeddingError(
            f"Invalid input shape for embedding. Expected {expected_size}, got {image.shape[:2]}"
        )
    
    # 2. BGR to RGB
    img_rgb = image[..., ::-1]
    
    # 3. Convert to float32 and standardize to [-1.0, 1.0]
    # ArcFace networks are typically trained with this specific normalization
    tensor = (img_rgb.astype(np.float32) - 127.5) / 127.5
    
    # 4. HWC to CHW (Channel, Height, Width)
    tensor = tensor.transpose(2, 0, 1)
    
    # 5. Add Batch Dimension (1, C, H, W)
    tensor = np.expand_dims(tensor, axis=0)
    
    return tensor
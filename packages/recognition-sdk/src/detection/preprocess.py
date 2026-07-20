import cv2
from dataclasses import dataclass
from typing import cast 

from ..sdk_types import ImageArray

@dataclass(frozen=True)
class LetterboxMetadata:
    scale: float
    pad_x: int
    pad_y: int

def letterbox_image(
    image: ImageArray, 
    target_size: tuple[int, int], 
    color: tuple[int, int, int] = (0, 0, 0)
) -> tuple[ImageArray, LetterboxMetadata]:
    """Resizes an image to the target size while maintaining aspect ratio."""
    shape = image.shape[:2]
    target_w, target_h = target_size
    
    r = min(target_w / shape[1], target_h / shape[0])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    
    dw: float = (target_w - new_unpad[0]) / 2.0
    dh: float = (target_h - new_unpad[1]) / 2.0
    
    if shape[::-1] != new_unpad:
        # Cast the OpenCV return type to satisfy Pylance
        image = cast(ImageArray, cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR))
        
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    padded_image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    
    # Cast the final output to satisfy the function's return signature
    padded_image = cast(ImageArray, padded_image)
    
    return padded_image, LetterboxMetadata(scale=r, pad_x=int(dw), pad_y=int(dh))
from typing import Any, Dict, Tuple
import numpy as np
from numpy.typing import NDArray

# Image and feature representations
ImageArray = NDArray[np.uint8]
EmbeddingVector = NDArray[np.float32]

# Spatial coordinates
BoundingBoxTuple = Tuple[float, float, float, float]
LandmarkPoint = Tuple[float, float]
LandmarkArray = NDArray[np.float32]  # Expected shape: (5, 2)

# Domain types
PersonID = str
GalleryID = str
TrackID = int
CameraID = str
Metadata = Dict[str, Any]
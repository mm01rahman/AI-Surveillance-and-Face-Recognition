import numpy as np

def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """Applies L2 normalization to a feature vector."""
    norm = np.linalg.norm(vector)
    
    if norm == 0:
        return vector
        
    return vector / norm
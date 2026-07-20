from enums import DistanceMetric

def compute_similarity(distance: float, metric: DistanceMetric) -> float:
    """Converts a raw backend distance/score into a bounded [0.0, 1.0] similarity."""
    if metric == DistanceMetric.COSINE:
        # FAISS Inner Product for normalized vectors ranges [-1.0, 1.0]
        return float((distance + 1.0) / 2.0)
        
    elif metric == DistanceMetric.L2:
        # FAISS L2 distance for normalized vectors ranges [0.0, 4.0]
        return float(1.0 / (1.0 + distance))
        
    return 0.0
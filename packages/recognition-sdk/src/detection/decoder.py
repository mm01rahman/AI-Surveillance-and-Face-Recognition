import numpy as np

class SCRFDDecoder:
    """Decodes raw ONNX outputs from the SCRFD network."""
    
    @staticmethod
    def decode(outputs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Assuming the ONNX graph includes the decode node for this example
        boxes = outputs[0]
        scores = outputs[1]
        landmarks = outputs[2]
        return boxes, scores, landmarks
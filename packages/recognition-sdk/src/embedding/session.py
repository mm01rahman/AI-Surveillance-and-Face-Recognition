import onnxruntime as ort
from config import EmbedderConfig
from enums import Device
from exceptions import EmbeddingError

def create_ort_session(config: EmbedderConfig) -> tuple[ort.InferenceSession, str]:
    """Initializes and returns an ONNX Runtime session and its input name."""
    providers = ['CPUExecutionProvider']
    if config.device in (Device.CUDA, Device.AUTO):
        providers.insert(0, 'CUDAExecutionProvider')
        
    try:
        session = ort.InferenceSession(str(config.model_path), providers=providers)
        input_name = session.get_inputs()[0].name
        return session, input_name
    except Exception as e:
        raise EmbeddingError(f"Failed to initialize Embedder session: {e}")
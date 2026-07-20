import time
import numpy as np
from typing import cast

from models import AlignedFace, Embedding, EmbeddingResult
from config import EmbedderConfig
from enums import EmbedderType
from exceptions import EmbeddingError

from embedder import Embedder
from session import create_ort_session
from preprocess import prepare_input_tensor
from normalize import l2_normalize


class ArcFaceEmbedder(Embedder):
    def __init__(self, config: EmbedderConfig):
        super().__init__(config)
        self.session, self.input_name = create_ort_session(config)

    @property
    def embedder_type(self) -> EmbedderType:
        return EmbedderType.ARCFACE

    def embed(self, face: AlignedFace) -> EmbeddingResult:
        start_time = time.perf_counter()
        
        # 1. Prepare Tensor
        input_tensor = prepare_input_tensor(face.image)
        
        # 2. Run ONNX Inference
        try:
            outputs = cast(
                list[np.ndarray],
                self.session.run(None, {self.input_name: input_tensor})
            )
        except Exception as e:
            raise EmbeddingError(f"ONNX embedding inference failed: {e}")
            
        # Extract the raw feature vector (flattening the batch dimension)
        raw_vector = outputs[0].flatten()
        
        # 3. Validate dimension
        if len(raw_vector) != self.config.embedding_size:
            raise EmbeddingError(
                f"Model returned vector of size {len(raw_vector)}, "
                f"but config expected {self.config.embedding_size}."
            )
        
        # 4. Normalize (if configured)
        is_normalized = False
        if self.config.normalize:
            raw_vector = l2_normalize(raw_vector)
            is_normalized = True
            
        # 5. Package Result
        embedding_model = Embedding(
            vector=raw_vector,
            normalized=is_normalized
        )
        
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        
        return EmbeddingResult(
            embedding=embedding_model,
            processing_time_ms=duration_ms
        )
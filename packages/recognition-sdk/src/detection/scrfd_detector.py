import time
import numpy as np
import onnxruntime as ort
from typing import cast
from ..sdk_types import ImageArray
from models import DetectionResult, DetectedFace
from config import DetectorConfig, QualityConfig
from enums import DetectorType, Device, QualityStatus
from exceptions import DetectionError

from detection.detector import Detector
from .preprocess import letterbox_image
from detection.postprocess import nms, map_and_clip_coordinates
from detection.decoder import SCRFDDecoder


class SCRFDDetector(Detector):
    def __init__(self, config: DetectorConfig, quality_config: QualityConfig | None = None):
        super().__init__(config)
        self.quality_config = quality_config
        self.session, self.input_name = self._create_session()

    @property
    def detector_type(self) -> DetectorType:
        return DetectorType.SCRFD

    def _create_session(self) -> tuple[ort.InferenceSession, str]:
        providers = ['CPUExecutionProvider']
        if self.config.device in (Device.CUDA, Device.AUTO):
            providers.insert(0, 'CUDAExecutionProvider')
            
        try:
            session = ort.InferenceSession(str(self.config.model_path), providers=providers)
            input_name = session.get_inputs()[0].name
            return session, input_name
        except Exception as e:
            raise DetectionError(f"Failed to initialize SCRFD session: {e}")

    def detect(self, image: ImageArray) -> DetectionResult:
        start_time = time.perf_counter()
        img_h, img_w = image.shape[:2]
        
        # 1. Preprocess
        padded_img, meta = letterbox_image(image, self.config.input_size)
        input_tensor = padded_img[..., ::-1].transpose(2, 0, 1).astype(np.float32)
        input_tensor = (input_tensor - 127.5) / 128.0
        input_tensor = np.expand_dims(input_tensor, axis=0)

        # 2. Inference
        try:
            outputs = cast(
                list[np.ndarray], 
                self.session.run(None, {self.input_name: input_tensor})
            )
        except Exception as e:
            raise DetectionError(f"ONNX inference failed: {e}")
            
        # 3. Decode & Filter
        raw_boxes, raw_scores, raw_landmarks = SCRFDDecoder.decode(outputs)
        
        valid_idx = np.where(raw_scores >= self.config.confidence_threshold)[0]
        boxes, scores, landmarks = raw_boxes[valid_idx], raw_scores[valid_idx], raw_landmarks[valid_idx]
        
        keep_idx = nms(boxes, scores, self.config.nms_threshold)
        boxes, scores, landmarks = boxes[keep_idx], scores[keep_idx], landmarks[keep_idx]
        
        # 4. Map, Clip, and Package
        faces: list[DetectedFace] = []
        for box, score, kpss in zip(boxes, scores, landmarks):
            bbox_model, landmark_model = map_and_clip_coordinates(
                box, kpss, score, meta, img_w, img_h
            )
            
            # Note: We would route through assess_face_quality() here if self.quality_config is set
            
            faces.append(
                DetectedFace(
                    bounding_box=bbox_model,
                    landmarks=landmark_model,
                    quality_score=1.0, 
                    quality_status=QualityStatus.PASS
                )
            )

        duration_ms = (time.perf_counter() - start_time) * 1000.0

        return DetectionResult(
            faces=faces,
            detector=self.detector_type,
            processing_time_ms=duration_ms,
            image_size=(img_w, img_h)
        )
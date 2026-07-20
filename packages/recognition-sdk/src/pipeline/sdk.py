import time
from typing import Any, Optional

from ..types import ImageArray, PersonID
from config import RecognitionConfig
from models import RecognitionResult, GalleryIdentity, DetectedFace, EmbeddingResult
from enums import RecognitionStatus, EnrollmentStrategy
from exceptions import RecognitionError

# Orchestrators
from detection.detector import Detector
from detection.scrfd_detector import SCRFDDetector
from alignment.aligner import Aligner
from alignment.arcface_aligner import ArcFaceAligner
from embedding.embedder import Embedder
from embedding.arcface_embedder import ArcFaceEmbedder
from recognition.backend import SearchBackend
from recognition.faiss_backend import FAISSBackend
from recognition.matcher import IdentityMatcher
from enrollment.persistence import JSONGalleryRepository
from enrollment.enrollment import EnrollmentManager


class RecognitionSDK:
    """
    The unified Facade for the Recognition SDK.
    Provides a stable, high-level API for recognition and gallery management.
    """
    
    def __init__(
        self, 
        config: RecognitionConfig,
        # Dependency Injection (Defaults provided if None)
        detector: Optional[Detector] = None,
        aligner: Optional[Aligner] = None,
        embedder: Optional[Embedder] = None,
        search_backend: Optional[SearchBackend] = None,
        matcher: Optional[IdentityMatcher] = None,
        enrollment: Optional[EnrollmentManager] = None
    ):
        self._config = config
        
        # 1. Initialize core AI modules (Use injected or default to ArcFace/SCRFD)
        self._detector = detector or SCRFDDetector(config.detector)
        self._aligner = aligner or ArcFaceAligner() 
        self._embedder = embedder or ArcFaceEmbedder(config.embedder)
        
        # 2. Initialize storage and matching
        self._search_backend = search_backend or FAISSBackend(
            dimension=config.embedder.embedding_size,
            metric=config.matcher.metric
        )
        self._matcher = matcher or IdentityMatcher(config.matcher, self._search_backend)
        
        # 3. Initialize enrollment and persistence
        repository = JSONGalleryRepository(config.repository)
        self._enrollment = enrollment or EnrollmentManager(config.repository, repository, self._search_backend)
        
        # Lazy loading based on configuration (avoids locking up on init for massive galleries)
        if getattr(config.repository, 'auto_load', False):
            self._enrollment.load_gallery()

    # --- Internal Pipeline Helpers ---

    def _process_face_pipeline(self, image: ImageArray, face: DetectedFace) -> EmbeddingResult:
        """DRY helper: Handles the Align -> Embed sequence."""
        aligned_face = self._aligner.align(image, face.landmarks)
        return self._embedder.embed(aligned_face)

    def _extract_target_face(self, faces: list[DetectedFace], strategy: EnrollmentStrategy) -> DetectedFace:
        """Applies business logic to select the correct face for enrollment."""
        if strategy == EnrollmentStrategy.SINGLE_ONLY and len(faces) > 1:
            raise RecognitionError("Multiple faces detected, but strategy requires SINGLE_ONLY.")
            
        if strategy == EnrollmentStrategy.HIGHEST_CONFIDENCE:
            return max(faces, key=lambda f: f.bounding_box.confidence)
            
        # Default to LARGEST
        return max(faces, key=lambda f: f.bounding_box.area)

    # --- High-Level Recognition API ---

    def recognize(self, image: ImageArray) -> RecognitionResult:
        """Runs the end-to-end recognition pipeline on an image."""
        start_time = time.perf_counter()
        
        # 1. Detection
        det_result = self._detector.detect(image)
        
        if not det_result.faces:
            return RecognitionResult(
                status=RecognitionStatus.NO_FACE,
                detection=det_result,
                total_processing_time_ms=(time.perf_counter() - start_time) * 1000.0
            )
            
        if len(det_result.faces) > 1:
            return RecognitionResult(
                status=RecognitionStatus.MULTIPLE_FACES,
                detection=det_result,
                total_processing_time_ms=(time.perf_counter() - start_time) * 1000.0
            )
            
        # 2 & 3. Align and Embed (using helper)
        emb_result = self._process_face_pipeline(image, det_result.faces[0])
        
        # 4. Matching
        match_result = self._matcher.search(emb_result.embedding)
        
        # 5. Status Resolution
        status = RecognitionStatus.MATCH if match_result.top_matches else RecognitionStatus.UNKNOWN
        
        return RecognitionResult(
            status=status,
            detection=det_result,
            embedding=emb_result,
            matching=match_result,
            total_processing_time_ms=(time.perf_counter() - start_time) * 1000.0
        )

    # --- High-Level Enrollment API ---

    def enroll(
        self, 
        image: ImageArray, 
        person_id: PersonID, 
        metadata: dict[str, Any] | None = None,
        strategy: EnrollmentStrategy = EnrollmentStrategy.LARGEST
    ) -> GalleryIdentity:
        """Extracts an embedding from an image and upserts it into the gallery."""
        det_result = self._detector.detect(image)
        
        if not det_result.faces:
            raise RecognitionError(f"Cannot enroll '{person_id}': No face detected.")
            
        # 1. Select Target Face based on strategy
        target_face = self._extract_target_face(list(det_result.faces), strategy)
        
        # 2 & 3. Align and Embed
        emb_result = self._process_face_pipeline(image, target_face)
        
        identity = GalleryIdentity(
            person_id=person_id,
            face_embeddings=[emb_result.embedding],
            metadata=metadata or {}
        )
        
        # 4. Upsert (Handles transaction and disk save)
        self._enrollment.upsert(identity)
        return identity

    # --- Gallery Management ---
    
    def delete_identity(self, person_id: PersonID) -> None:
        self._enrollment.delete_identity(person_id)
        
    def load_gallery(self) -> None:
        self._enrollment.load_gallery()
        
    def save_gallery(self) -> None:
        self._enrollment.save_gallery()

    # --- Advanced Access ---
    
    @property
    def detector(self) -> Detector: return self._detector
    
    @property
    def aligner(self) -> Aligner: return self._aligner
    
    @property
    def embedder(self) -> Embedder: return self._embedder
    
    @property
    def matcher(self) -> IdentityMatcher: return self._matcher
    
    @property
    def gallery(self) -> EnrollmentManager: return self._enrollment
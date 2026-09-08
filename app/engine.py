import cv2
import numpy as np
from fastapi import HTTPException, status
from insightface.app import FaceAnalysis

from app.config import get_settings

settings = get_settings()


class BiometricEngine:
    def __init__(self):
        self._app: FaceAnalysis | None = None

    def initialize(self):
        """Pre-loads the Buffalo_l model weights into memory at startup."""
        self._app = FaceAnalysis(
            name=settings.INSIGHTFACE_MODEL_NAME,
            providers=[settings.EXECUTION_PROVIDER],
        )
        self._app.prepare(
            ctx_id=0,
            det_size=(settings.DETECTION_SIZE, settings.DETECTION_SIZE),
        )

    @property
    def is_ready(self) -> bool:
        return self._app is not None

    def _decode_image(self, file_bytes: bytes) -> np.ndarray:
        if len(file_bytes) > settings.MAX_IMAGE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Image exceeds {settings.MAX_IMAGE_SIZE_MB}MB size limit.",
            )

        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Unable to decode image. Ensure valid JPEG or PNG format.",
            )
        return img

    def extract_embedding(self, file_bytes: bytes) -> np.ndarray:
        """Runs face detection and returns the normalized 512-D embedding."""
        if not self._app:
            raise RuntimeError("Biometric engine has not been initialized.")

        img = self._decode_image(file_bytes)
        faces = self._app.get(img)

        if len(faces) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="NO_FACE_DETECTED: Keep face within frame and check lighting.",
            )
        if len(faces) > 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="MULTIPLE_FACES_DETECTED: Exactly one individual must be present.",
            )

        face = faces[0]

        # Enforce minimum detection confidence
        if getattr(face, "det_score", 0.0) < 0.65:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="FACE_LOW_CONFIDENCE: Image is too blurry or dark for verification.",
            )

        return face.normed_embedding

    def combine_embeddings(
        self, center_bytes: bytes, left_bytes: bytes, right_bytes: bytes
    ) -> np.ndarray:
        """Averages 3 angles (Center, Left 30°, Right 30°) and re-normalizes."""
        v_center = self.extract_embedding(center_bytes)
        v_left = self.extract_embedding(left_bytes)
        v_right = self.extract_embedding(right_bytes)

        master = (v_center + v_left + v_right) / 3.0
        norm = np.linalg.norm(master)
        if norm == 0:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to normalize composite facial vector.",
            )
        return master / norm

    @staticmethod
    def compute_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculates cosine similarity between two unit vectors."""
        return float(np.dot(v1, v2))


engine = BiometricEngine()
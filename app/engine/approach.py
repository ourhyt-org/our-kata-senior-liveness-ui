from typing import Any, Dict, List, Tuple

import boto3
import cv2
import numpy as np

from app.utils.images import debug_image_info

s3 = boto3.client("s3")


class ApproachEngine:
    def __init__(self) -> None:
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def load_frame_from_s3(self, bucket: str, key: str):
        try:
            obj = s3.get_object(Bucket=bucket, Key=key)
            data = obj["Body"].read()
            arr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            print(f"❌ Error cargando frame s3://{bucket}/{key}: {e}")
            return None

    def _detect_face(self, gray) -> Tuple[int, int, int, int] | None:
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(80, 80),
        )
        if len(faces) == 0:
            return None
        return faces[0]

    def _evaluate_approach(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        if len(frames) < 2:
            return {
                "livenessScore": 0.0,
                "passed": False,
                "reason": "Se requieren al menos 2 frames para APPROACH.",
                "stats": {"framesCount": len(frames)},
            }

        f1 = frames[0]
        f2 = frames[-1]

        try:
            gray1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            return {
                "livenessScore": 0.0,
                "passed": False,
                "reason": f"Error convirtiendo a escala de grises: {str(e)}",
                "stats": {"framesCount": len(frames)},
            }

        face1 = self._detect_face(gray1)
        face2 = self._detect_face(gray2)

        if face1 is None or face2 is None:
            return {
                "livenessScore": 0.0,
                "passed": False,
                "reason": "No se detectó rostro en uno de los frames.",
                "stats": {
                    "framesCount": len(frames),
                    "face1Detected": face1 is not None,
                    "face2Detected": face2 is not None,
                },
            }

        (x1, y1, w1, h1) = face1
        (x2, y2, w2, h2) = face2

        area1 = float(w1 * h1)
        area2 = float(w2 * h2)

        if area1 <= 0:
            return {
                "livenessScore": 0.0,
                "passed": False,
                "reason": "Área de rostro inicial inválida.",
                "stats": {
                    "framesCount": len(frames),
                    "face1Area": area1,
                    "face2Area": area2,
                },
            }

        scale_change = area2 / area1

        THRESHOLD_SCALE = 1.15
        passed = scale_change > THRESHOLD_SCALE

        raw_score = (scale_change - 1.0) / 0.8
        liveness_score = max(0.0, min(1.0, round(raw_score, 3)))

        reason = None
        if not passed:
            reason = (
                f"Cambio muy bajo entre frames (scale={scale_change:.2f}). "
                "Acércate más a la cámara y repite el reto."
            )

        return {
            "livenessScore": liveness_score,
            "passed": passed,
            "reason": reason,
            "stats": {
                "framesCount": len(frames),
                "face1Area": area1,
                "face2Area": area2,
                "scaleChange": scale_change,
                "firstFrameIndex": 0,
                "lastFrameIndex": len(frames) - 1,
            },
        }

    def run(self, bucket: str, frame_keys: List[str]) -> Dict[str, Any]:
        frames: List[np.ndarray] = []

        use_keys = frame_keys[:2]

        for key in use_keys:
            img = self.load_frame_from_s3(bucket, key)
            if img is None:
                continue
            print(
                f"✅ Frame cargado (APPROACH): s3://{bucket}/{key}, info={debug_image_info(img)}"
            )
            frames.append(img)

        if len(frames) < 2:
            return {
                "livenessScore": 0.0,
                "passed": False,
                "reason": "No se pudieron cargar al menos 2 frames desde S3 para APPROACH.",
                "stats": {"framesCount": len(frames)},
            }

        return self._evaluate_approach(frames)
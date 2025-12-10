from typing import Any, Dict, List, Optional, Tuple

import boto3
import cv2
import numpy as np

s3 = boto3.client("s3")


class ApproachEngine:
    """
    Approach detection engine for liveness verification.
    
    Detects if the user moved closer to the camera by comparing face size
    between the first and last frames of a sequence.
    """
    
    THRESHOLD_SCALE = 1.07
    MIN_FACE_SIZE_SMALL = (60, 60)
    MIN_FACE_SIZE_LARGE = (150, 150)
    
    def __init__(self) -> None:
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.face_cascade_alt = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
        )

    def load_frame_from_s3(self, bucket: str, key: str) -> Optional[np.ndarray]:
        try:
            obj = s3.get_object(Bucket=bucket, Key=key)
            data = obj["Body"].read()
            arr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            print(f"❌ S3 error: {key} - {e}")
            return None

    def _detect_face_robust(
        self, gray: np.ndarray, frame_label: str = ""
    ) -> Tuple[Optional[Tuple[int, int, int, int]], Dict[str, Any]]:
        """Robust face detection with multiple strategies."""
        debug_info = {
            "frame_label": frame_label,
            "image_shape": gray.shape,
            "strategies_tried": [],
            "faces_found": 0,
        }
        
        all_faces = []
        
        # Strategy 1: Default parameters, small minSize
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=self.MIN_FACE_SIZE_SMALL,
        )
        debug_info["strategies_tried"].append({"name": "default_small", "found": len(faces)})
        all_faces.extend(faces)
        
        # Strategy 2: More permissive parameters
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=2,
            minSize=self.MIN_FACE_SIZE_LARGE,
        )
        debug_info["strategies_tried"].append({"name": "permissive_large", "found": len(faces)})
        all_faces.extend(faces)
        
        # Strategy 3: Alternative cascade
        faces = self.face_cascade_alt.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=self.MIN_FACE_SIZE_SMALL,
        )
        debug_info["strategies_tried"].append({"name": "alt_cascade", "found": len(faces)})
        all_faces.extend(faces)
        
        # Strategy 4: Downscaled image
        h, w = gray.shape[:2]
        if w > 800:
            scale = 640 / w
            small = cv2.resize(gray, None, fx=scale, fy=scale)
            faces_small = self.face_cascade.detectMultiScale(
                small,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(40, 40),
            )
            faces_scaled = [
                (int(x/scale), int(y/scale), int(w_/scale), int(h_/scale))
                for (x, y, w_, h_) in faces_small
            ]
            debug_info["strategies_tried"].append({"name": "downscaled", "found": len(faces_scaled)})
            all_faces.extend(faces_scaled)
        
        debug_info["faces_found"] = len(all_faces)
        
        if len(all_faces) == 0:
            return None, debug_info
        
        largest_face = max(all_faces, key=lambda f: f[2] * f[3])
        debug_info["selected_face"] = {
            "x": int(largest_face[0]),
            "y": int(largest_face[1]),
            "w": int(largest_face[2]),
            "h": int(largest_face[3]),
            "area": int(largest_face[2] * largest_face[3]),
        }
        
        return tuple(largest_face), debug_info

    def _evaluate_approach(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Evaluate approach by comparing face size between first and last frame."""
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

        face1, debug1 = self._detect_face_robust(gray1, "first_frame")
        face2, debug2 = self._detect_face_robust(gray2, "last_frame")

        if face1 is None or face2 is None:
            # Try with intermediate frames if last frame failed
            if face2 is None and len(frames) > 2:
                for i in range(len(frames) - 2, 0, -1):
                    try:
                        gray_mid = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
                        face2, debug2 = self._detect_face_robust(gray_mid, f"frame_{i}")
                        if face2 is not None:
                            break
                    except Exception:
                        continue
            
            if face1 is None or face2 is None:
                return {
                    "livenessScore": 0.0,
                    "passed": False,
                    "reason": "No se detectó rostro en uno de los frames. Asegúrate de que tu cara esté visible y bien iluminada.",
                    "stats": {
                        "framesCount": len(frames),
                        "face1Detected": face1 is not None,
                        "face2Detected": face2 is not None,
                        "face1Debug": debug1,
                        "face2Debug": debug2,
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
        passed = scale_change > self.THRESHOLD_SCALE

        raw_score = (scale_change - 1.0) / 0.3
        liveness_score = max(0.0, min(1.0, round(raw_score, 3)))

        reason = None
        if not passed:
            if scale_change < 1.0:
                reason = (
                    f"Te alejaste de la cámara en lugar de acercarte (scale={scale_change:.2f}). "
                    "Por favor, acércate gradualmente a la cámara durante el reto."
                )
            else:
                reason = (
                    f"El acercamiento fue muy leve (scale={scale_change:.2f}). "
                    "Acércate más a la cámara de forma gradual y repite el reto."
                )

        return {
            "livenessScore": liveness_score,
            "passed": passed,
            "reason": reason,
            "stats": {
                "framesCount": len(frames),
                "face1Area": area1,
                "face2Area": area2,
                "face1Bbox": [int(x1), int(y1), int(w1), int(h1)],
                "face2Bbox": [int(x2), int(y2), int(w2), int(h2)],
                "scaleChange": round(scale_change, 4),
                "threshold": self.THRESHOLD_SCALE,
                "firstFrameIndex": 0,
                "lastFrameIndex": len(frames) - 1,
            },
        }

    def run(self, bucket: str, frame_keys: List[str]) -> Dict[str, Any]:
        """Load frames from S3 and evaluate approach."""
        if len(frame_keys) < 2:
            return {
                "livenessScore": 0.0,
                "passed": False,
                "reason": "Se requieren al menos 2 frame keys para APPROACH.",
                "stats": {"framesCount": len(frame_keys)},
            }
        
        indices_to_load = [0]
        
        if len(frame_keys) > 4:
            indices_to_load.append(int(len(frame_keys) * 0.6))
            indices_to_load.append(int(len(frame_keys) * 0.8))
        
        indices_to_load.append(len(frame_keys) - 1)
        indices_to_load = sorted(set(indices_to_load))
        
        frames: List[np.ndarray] = []
        
        for idx in indices_to_load:
            key = frame_keys[idx]
            img = self.load_frame_from_s3(bucket, key)
            if img is not None:
                frames.append(img)

        if len(frames) < 2:
            return {
                "livenessScore": 0.0,
                "passed": False,
                "reason": "No se pudieron cargar al menos 2 frames desde S3 para APPROACH.",
                "stats": {"framesCount": len(frames)},
            }

        return self._evaluate_approach(frames)

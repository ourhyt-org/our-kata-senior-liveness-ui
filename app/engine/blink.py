import json
from typing import Any, Dict, List, Tuple

import boto3
import cv2
import numpy as np

from utils.images import (
    to_gray,
    resize,
    frame_difference,
    debug_image_info,
)

s3 = boto3.client("s3")


class BlinkEngine:
    def __init__(self) -> None:
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
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

    def extract_eye_band(self, img) -> Tuple[np.ndarray | None, Dict[str, Any]]:
        info: Dict[str, Any] = {"face_found": False, "eyes_band_shape": None}

        gray = to_gray(img)
        if gray is None:
            return None, info

        small = resize(gray, width=400)

        faces = self.face_cascade.detectMultiScale(
            small,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(80, 80),
        )

        if len(faces) == 0:
            return None, info

        (x, y, w, h) = faces[0]
        info["face_found"] = True

        y1 = int(y + 0.15 * h)
        y2 = int(y + 0.5 * h)
        y2 = min(y2, small.shape[0])

        eye_band = small[y1:y2, x : x + w]
        info["eyes_band_shape"] = eye_band.shape if eye_band is not None else None

        return eye_band, info

    def analyze_blink(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        if len(frames) < 3:
            return {
                "passed": False,
                "livenessScore": 0.0,
                "reason": "Muy pocos frames para analizar parpadeo (mínimo 3).",
                "stats": {"framesCount": len(frames)},
            }

        eye_brightness: List[float] = []
        valid_indexes: List[int] = []
        eye_band_debug: List[Dict[str, Any]] = []

        for idx, img in enumerate(frames):
            eye_band, info = self.extract_eye_band(img)
            info["frame_index"] = idx
            eye_band_debug.append(info)

            if eye_band is None:
                eye_brightness.append(0.0)
                continue

            bright = float(np.mean(eye_band))
            eye_brightness.append(bright)
            valid_indexes.append(idx)

        print("🔍 Debug banda ojos:", json.dumps(eye_band_debug))

        if len(valid_indexes) < 2:
            return {
                "passed": False,
                "livenessScore": 0.1,
                "reason": "No se detectó rostro de forma consistente en los frames.",
                "stats": {
                    "framesCount": len(frames),
                    "validEyeFrames": len(valid_indexes),
                    "eyeBrightness": eye_brightness,
                },
            }

        max_b = max(eye_brightness)
        has_positive = any(b > 0 for b in eye_brightness)
        min_b = (
            min(b for b in eye_brightness if b > 0) if has_positive else 0.0
        )
        amplitude = max_b - min_b

        diffs: List[float] = []
        for i in range(len(frames) - 1):
            d = frame_difference(frames[i], frames[i + 1])
            diffs.append(d)

        max_diff = max(diffs) if diffs else 0.0
        avg_diff = float(np.mean(diffs)) if diffs else 0.0

        print(
            f"📊 Blink metrics -> max_b: {max_b:.2f}, min_b: {min_b:.2f}, amplitude: {amplitude:.2f}, "
            f"max_diff: {max_diff:.4f}, avg_diff: {avg_diff:.4f}"
        )

        BRIGHTNESS_MIN_AMPLITUDE = 18.0
        GLOBAL_DIFF_MIN = 0.05

        blink_detected = amplitude >= BRIGHTNESS_MIN_AMPLITUDE and max_diff >= GLOBAL_DIFF_MIN

        score_brightness = min(1.0, amplitude / 40.0)
        score_motion = min(1.0, max_diff / 0.15)
        liveness_score = float(round(0.6 * score_brightness + 0.4 * score_motion, 3))

        if not blink_detected:
            reason = (
                "No se detectó un parpadeo claro. Intenta cerrar y abrir los ojos de forma más marcada "
                "mirando a la cámara."
            )
        else:
            reason = None

        return {
            "passed": blink_detected,
            "livenessScore": liveness_score,
            "reason": reason,
            "stats": {
                "framesCount": len(frames),
                "validEyeFrames": len(valid_indexes),
                "eyeBrightness": eye_brightness,
                "maxBrightness": max_b,
                "minBrightness": min_b,
                "brightnessAmplitude": amplitude,
                "maxFrameDiff": max_diff,
                "avgFrameDiff": avg_diff,
            },
        }

    def run(self, bucket: str, frame_keys: List[str]) -> Dict[str, Any]:
        frames: List[np.ndarray] = []

        for key in frame_keys:
            img = self.load_frame_from_s3(bucket, key)
            if img is None:
                continue
            print(
                f"✅ Frame cargado: s3://{bucket}/{key}, info={debug_image_info(img)}"
            )
            frames.append(img)

        if not frames:
            return {
                "passed": False,
                "livenessScore": 0.0,
                "reason": "No se pudo cargar ninguno de los frames desde S3.",
                "stats": {"framesCount": 0},
            }

        return self.analyze_blink(frames)
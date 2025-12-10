from typing import Any, Dict, List, Optional, Tuple

import boto3
import cv2
import numpy as np

from app.utils.images import (
    to_gray,
    resize,
    frame_difference,
)

s3 = boto3.client("s3")


class BlinkEngine:

    MIN_DROP_FROM_BASELINE = 2.5
    MIN_LOCAL_VALLEY_DROP = 1.5
    MAX_RECOVERY_DEVIATION_PCT = 0.08
    VALLEY_START_FRACTION = 0.25
    VALLEY_END_FRACTION = 0.75
    MAX_BASELINE_STDDEV = 3.0
    MIN_FRAMES = 5
    MIN_VALID_EYE_FRAMES = 4

    def __init__(self) -> None:
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
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

    def extract_eye_band(self, img: np.ndarray) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
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
        info["face_bbox"] = [int(x), int(y), int(w), int(h)]

        y1 = int(y + 0.15 * h)
        y2 = int(y + 0.5 * h)
        y2 = min(y2, small.shape[0])

        eye_band = small[y1:y2, x : x + w]
        info["eyes_band_shape"] = eye_band.shape if eye_band is not None else None

        return eye_band, info

    def _find_baseline_and_valley(self, brightness: List[float]) -> Dict[str, Any]:
        n = len(brightness)
        
        valid_values = [b for b in brightness if b > 0]
        
        if len(valid_values) < self.MIN_VALID_EYE_FRAMES:
            return {
                "baseline": 0.0,
                "baseline_stddev": 0.0,
                "valley_index": None,
                "valley_value": 0.0,
                "drop_from_baseline": 0.0,
                "local_drop": 0.0,
                "recovery_value": 0.0,
                "recovery_deviation": 0.0,
                "is_valid_blink": False,
                "rejection_reason": "insufficient_valid_frames",
            }
        
        sorted_vals = sorted(valid_values, reverse=True)
        top_half = sorted_vals[: max(2, len(sorted_vals) // 2)]
        baseline = float(np.mean(top_half))
        
        non_min_values = [v for v in valid_values if v > min(valid_values) + 0.5]
        baseline_stddev = float(np.std(non_min_values)) if len(non_min_values) > 1 else 0.0
        
        min_valley_idx = max(1, int(n * self.VALLEY_START_FRACTION))
        max_valley_idx = min(n - 2, int(n * self.VALLEY_END_FRACTION))
        
        best_valley_idx = None
        best_local_drop = 0.0
        
        for i in range(min_valley_idx, max_valley_idx + 1):
            cur = brightness[i]
            if cur <= 0:
                continue
                
            prev_val = self._get_valid_neighbor(brightness, i, direction=-1)
            next_val = self._get_valid_neighbor(brightness, i, direction=+1)
            
            if prev_val is None or next_val is None:
                continue
            
            neighbor_avg = (prev_val + next_val) / 2.0
            local_drop = neighbor_avg - cur
            
            if cur < prev_val and cur < next_val and local_drop > best_local_drop:
                best_local_drop = local_drop
                best_valley_idx = i
        
        if best_valley_idx is None:
            return {
                "baseline": baseline,
                "baseline_stddev": baseline_stddev,
                "valley_index": None,
                "valley_value": 0.0,
                "drop_from_baseline": 0.0,
                "local_drop": 0.0,
                "recovery_value": 0.0,
                "recovery_deviation": 0.0,
                "is_valid_blink": False,
                "rejection_reason": "no_valley_found",
            }
        
        valley_value = brightness[best_valley_idx]
        drop_from_baseline = baseline - valley_value
        
        post_valley_values = [
            brightness[i] for i in range(best_valley_idx + 1, n)
            if brightness[i] > 0
        ]
        
        if not post_valley_values:
            recovery_value = 0.0
            recovery_deviation = 1.0
        else:
            recovery_value = float(np.mean(post_valley_values))
            recovery_deviation = abs(recovery_value - baseline) / baseline if baseline > 0 else 1.0
        
        rejection_reason = None
        
        if drop_from_baseline < self.MIN_DROP_FROM_BASELINE:
            rejection_reason = f"drop_too_small ({drop_from_baseline:.2f} < {self.MIN_DROP_FROM_BASELINE})"
        elif best_local_drop < self.MIN_LOCAL_VALLEY_DROP:
            rejection_reason = f"local_drop_too_small ({best_local_drop:.2f} < {self.MIN_LOCAL_VALLEY_DROP})"
        elif recovery_deviation > self.MAX_RECOVERY_DEVIATION_PCT:
            rejection_reason = f"poor_recovery ({recovery_deviation:.2%} > {self.MAX_RECOVERY_DEVIATION_PCT:.0%})"
        elif baseline_stddev > self.MAX_BASELINE_STDDEV:
            rejection_reason = f"unstable_baseline (stddev={baseline_stddev:.2f} > {self.MAX_BASELINE_STDDEV})"
        
        is_valid_blink = rejection_reason is None
        
        return {
            "baseline": baseline,
            "baseline_stddev": baseline_stddev,
            "valley_index": best_valley_idx,
            "valley_value": valley_value,
            "drop_from_baseline": drop_from_baseline,
            "local_drop": best_local_drop,
            "recovery_value": recovery_value,
            "recovery_deviation": recovery_deviation,
            "is_valid_blink": is_valid_blink,
            "rejection_reason": rejection_reason,
        }
    
    def _get_valid_neighbor(
        self, brightness: List[float], idx: int, direction: int
    ) -> Optional[float]:
        n = len(brightness)
        i = idx + direction
        while 0 <= i < n:
            if brightness[i] > 0:
                return brightness[i]
            i += direction
        return None

    def _compute_liveness_score(
        self, analysis: Dict[str, Any], frame_diffs: List[float]
    ) -> float:
        if not analysis["is_valid_blink"]:
            drop = analysis.get("drop_from_baseline", 0)
            return min(0.3, drop / 10.0)
        
        drop = analysis["drop_from_baseline"]
        local_drop = analysis["local_drop"]
        
        drop_score = min(1.0, drop / 6.0)
        local_score = min(1.0, local_drop / 4.0)
        
        recovery_dev = analysis["recovery_deviation"]
        recovery_score = max(0, 1.0 - (recovery_dev / self.MAX_RECOVERY_DEVIATION_PCT))
        
        baseline_std = analysis["baseline_stddev"]
        stability_score = max(0, 1.0 - (baseline_std / self.MAX_BASELINE_STDDEV))
        
        max_diff = max(frame_diffs) if frame_diffs else 0
        
        motion_penalty = 0.0
        if max_diff > 0.015:
            motion_penalty = min(0.3, (max_diff - 0.015) * 10)
        
        blink_score = (
            0.40 * drop_score +
            0.25 * local_score +
            0.20 * recovery_score +
            0.15 * stability_score
        )
        
        final_score = max(0.0, min(1.0, blink_score - motion_penalty))
        return round(final_score, 3)

    def analyze_blink(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        if len(frames) < self.MIN_FRAMES:
            return {
                "passed": False,
                "livenessScore": 0.0,
                "reason": f"Muy pocos frames para analizar parpadeo (mínimo {self.MIN_FRAMES}, recibidos {len(frames)}).",
                "stats": {"framesCount": len(frames)},
            }

        eye_brightness: List[float] = []
        valid_indexes: List[int] = []

        for idx, img in enumerate(frames):
            eye_band, info = self.extract_eye_band(img)

            if eye_band is None:
                eye_brightness.append(0.0)
                continue

            bright = float(np.mean(eye_band))
            eye_brightness.append(bright)
            valid_indexes.append(idx)

        if len(valid_indexes) < self.MIN_VALID_EYE_FRAMES:
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

        frame_diffs: List[float] = []
        for i in range(len(frames) - 1):
            d = frame_difference(frames[i], frames[i + 1])
            frame_diffs.append(d)

        max_diff = max(frame_diffs) if frame_diffs else 0.0
        avg_diff = float(np.mean(frame_diffs)) if frame_diffs else 0.0

        analysis = self._find_baseline_and_valley(eye_brightness)
        liveness_score = self._compute_liveness_score(analysis, frame_diffs)
        blink_detected = analysis["is_valid_blink"]

        if not blink_detected:
            rejection = analysis.get("rejection_reason", "unknown")
            if "drop_too_small" in str(rejection) or "local_drop_too_small" in str(rejection):
                reason = (
                    "No se detectó un parpadeo claro. Por favor, cierra los ojos completamente "
                    "durante un instante y luego ábrelos, mirando directamente a la cámara."
                )
            elif "no_valley" in str(rejection):
                reason = (
                    "No se detectó el momento del parpadeo. Intenta parpadear de forma más "
                    "marcada y natural durante la captura."
                )
            elif "poor_recovery" in str(rejection):
                reason = (
                    "Se detectó un posible parpadeo pero los ojos no se abrieron completamente después. "
                    "Intenta de nuevo con un parpadeo más natural."
                )
            elif "unstable_baseline" in str(rejection):
                reason = (
                    "Se detectó mucho movimiento o variación. Por favor, mantén la cabeza quieta "
                    "y parpadea de forma natural mirando a la cámara."
                )
            else:
                reason = (
                    "No se pudo verificar el parpadeo. Intenta de nuevo mirando a la cámara "
                    "y parpadeando de forma clara y natural."
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
                "baseline": analysis["baseline"],
                "baselineStddev": analysis["baseline_stddev"],
                "valleyIndex": analysis["valley_index"],
                "valleyValue": analysis["valley_value"],
                "dropFromBaseline": analysis["drop_from_baseline"],
                "localDrop": analysis["local_drop"],
                "recoveryValue": analysis["recovery_value"],
                "recoveryDeviation": analysis["recovery_deviation"],
                "isValidBlink": analysis["is_valid_blink"],
                "rejectionReason": analysis["rejection_reason"],
                "maxFrameDiff": max_diff,
                "avgFrameDiff": avg_diff,
                "brightnessAmplitude": max(eye_brightness) - min([b for b in eye_brightness if b > 0] or [0]),
                "valleyDrop": analysis["local_drop"],
            },
        }

    def run(self, bucket: str, frame_keys: List[str]) -> Dict[str, Any]:
        frames: List[np.ndarray] = []

        for key in frame_keys:
            img = self.load_frame_from_s3(bucket, key)
            if img is not None:
                frames.append(img)

        if not frames:
            return {
                "passed": False,
                "livenessScore": 0.0,
                "reason": "No se pudo cargar ninguno de los frames desde S3.",
                "stats": {"framesCount": 0},
            }

        return self.analyze_blink(frames)

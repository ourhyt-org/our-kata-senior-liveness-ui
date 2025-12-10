# app/main.py
import json
from typing import Any, Dict, List, Optional

from app.engine.blink import BlinkEngine
from app.engine.approach import ApproachEngine
from app.engine.face_match import compare_face_reference

blink_engine = BlinkEngine()
approach_engine = ApproachEngine()


def handler(event, context):
    print("📥 Event recibido en liveness-engine:", json.dumps(event))

    auth_id: Optional[str] = event.get("authId")
    challenge_type: Optional[str] = event.get("challengeType")
    bucket: Optional[str] = event.get("bucket")
    frame_keys: List[str] = event.get("frameKeys", []) or []
    doc_number: Optional[str] = event.get("docNumber")

    if not bucket or not frame_keys:
        body = {
            "authId": auth_id,
            "challengeType": challenge_type,
            "livenessScore": 0.0,
            "passed": False,
            "reason": "Faltan parámetros: 'bucket' o 'frameKeys'.",
            "engine": "liveness-engine",
        }
        return {
            "statusCode": 400,
            "body": json.dumps(body),
        }

    if not challenge_type:
        body = {
            "authId": auth_id,
            "challengeType": None,
            "livenessScore": 0.0,
            "passed": False,
            "reason": "Falta 'challengeType' en el evento.",
            "engine": "liveness-engine",
        }
        return {
            "statusCode": 400,
            "body": json.dumps(body),
        }

    challenge_type = challenge_type.upper()

    if challenge_type == "BLINK":
        result = blink_engine.run(bucket=bucket, frame_keys=frame_keys)
        engine_name = "blink-v1"
    elif challenge_type == "APPROACH":
        result = approach_engine.run(bucket=bucket, frame_keys=frame_keys)
        engine_name = "approach-v1"
    else:
        response = {
            "authId": auth_id,
            "challengeType": challenge_type,
            "bucket": bucket,
            "frameKeys": frame_keys,
            "livenessScore": 0.0,
            "passed": False,
            "reason": f"challengeType no soportado: {challenge_type}",
            "engine": "unsupported",
        }
        print("📤 Respuesta liveness-engine (unsupported):", json.dumps(response))
        return {
            "statusCode": 200,
            "body": json.dumps(response),
        }
    
    face_match_info: Dict[str, Any] = {
        "enabled": False,
        "match": None,
        "similarity": None,
        "error": None,
    }

    if doc_number:
        reference_key = f"idcard/{doc_number}.jpg"
        live_key = frame_keys[len(frame_keys) // 2]

        print(
            f"🔍 FaceMatch: reference=s3://{bucket}/{reference_key}, "
            f"live=s3://{bucket}/{live_key}"
        )

        face_match_info = compare_face_reference(
            bucket=bucket,
            reference_key=reference_key,
            live_key=live_key,
        )

    response: Dict[str, Any] = {
        "authId": auth_id,
        "challengeType": challenge_type,
        "bucket": bucket,
        "frameKeys": frame_keys,
        "livenessScore": float(result.get("livenessScore", 0.0)),
        "passed": bool(result.get("passed", False)),
        "reason": result.get("reason"),
        "engine": engine_name,
        "stats": result.get("stats", {}),
        "faceMatch": face_match_info.get("match"),
        "faceSimilarity": face_match_info.get("similarity"),
        "faceMatchInfo": {
            "enabled": face_match_info.get("enabled"),
            "error": face_match_info.get("error"),
        },
    }

    print("📤 Respuesta liveness-engine:", json.dumps(response))
    return {
        "statusCode": 200,
        "body": json.dumps(response),
    }
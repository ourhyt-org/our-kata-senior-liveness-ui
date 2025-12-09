import json
from typing import Any, Dict, List

from app.engine.blink import BlinkEngine

blink_engine = BlinkEngine()


def handler(event, context):
    print("📥 Event recibido en liveness-engine:", json.dumps(event))

    auth_id: str | None = event.get("authId")
    challenge_type: str | None = event.get("challengeType")
    bucket: str | None = event.get("bucket")
    frame_keys: List[str] = event.get("frameKeys", []) or []

    if not bucket or not frame_keys:
        body = {
            "authId": auth_id,
            "challengeType": challenge_type,
            "livenessScore": 0.0,
            "passed": False,
            "reason": "Faltan parámetros: 'bucket' o 'frameKeys'.",
            "engine": "blink-v1",
        }
        return {
            "statusCode": 400,
            "body": json.dumps(body),
        }

    if challenge_type == "BLINK":
        result = blink_engine.run(bucket=bucket, frame_keys=frame_keys)
        response: Dict[str, Any] = {
            "authId": auth_id,
            "challengeType": challenge_type,
            "bucket": bucket,
            "frameKeys": frame_keys,
            "livenessScore": result.get("livenessScore", 0.0),
            "passed": result.get("passed", False),
            "reason": result.get("reason"),
            "engine": "blink-v1",
            "stats": result.get("stats", {}),
        }
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

    print("📤 Respuesta liveness-engine:", json.dumps(response))
    return {
        "statusCode": 200,
        "body": json.dumps(response),
    }
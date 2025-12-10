
import json
from typing import Any, Dict, List, Optional

from app.engine.blink import BlinkEngine
from app.engine.approach import ApproachEngine
from app.engine.face_match import compare_face_reference

blink_engine = BlinkEngine()
approach_engine = ApproachEngine()


def _truncate_log(data: Dict[str, Any], max_items: int = 3) -> Dict[str, Any]:
    result = {}
    for k, v in data.items():
        if isinstance(v, list) and len(v) > max_items:
            result[k] = v[:max_items] + [f"... +{len(v) - max_items} more"]
        elif isinstance(v, dict):
            result[k] = _truncate_log(v, max_items)
        else:
            result[k] = v
    return result


def handler(event, context):
    auth_id: Optional[str] = event.get("authId")
    challenge_type: Optional[str] = event.get("challengeType")
    bucket: Optional[str] = event.get("bucket")
    frame_keys: List[str] = event.get("frameKeys", []) or []
    doc_number: Optional[str] = event.get("docNumber")

    log_event = _truncate_log({
        "authId": auth_id,
        "challengeType": challenge_type,
        "bucket": bucket,
        "frameKeys": frame_keys,
        "docNumber": doc_number,
    })
    print(f"📥 IN: {json.dumps(log_event)}")

    if not bucket or not frame_keys:
        body = {
            "authId": auth_id,
            "challengeType": challenge_type,
            "livenessScore": 0.0,
            "passed": False,
            "reason": "Faltan parámetros: 'bucket' o 'frameKeys'.",
            "engine": "liveness-engine",
        }
        print(f"📤 OUT: passed=False, reason=missing_params")
        return {"statusCode": 400, "body": json.dumps(body)}

    if not challenge_type:
        body = {
            "authId": auth_id,
            "challengeType": None,
            "livenessScore": 0.0,
            "passed": False,
            "reason": "Falta 'challengeType' en el evento.",
            "engine": "liveness-engine",
        }
        print(f"📤 OUT: passed=False, reason=missing_challenge_type")
        return {"statusCode": 400, "body": json.dumps(body)}

    challenge_type = challenge_type.upper()

    if challenge_type == "BLINK":
        result = blink_engine.run(bucket=bucket, frame_keys=frame_keys)
        engine_name = "blink-v1"
    elif challenge_type == "APPROACH":
        result = approach_engine.run(bucket=bucket, frame_keys=frame_keys)
        engine_name = "approach-v1"
    else:
        print(f"📤 OUT: passed=False, reason=unsupported_challenge")
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
        return {"statusCode": 200, "body": json.dumps(response)}
    
    face_match_info: Dict[str, Any] = {
        "enabled": False,
        "match": None,
        "similarity": None,
        "error": None,
    }

    if doc_number:
        reference_key = f"idcard/{doc_number}.jpg"
        live_key = frame_keys[len(frame_keys) // 2]
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

    print(
        f"📤 OUT: passed={response['passed']}, "
        f"score={response['livenessScore']:.2f}, "
        f"engine={engine_name}, "
        f"faceMatch={face_match_info.get('match')}"
    )

    return {"statusCode": 200, "body": json.dumps(response)}

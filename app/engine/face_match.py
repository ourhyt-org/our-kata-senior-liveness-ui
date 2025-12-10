from typing import Any, Dict

import boto3

rekognition = boto3.client("rekognition")


def compare_face_reference(
    bucket: str,
    reference_key: str,
    live_key: str,
    similarity_threshold: float = 80.0,
) -> Dict[str, Any]:
    try:
        resp = rekognition.compare_faces(
            SourceImage={
                "S3Object": {"Bucket": bucket, "Name": reference_key}
            },
            TargetImage={
                "S3Object": {"Bucket": bucket, "Name": live_key}
            },
            SimilarityThreshold=similarity_threshold,
        )
    except Exception as e:
        print(f"❌ Rekognition error: {e}")
        return {
            "enabled": True,
            "match": None,
            "similarity": None,
            "error": str(e),
        }

    matches = resp.get("FaceMatches", [])
    if not matches:
        return {
            "enabled": True,
            "match": False,
            "similarity": 0.0,
            "error": None,
        }

    best = max(matches, key=lambda m: m.get("Similarity", 0.0))
    similarity = float(best.get("Similarity", 0.0))

    return {
        "enabled": True,
        "match": similarity >= 85.0,
        "similarity": similarity,
        "error": None,
    }

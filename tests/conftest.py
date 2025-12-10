import io
import json
from typing import List
from unittest.mock import MagicMock, patch

import boto3
import cv2
import numpy as np
import pytest
from moto import mock_s3

@pytest.fixture
def blank_image() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def white_image() -> np.ndarray:
    return np.ones((480, 640, 3), dtype=np.uint8) * 255


@pytest.fixture
def gray_image() -> np.ndarray:
    return np.ones((480, 640, 3), dtype=np.uint8) * 128


@pytest.fixture
def random_image() -> np.ndarray:
    np.random.seed(42)
    return np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def synthetic_face_image() -> np.ndarray:
    img = np.ones((720, 1280, 3), dtype=np.uint8) * 200
    
    center = (640, 300)
    axes = (100, 130)
    cv2.ellipse(img, center, axes, 0, 0, 360, (180, 150, 140), -1)
    
    eye_y = 270
    cv2.circle(img, (600, eye_y), 15, (60, 60, 60), -1)
    cv2.circle(img, (680, eye_y), 15, (60, 60, 60), -1)
    
    cv2.ellipse(img, (640, 350), (30, 10), 0, 0, 360, (100, 80, 80), -1)
    
    return img


@pytest.fixture
def synthetic_face_small() -> np.ndarray:
    img = np.ones((720, 1280, 3), dtype=np.uint8) * 200
    
    center = (640, 300)
    axes = (60, 80)
    cv2.ellipse(img, center, axes, 0, 0, 360, (180, 150, 140), -1)
    
    eye_y = 280
    cv2.circle(img, (620, eye_y), 8, (60, 60, 60), -1)
    cv2.circle(img, (660, eye_y), 8, (60, 60, 60), -1)
    
    return img


@pytest.fixture
def synthetic_face_large() -> np.ndarray:
    img = np.ones((720, 1280, 3), dtype=np.uint8) * 200
    
    center = (640, 360)
    axes = (180, 240)
    cv2.ellipse(img, center, axes, 0, 0, 360, (180, 150, 140), -1)
    
    eye_y = 280
    cv2.circle(img, (560, eye_y), 25, (60, 60, 60), -1)
    cv2.circle(img, (720, eye_y), 25, (60, 60, 60), -1)
    
    return img


def create_blink_sequence(
    num_frames: int = 8,
    blink_frame: int = 4,
    base_brightness: int = 200,
    blink_darkness: int = 150,
) -> List[np.ndarray]:
    frames = []
    
    for i in range(num_frames):
        img = np.ones((720, 1280, 3), dtype=np.uint8) * 200
        
        cv2.ellipse(img, (640, 300), (100, 130), 0, 0, 360, (180, 150, 140), -1)
        
        if i == blink_frame:
            eye_brightness = blink_darkness
        else:
            eye_brightness = base_brightness + np.random.randint(-5, 5)
        
        eye_color = (eye_brightness, eye_brightness, eye_brightness)
        eye_y = 270
        cv2.circle(img, (600, eye_y), 15, eye_color, -1)
        cv2.circle(img, (680, eye_y), 15, eye_color, -1)
        
        frames.append(img)
    
    return frames


@pytest.fixture
def blink_sequence_valid() -> List[np.ndarray]:
    return create_blink_sequence(
        num_frames=8,
        blink_frame=4,
        base_brightness=200,
        blink_darkness=140,
    )


@pytest.fixture
def blink_sequence_no_blink() -> List[np.ndarray]:
    return create_blink_sequence(
        num_frames=8,
        blink_frame=-1,
        base_brightness=200,
        blink_darkness=200,
    )


@pytest.fixture
def approach_sequence() -> List[np.ndarray]:
    frames = []
    
    sizes = [(60, 80), (80, 100), (100, 130), (120, 160), (150, 200)]
    
    for axes in sizes:
        img = np.ones((720, 1280, 3), dtype=np.uint8) * 200
        cv2.ellipse(img, (640, 360), axes, 0, 0, 360, (180, 150, 140), -1)
        
        eye_scale = axes[0] / 100.0
        eye_y = int(360 - axes[1] * 0.3)
        eye_offset = int(40 * eye_scale)
        eye_size = int(15 * eye_scale)
        
        cv2.circle(img, (640 - eye_offset, eye_y), eye_size, (60, 60, 60), -1)
        cv2.circle(img, (640 + eye_offset, eye_y), eye_size, (60, 60, 60), -1)
        
        frames.append(img)
    
    return frames



@pytest.fixture
def aws_credentials():
    import os
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"


@pytest.fixture
def s3_client(aws_credentials):
    with mock_s3():
        client = boto3.client("s3", region_name="us-east-1")
        yield client


@pytest.fixture
def s3_bucket(s3_client):
    bucket_name = "test-bucket"
    s3_client.create_bucket(Bucket=bucket_name)
    return bucket_name


def upload_image_to_s3(s3_client, bucket: str, key: str, image: np.ndarray):
    _, buffer = cv2.imencode(".jpg", image)
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=buffer.tobytes(),
        ContentType="image/jpeg",
    )


@pytest.fixture
def s3_with_blink_frames(s3_client, s3_bucket, blink_sequence_valid):
    for i, img in enumerate(blink_sequence_valid):
        key = f"liveness/test/frame_{i+1:03d}.jpg"
        upload_image_to_s3(s3_client, s3_bucket, key, img)
    
    return {
        "bucket": s3_bucket,
        "keys": [f"liveness/test/frame_{i+1:03d}.jpg" for i in range(len(blink_sequence_valid))],
    }


@pytest.fixture
def s3_with_approach_frames(s3_client, s3_bucket, approach_sequence):
    for i, img in enumerate(approach_sequence):
        key = f"liveness/test/approach_{i+1:03d}.jpg"
        upload_image_to_s3(s3_client, s3_bucket, key, img)
    
    return {
        "bucket": s3_bucket,
        "keys": [f"liveness/test/approach_{i+1:03d}.jpg" for i in range(len(approach_sequence))],
    }


@pytest.fixture
def blink_event():
    return {
        "authId": "test-auth-id-123",
        "challengeType": "BLINK",
        "bucket": "test-bucket",
        "frameKeys": [
            "liveness/test/frame_001.jpg",
            "liveness/test/frame_002.jpg",
            "liveness/test/frame_003.jpg",
            "liveness/test/frame_004.jpg",
            "liveness/test/frame_005.jpg",
            "liveness/test/frame_006.jpg",
            "liveness/test/frame_007.jpg",
            "liveness/test/frame_008.jpg",
        ],
    }


@pytest.fixture
def approach_event():
    return {
        "authId": "test-auth-id-456",
        "challengeType": "APPROACH",
        "bucket": "test-bucket",
        "frameKeys": [
            "liveness/test/approach_001.jpg",
            "liveness/test/approach_002.jpg",
            "liveness/test/approach_003.jpg",
            "liveness/test/approach_004.jpg",
            "liveness/test/approach_005.jpg",
        ],
    }


@pytest.fixture
def invalid_event_no_bucket():
    return {
        "authId": "test-auth-id",
        "challengeType": "BLINK",
        "frameKeys": ["frame_001.jpg"],
    }


@pytest.fixture
def invalid_event_no_frames():
    return {
        "authId": "test-auth-id",
        "challengeType": "BLINK",
        "bucket": "test-bucket",
    }


@pytest.fixture
def invalid_event_no_challenge():
    return {
        "authId": "test-auth-id",
        "bucket": "test-bucket",
        "frameKeys": ["frame_001.jpg"],
    }


@pytest.fixture
def mock_s3_client():
    mock_client = MagicMock()
    return mock_client


@pytest.fixture
def mock_lambda_context():
    context = MagicMock()
    context.function_name = "test-liveness-engine"
    context.memory_limit_in_mb = 512
    context.invoked_function_arn = "arn:aws:lambda:us-east-1:123456789:function:test"
    context.aws_request_id = "test-request-id"
    return context


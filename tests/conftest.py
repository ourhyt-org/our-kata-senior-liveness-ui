"""
Shared fixtures for liveness engine tests.

Provides:
- Synthetic test images (faces, eye bands)
- Mocked S3 client
- Sample event payloads
"""

import io
import json
from typing import List
from unittest.mock import MagicMock, patch

import boto3
import cv2
import numpy as np
import pytest
from moto import mock_s3

# ============================================================================
# IMAGE FIXTURES - Synthetic images for testing
# ============================================================================


@pytest.fixture
def blank_image() -> np.ndarray:
    """Create a blank 640x480 BGR image (black)."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def white_image() -> np.ndarray:
    """Create a white 640x480 BGR image."""
    return np.ones((480, 640, 3), dtype=np.uint8) * 255


@pytest.fixture
def gray_image() -> np.ndarray:
    """Create a gray 640x480 BGR image (mid-brightness)."""
    return np.ones((480, 640, 3), dtype=np.uint8) * 128


@pytest.fixture
def random_image() -> np.ndarray:
    """Create a random noise 640x480 BGR image."""
    np.random.seed(42)
    return np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def synthetic_face_image() -> np.ndarray:
    """
    Create a synthetic image with a simple "face-like" pattern.
    
    This creates an oval shape that Haar Cascade might detect as a face.
    Note: Real Haar Cascade detection requires actual face features,
    so this is mainly for testing image processing, not detection.
    """
    img = np.ones((720, 1280, 3), dtype=np.uint8) * 200  # Light gray background
    
    # Draw an oval "face" in the center
    center = (640, 300)
    axes = (100, 130)  # Width, Height of face oval
    cv2.ellipse(img, center, axes, 0, 0, 360, (180, 150, 140), -1)
    
    # Draw "eyes" - two darker circles
    eye_y = 270
    cv2.circle(img, (600, eye_y), 15, (60, 60, 60), -1)  # Left eye
    cv2.circle(img, (680, eye_y), 15, (60, 60, 60), -1)  # Right eye
    
    # Draw "mouth" - a darker ellipse
    cv2.ellipse(img, (640, 350), (30, 10), 0, 0, 360, (100, 80, 80), -1)
    
    return img


@pytest.fixture
def synthetic_face_small() -> np.ndarray:
    """Create a synthetic face image with smaller face (simulating distance)."""
    img = np.ones((720, 1280, 3), dtype=np.uint8) * 200
    
    center = (640, 300)
    axes = (60, 80)  # Smaller face
    cv2.ellipse(img, center, axes, 0, 0, 360, (180, 150, 140), -1)
    
    eye_y = 280
    cv2.circle(img, (620, eye_y), 8, (60, 60, 60), -1)
    cv2.circle(img, (660, eye_y), 8, (60, 60, 60), -1)
    
    return img


@pytest.fixture
def synthetic_face_large() -> np.ndarray:
    """Create a synthetic face image with larger face (simulating closeness)."""
    img = np.ones((720, 1280, 3), dtype=np.uint8) * 200
    
    center = (640, 360)
    axes = (180, 240)  # Larger face
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
    """
    Create a sequence of synthetic frames simulating a blink.
    
    Args:
        num_frames: Total frames in sequence
        blink_frame: Which frame should be the "blink" (darkest)
        base_brightness: Brightness of eyes-open frames
        blink_darkness: Brightness of eyes-closed frame
    
    Returns:
        List of BGR images
    """
    frames = []
    
    for i in range(num_frames):
        img = np.ones((720, 1280, 3), dtype=np.uint8) * 200
        
        # Face oval
        cv2.ellipse(img, (640, 300), (100, 130), 0, 0, 360, (180, 150, 140), -1)
        
        # Eyes - brightness varies based on frame
        if i == blink_frame:
            eye_brightness = blink_darkness  # Closed eyes (darker)
        else:
            # Slight variation in open eyes
            eye_brightness = base_brightness + np.random.randint(-5, 5)
        
        eye_color = (eye_brightness, eye_brightness, eye_brightness)
        eye_y = 270
        cv2.circle(img, (600, eye_y), 15, eye_color, -1)
        cv2.circle(img, (680, eye_y), 15, eye_color, -1)
        
        frames.append(img)
    
    return frames


@pytest.fixture
def blink_sequence_valid() -> List[np.ndarray]:
    """Create a valid blink sequence (clear blink in the middle)."""
    return create_blink_sequence(
        num_frames=8,
        blink_frame=4,
        base_brightness=200,
        blink_darkness=140,
    )


@pytest.fixture
def blink_sequence_no_blink() -> List[np.ndarray]:
    """Create a sequence with no blink (constant brightness)."""
    return create_blink_sequence(
        num_frames=8,
        blink_frame=-1,  # No blink frame
        base_brightness=200,
        blink_darkness=200,  # Same as base
    )


@pytest.fixture
def approach_sequence() -> List[np.ndarray]:
    """Create a sequence of frames simulating approach (face getting larger)."""
    frames = []
    
    # Face size increases from small to large
    sizes = [(60, 80), (80, 100), (100, 130), (120, 160), (150, 200)]
    
    for axes in sizes:
        img = np.ones((720, 1280, 3), dtype=np.uint8) * 200
        cv2.ellipse(img, (640, 360), axes, 0, 0, 360, (180, 150, 140), -1)
        
        # Scale eye positions based on face size
        eye_scale = axes[0] / 100.0
        eye_y = int(360 - axes[1] * 0.3)
        eye_offset = int(40 * eye_scale)
        eye_size = int(15 * eye_scale)
        
        cv2.circle(img, (640 - eye_offset, eye_y), eye_size, (60, 60, 60), -1)
        cv2.circle(img, (640 + eye_offset, eye_y), eye_size, (60, 60, 60), -1)
        
        frames.append(img)
    
    return frames


# ============================================================================
# S3 FIXTURES - Mocked AWS S3
# ============================================================================


@pytest.fixture
def aws_credentials():
    """Mock AWS credentials for moto."""
    import os
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"


@pytest.fixture
def s3_client(aws_credentials):
    """Create a mocked S3 client."""
    with mock_s3():
        client = boto3.client("s3", region_name="us-east-1")
        yield client


@pytest.fixture
def s3_bucket(s3_client):
    """Create a test S3 bucket with some test images."""
    bucket_name = "test-bucket"
    s3_client.create_bucket(Bucket=bucket_name)
    return bucket_name


def upload_image_to_s3(s3_client, bucket: str, key: str, image: np.ndarray):
    """Helper to upload an image to mocked S3."""
    _, buffer = cv2.imencode(".jpg", image)
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=buffer.tobytes(),
        ContentType="image/jpeg",
    )


@pytest.fixture
def s3_with_blink_frames(s3_client, s3_bucket, blink_sequence_valid):
    """Upload blink sequence to mocked S3."""
    for i, img in enumerate(blink_sequence_valid):
        key = f"liveness/test/frame_{i+1:03d}.jpg"
        upload_image_to_s3(s3_client, s3_bucket, key, img)
    
    return {
        "bucket": s3_bucket,
        "keys": [f"liveness/test/frame_{i+1:03d}.jpg" for i in range(len(blink_sequence_valid))],
    }


@pytest.fixture
def s3_with_approach_frames(s3_client, s3_bucket, approach_sequence):
    """Upload approach sequence to mocked S3."""
    for i, img in enumerate(approach_sequence):
        key = f"liveness/test/approach_{i+1:03d}.jpg"
        upload_image_to_s3(s3_client, s3_bucket, key, img)
    
    return {
        "bucket": s3_bucket,
        "keys": [f"liveness/test/approach_{i+1:03d}.jpg" for i in range(len(approach_sequence))],
    }


# ============================================================================
# EVENT FIXTURES - Lambda event payloads
# ============================================================================


@pytest.fixture
def blink_event():
    """Sample BLINK challenge event."""
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
    """Sample APPROACH challenge event."""
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
    """Invalid event missing bucket."""
    return {
        "authId": "test-auth-id",
        "challengeType": "BLINK",
        "frameKeys": ["frame_001.jpg"],
    }


@pytest.fixture
def invalid_event_no_frames():
    """Invalid event missing frameKeys."""
    return {
        "authId": "test-auth-id",
        "challengeType": "BLINK",
        "bucket": "test-bucket",
    }


@pytest.fixture
def invalid_event_no_challenge():
    """Invalid event missing challengeType."""
    return {
        "authId": "test-auth-id",
        "bucket": "test-bucket",
        "frameKeys": ["frame_001.jpg"],
    }


# ============================================================================
# MOCK FIXTURES - For isolating units
# ============================================================================


@pytest.fixture
def mock_s3_client():
    """Create a mock S3 client for unit tests that don't need real S3."""
    mock_client = MagicMock()
    return mock_client


@pytest.fixture
def mock_lambda_context():
    """Create a mock Lambda context object."""
    context = MagicMock()
    context.function_name = "test-liveness-engine"
    context.memory_limit_in_mb = 512
    context.invoked_function_arn = "arn:aws:lambda:us-east-1:123456789:function:test"
    context.aws_request_id = "test-request-id"
    return context


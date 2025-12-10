"""
Tests for app/main.py (Lambda handler)

Tests the main Lambda handler function and its routing logic.
"""

import json
from unittest.mock import MagicMock, patch
import pytest

from app.main import handler


class TestHandlerValidation:
    """Tests for input validation in handler."""

    def test_missing_bucket_returns_400(self, invalid_event_no_bucket, mock_lambda_context):
        """Should return 400 when bucket is missing."""
        result = handler(invalid_event_no_bucket, mock_lambda_context)
        
        assert result["statusCode"] == 400
        body = json.loads(result["body"])
        assert body["passed"] is False
        assert "bucket" in body["reason"].lower() or "frameKeys" in body["reason"]

    def test_missing_frame_keys_returns_400(self, invalid_event_no_frames, mock_lambda_context):
        """Should return 400 when frameKeys is missing."""
        result = handler(invalid_event_no_frames, mock_lambda_context)
        
        assert result["statusCode"] == 400
        body = json.loads(result["body"])
        assert body["passed"] is False

    def test_missing_challenge_type_returns_400(self, invalid_event_no_challenge, mock_lambda_context):
        """Should return 400 when challengeType is missing."""
        result = handler(invalid_event_no_challenge, mock_lambda_context)
        
        assert result["statusCode"] == 400
        body = json.loads(result["body"])
        assert body["passed"] is False
        assert "challengeType" in body["reason"]

    def test_empty_frame_keys_returns_400(self, mock_lambda_context):
        """Should return 400 when frameKeys is empty list."""
        event = {
            "authId": "test",
            "challengeType": "BLINK",
            "bucket": "test-bucket",
            "frameKeys": [],
        }
        
        result = handler(event, mock_lambda_context)
        
        assert result["statusCode"] == 400


class TestHandlerRouting:
    """Tests for challenge type routing."""

    def test_routes_blink_to_blink_engine(self, blink_event, mock_lambda_context):
        """Should route BLINK challenge to BlinkEngine."""
        with patch('app.main.blink_engine') as mock_engine:
            mock_engine.run.return_value = {
                "passed": True,
                "livenessScore": 0.8,
                "reason": None,
                "stats": {},
            }
            
            result = handler(blink_event, mock_lambda_context)
            
            mock_engine.run.assert_called_once()
            body = json.loads(result["body"])
            assert body["engine"] == "blink-v1"

    def test_routes_approach_to_approach_engine(self, approach_event, mock_lambda_context):
        """Should route APPROACH challenge to ApproachEngine."""
        with patch('app.main.approach_engine') as mock_engine:
            mock_engine.run.return_value = {
                "passed": True,
                "livenessScore": 0.9,
                "reason": None,
                "stats": {},
            }
            
            result = handler(approach_event, mock_lambda_context)
            
            mock_engine.run.assert_called_once()
            body = json.loads(result["body"])
            assert body["engine"] == "approach-v1"

    def test_challenge_type_case_insensitive(self, mock_lambda_context):
        """Should handle challengeType case-insensitively."""
        event = {
            "authId": "test",
            "challengeType": "blink",  # lowercase
            "bucket": "test-bucket",
            "frameKeys": ["frame1.jpg"],
        }
        
        with patch('app.main.blink_engine') as mock_engine:
            mock_engine.run.return_value = {
                "passed": True,
                "livenessScore": 0.5,
                "reason": None,
                "stats": {},
            }
            
            result = handler(event, mock_lambda_context)
            
            mock_engine.run.assert_called_once()

    def test_unsupported_challenge_type(self, mock_lambda_context):
        """Should return error for unsupported challengeType."""
        event = {
            "authId": "test",
            "challengeType": "UNKNOWN_TYPE",
            "bucket": "test-bucket",
            "frameKeys": ["frame1.jpg"],
        }
        
        result = handler(event, mock_lambda_context)
        
        assert result["statusCode"] == 200
        body = json.loads(result["body"])
        assert body["passed"] is False
        assert "no soportado" in body["reason"].lower()
        assert body["engine"] == "unsupported"


class TestHandlerResponse:
    """Tests for response structure."""

    def test_response_contains_required_fields(self, blink_event, mock_lambda_context):
        """Should include all required fields in response."""
        with patch('app.main.blink_engine') as mock_engine:
            mock_engine.run.return_value = {
                "passed": True,
                "livenessScore": 0.75,
                "reason": None,
                "stats": {"test": "data"},
            }
            
            result = handler(blink_event, mock_lambda_context)
            body = json.loads(result["body"])
        
        # Required fields
        assert "authId" in body
        assert "challengeType" in body
        assert "livenessScore" in body
        assert "passed" in body
        assert "reason" in body
        assert "engine" in body
        assert "stats" in body
        # Face match fields
        assert "faceMatch" in body
        assert "faceSimilarity" in body
        assert "faceMatchInfo" in body

    def test_preserves_auth_id(self, mock_lambda_context):
        """Should preserve authId in response."""
        event = {
            "authId": "unique-auth-id-12345",
            "challengeType": "BLINK",
            "bucket": "test-bucket",
            "frameKeys": ["frame1.jpg"],
        }
        
        with patch('app.main.blink_engine') as mock_engine:
            mock_engine.run.return_value = {
                "passed": True,
                "livenessScore": 0.5,
                "reason": None,
                "stats": {},
            }
            
            result = handler(event, mock_lambda_context)
            body = json.loads(result["body"])
        
        assert body["authId"] == "unique-auth-id-12345"

    def test_liveness_score_is_float(self, blink_event, mock_lambda_context):
        """Should ensure livenessScore is a float."""
        with patch('app.main.blink_engine') as mock_engine:
            mock_engine.run.return_value = {
                "passed": True,
                "livenessScore": 1,  # Integer from engine
                "reason": None,
                "stats": {},
            }
            
            result = handler(blink_event, mock_lambda_context)
            body = json.loads(result["body"])
        
        assert isinstance(body["livenessScore"], float)

    def test_passed_is_bool(self, blink_event, mock_lambda_context):
        """Should ensure passed is a boolean."""
        with patch('app.main.blink_engine') as mock_engine:
            mock_engine.run.return_value = {
                "passed": 1,  # Truthy integer
                "livenessScore": 0.5,
                "reason": None,
                "stats": {},
            }
            
            result = handler(blink_event, mock_lambda_context)
            body = json.loads(result["body"])
        
        assert isinstance(body["passed"], bool)


class TestHandlerFaceMatch:
    """Tests for face matching integration in handler."""

    def test_face_match_disabled_when_no_doc_number(self, mock_lambda_context):
        """Should not run face match when docNumber is not provided."""
        event = {
            "authId": "test",
            "challengeType": "BLINK",
            "bucket": "test-bucket",
            "frameKeys": ["frame1.jpg", "frame2.jpg", "frame3.jpg"],
            # No docNumber
        }
        
        with patch('app.main.blink_engine') as mock_engine:
            mock_engine.run.return_value = {
                "passed": True,
                "livenessScore": 0.8,
                "reason": None,
                "stats": {},
            }
            
            result = handler(event, mock_lambda_context)
            body = json.loads(result["body"])
        
        assert body["faceMatch"] is None
        assert body["faceSimilarity"] is None
        assert body["faceMatchInfo"]["enabled"] is False

    def test_face_match_enabled_when_doc_number_provided(self, mock_lambda_context):
        """Should run face match when docNumber is provided."""
        event = {
            "authId": "test",
            "challengeType": "BLINK",
            "bucket": "test-bucket",
            "frameKeys": ["frame1.jpg", "frame2.jpg", "frame3.jpg"],
            "docNumber": "1234567890",
        }
        
        with patch('app.main.blink_engine') as mock_engine, \
             patch('app.main.compare_face_reference') as mock_face_match:
            mock_engine.run.return_value = {
                "passed": True,
                "livenessScore": 0.8,
                "reason": None,
                "stats": {},
            }
            mock_face_match.return_value = {
                "enabled": True,
                "match": True,
                "similarity": 95.5,
                "error": None,
            }
            
            result = handler(event, mock_lambda_context)
            body = json.loads(result["body"])
        
        assert body["faceMatch"] is True
        assert body["faceSimilarity"] == 95.5
        assert body["faceMatchInfo"]["enabled"] is True
        assert body["faceMatchInfo"]["error"] is None

    def test_face_match_uses_correct_reference_key(self, mock_lambda_context):
        """Should use correct S3 key for reference image based on docNumber."""
        event = {
            "authId": "test",
            "challengeType": "BLINK",
            "bucket": "test-bucket",
            "frameKeys": ["frame1.jpg", "frame2.jpg", "frame3.jpg"],
            "docNumber": "9876543210",
        }
        
        with patch('app.main.blink_engine') as mock_engine, \
             patch('app.main.compare_face_reference') as mock_face_match:
            mock_engine.run.return_value = {
                "passed": True,
                "livenessScore": 0.8,
                "reason": None,
                "stats": {},
            }
            mock_face_match.return_value = {
                "enabled": True,
                "match": True,
                "similarity": 90.0,
                "error": None,
            }
            
            handler(event, mock_lambda_context)
            
            # Verify correct reference key was used
            call_args = mock_face_match.call_args
            assert call_args.kwargs["reference_key"] == "idcard/9876543210.jpg"

    def test_face_match_uses_middle_frame_as_live(self, mock_lambda_context):
        """Should use middle frame as live image for comparison."""
        event = {
            "authId": "test",
            "challengeType": "BLINK",
            "bucket": "test-bucket",
            "frameKeys": ["f1.jpg", "f2.jpg", "f3.jpg", "f4.jpg", "f5.jpg"],
            "docNumber": "123",
        }
        
        with patch('app.main.blink_engine') as mock_engine, \
             patch('app.main.compare_face_reference') as mock_face_match:
            mock_engine.run.return_value = {
                "passed": True,
                "livenessScore": 0.8,
                "reason": None,
                "stats": {},
            }
            mock_face_match.return_value = {
                "enabled": True,
                "match": True,
                "similarity": 90.0,
                "error": None,
            }
            
            handler(event, mock_lambda_context)
            
            # Middle frame of 5 frames (index 2) should be f3.jpg
            call_args = mock_face_match.call_args
            assert call_args.kwargs["live_key"] == "f3.jpg"

    def test_face_match_no_match_scenario(self, mock_lambda_context):
        """Should handle face match returning no match."""
        event = {
            "authId": "test",
            "challengeType": "BLINK",
            "bucket": "test-bucket",
            "frameKeys": ["frame1.jpg", "frame2.jpg", "frame3.jpg"],
            "docNumber": "1234567890",
        }
        
        with patch('app.main.blink_engine') as mock_engine, \
             patch('app.main.compare_face_reference') as mock_face_match:
            mock_engine.run.return_value = {
                "passed": True,
                "livenessScore": 0.8,
                "reason": None,
                "stats": {},
            }
            mock_face_match.return_value = {
                "enabled": True,
                "match": False,
                "similarity": 45.0,
                "error": None,
            }
            
            result = handler(event, mock_lambda_context)
            body = json.loads(result["body"])
        
        assert body["faceMatch"] is False
        assert body["faceSimilarity"] == 45.0

    def test_face_match_error_scenario(self, mock_lambda_context):
        """Should handle face match API errors gracefully."""
        event = {
            "authId": "test",
            "challengeType": "BLINK",
            "bucket": "test-bucket",
            "frameKeys": ["frame1.jpg", "frame2.jpg", "frame3.jpg"],
            "docNumber": "1234567890",
        }
        
        with patch('app.main.blink_engine') as mock_engine, \
             patch('app.main.compare_face_reference') as mock_face_match:
            mock_engine.run.return_value = {
                "passed": True,
                "livenessScore": 0.8,
                "reason": None,
                "stats": {},
            }
            mock_face_match.return_value = {
                "enabled": True,
                "match": None,
                "similarity": None,
                "error": "InvalidS3ObjectException: Image not found",
            }
            
            result = handler(event, mock_lambda_context)
            body = json.loads(result["body"])
        
        assert body["faceMatch"] is None
        assert body["faceSimilarity"] is None
        assert body["faceMatchInfo"]["enabled"] is True
        assert "InvalidS3ObjectException" in body["faceMatchInfo"]["error"]


class TestHandlerIntegration:
    """Integration tests using real engine classes (mocked S3)."""

    def test_blink_engine_integration(self, mock_lambda_context):
        """Test full flow with BlinkEngine (mocked S3)."""
        from app.engine.blink import BlinkEngine
        
        event = {
            "authId": "integration-test",
            "challengeType": "BLINK",
            "bucket": "test-bucket",
            "frameKeys": ["f1.jpg", "f2.jpg", "f3.jpg", "f4.jpg", "f5.jpg"],
        }
        
        # Mock S3 to return None (no frames)
        with patch.object(BlinkEngine, 'load_frame_from_s3', return_value=None):
            result = handler(event, mock_lambda_context)
        
        body = json.loads(result["body"])
        assert body["passed"] is False
        assert body["engine"] == "blink-v1"

    def test_approach_engine_integration(self, mock_lambda_context):
        """Test full flow with ApproachEngine (mocked S3)."""
        from app.engine.approach import ApproachEngine
        
        event = {
            "authId": "integration-test",
            "challengeType": "APPROACH",
            "bucket": "test-bucket",
            "frameKeys": ["f1.jpg", "f2.jpg"],
        }
        
        # Mock S3 to return None (no frames)
        with patch.object(ApproachEngine, 'load_frame_from_s3', return_value=None):
            result = handler(event, mock_lambda_context)
        
        body = json.loads(result["body"])
        assert body["passed"] is False
        assert body["engine"] == "approach-v1"

    def test_full_flow_with_face_match(self, mock_lambda_context):
        """Test full flow including face match."""
        from app.engine.blink import BlinkEngine
        
        event = {
            "authId": "integration-test",
            "challengeType": "BLINK",
            "bucket": "test-bucket",
            "frameKeys": ["f1.jpg", "f2.jpg", "f3.jpg", "f4.jpg", "f5.jpg"],
            "docNumber": "123456789",
        }
        
        with patch.object(BlinkEngine, 'load_frame_from_s3', return_value=None), \
             patch('app.main.compare_face_reference') as mock_face_match:
            mock_face_match.return_value = {
                "enabled": True,
                "match": True,
                "similarity": 92.0,
                "error": None,
            }
            
            result = handler(event, mock_lambda_context)
        
        body = json.loads(result["body"])
        assert body["engine"] == "blink-v1"
        assert body["faceMatch"] is True
        assert body["faceSimilarity"] == 92.0

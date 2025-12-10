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


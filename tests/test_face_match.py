"""
Tests for app/engine/face_match.py

Tests the face comparison functionality using AWS Rekognition.
"""

from unittest.mock import MagicMock, patch
import pytest

from app.engine.face_match import compare_face_reference


class TestCompareFaceReference:
    """Tests for compare_face_reference function."""

    def test_returns_match_when_faces_match(self):
        """Should return match=True when similarity is above threshold."""
        mock_response = {
            "FaceMatches": [
                {
                    "Similarity": 95.5,
                    "Face": {
                        "BoundingBox": {"Width": 0.5, "Height": 0.5, "Left": 0.25, "Top": 0.25}
                    }
                }
            ],
            "UnmatchedFaces": []
        }
        
        with patch('app.engine.face_match.rekognition') as mock_rekognition:
            mock_rekognition.compare_faces.return_value = mock_response
            
            result = compare_face_reference(
                bucket="test-bucket",
                reference_key="reference/face.jpg",
                live_key="live/frame.jpg",
            )
        
        assert result["enabled"] is True
        assert result["match"] is True
        assert result["similarity"] == 95.5
        assert result["error"] is None

    def test_returns_no_match_when_similarity_below_threshold(self):
        """Should return match=False when similarity is below 85%."""
        mock_response = {
            "FaceMatches": [
                {
                    "Similarity": 75.0,  # Below 85% threshold
                    "Face": {}
                }
            ],
            "UnmatchedFaces": []
        }
        
        with patch('app.engine.face_match.rekognition') as mock_rekognition:
            mock_rekognition.compare_faces.return_value = mock_response
            
            result = compare_face_reference(
                bucket="test-bucket",
                reference_key="reference/face.jpg",
                live_key="live/frame.jpg",
            )
        
        assert result["enabled"] is True
        assert result["match"] is False
        assert result["similarity"] == 75.0
        assert result["error"] is None

    def test_returns_no_match_when_no_faces_found(self):
        """Should return match=False with 0 similarity when no faces match."""
        mock_response = {
            "FaceMatches": [],
            "UnmatchedFaces": [{"BoundingBox": {}}]
        }
        
        with patch('app.engine.face_match.rekognition') as mock_rekognition:
            mock_rekognition.compare_faces.return_value = mock_response
            
            result = compare_face_reference(
                bucket="test-bucket",
                reference_key="reference/face.jpg",
                live_key="live/frame.jpg",
            )
        
        assert result["enabled"] is True
        assert result["match"] is False
        assert result["similarity"] == 0.0
        assert result["error"] is None

    def test_selects_best_match_from_multiple(self):
        """Should select the face with highest similarity when multiple matches."""
        mock_response = {
            "FaceMatches": [
                {"Similarity": 80.0, "Face": {}},
                {"Similarity": 92.5, "Face": {}},  # Best match
                {"Similarity": 88.0, "Face": {}},
            ],
            "UnmatchedFaces": []
        }
        
        with patch('app.engine.face_match.rekognition') as mock_rekognition:
            mock_rekognition.compare_faces.return_value = mock_response
            
            result = compare_face_reference(
                bucket="test-bucket",
                reference_key="reference/face.jpg",
                live_key="live/frame.jpg",
            )
        
        assert result["similarity"] == 92.5
        assert result["match"] is True

    def test_handles_rekognition_error(self):
        """Should handle Rekognition API errors gracefully."""
        with patch('app.engine.face_match.rekognition') as mock_rekognition:
            mock_rekognition.compare_faces.side_effect = Exception(
                "InvalidS3ObjectException: Unable to get object"
            )
            
            result = compare_face_reference(
                bucket="test-bucket",
                reference_key="nonexistent/face.jpg",
                live_key="live/frame.jpg",
            )
        
        assert result["enabled"] is True
        assert result["match"] is None
        assert result["similarity"] is None
        assert result["error"] is not None
        assert "InvalidS3ObjectException" in result["error"]

    def test_handles_invalid_image_format_error(self):
        """Should handle invalid image format errors."""
        with patch('app.engine.face_match.rekognition') as mock_rekognition:
            mock_rekognition.compare_faces.side_effect = Exception(
                "InvalidImageFormatException: Request has invalid image format"
            )
            
            result = compare_face_reference(
                bucket="test-bucket",
                reference_key="reference/corrupted.jpg",
                live_key="live/frame.jpg",
            )
        
        assert result["enabled"] is True
        assert result["match"] is None
        assert "InvalidImageFormatException" in result["error"]

    def test_handles_no_face_detected_in_source(self):
        """Should handle when no face detected in source image."""
        with patch('app.engine.face_match.rekognition') as mock_rekognition:
            mock_rekognition.compare_faces.side_effect = Exception(
                "InvalidParameterException: No face detected in source image"
            )
            
            result = compare_face_reference(
                bucket="test-bucket",
                reference_key="reference/no_face.jpg",
                live_key="live/frame.jpg",
            )
        
        assert result["match"] is None
        assert "No face detected" in result["error"]

    def test_uses_custom_similarity_threshold(self):
        """Should pass custom similarity threshold to Rekognition."""
        mock_response = {"FaceMatches": [], "UnmatchedFaces": []}
        
        with patch('app.engine.face_match.rekognition') as mock_rekognition:
            mock_rekognition.compare_faces.return_value = mock_response
            
            compare_face_reference(
                bucket="test-bucket",
                reference_key="reference/face.jpg",
                live_key="live/frame.jpg",
                similarity_threshold=90.0,
            )
            
            # Verify the threshold was passed to Rekognition
            call_args = mock_rekognition.compare_faces.call_args
            assert call_args.kwargs["SimilarityThreshold"] == 90.0

    def test_passes_correct_s3_objects_to_rekognition(self):
        """Should pass correct S3 bucket and keys to Rekognition."""
        mock_response = {"FaceMatches": [], "UnmatchedFaces": []}
        
        with patch('app.engine.face_match.rekognition') as mock_rekognition:
            mock_rekognition.compare_faces.return_value = mock_response
            
            compare_face_reference(
                bucket="my-bucket",
                reference_key="path/to/reference.jpg",
                live_key="path/to/live.jpg",
            )
            
            call_args = mock_rekognition.compare_faces.call_args
            
            # Verify source image
            assert call_args.kwargs["SourceImage"]["S3Object"]["Bucket"] == "my-bucket"
            assert call_args.kwargs["SourceImage"]["S3Object"]["Name"] == "path/to/reference.jpg"
            
            # Verify target image
            assert call_args.kwargs["TargetImage"]["S3Object"]["Bucket"] == "my-bucket"
            assert call_args.kwargs["TargetImage"]["S3Object"]["Name"] == "path/to/live.jpg"

    def test_boundary_similarity_exactly_85(self):
        """Should return match=True when similarity is exactly 85%."""
        mock_response = {
            "FaceMatches": [{"Similarity": 85.0, "Face": {}}],
            "UnmatchedFaces": []
        }
        
        with patch('app.engine.face_match.rekognition') as mock_rekognition:
            mock_rekognition.compare_faces.return_value = mock_response
            
            result = compare_face_reference(
                bucket="test-bucket",
                reference_key="ref.jpg",
                live_key="live.jpg",
            )
        
        assert result["match"] is True
        assert result["similarity"] == 85.0

    def test_boundary_similarity_just_below_85(self):
        """Should return match=False when similarity is just below 85%."""
        mock_response = {
            "FaceMatches": [{"Similarity": 84.99, "Face": {}}],
            "UnmatchedFaces": []
        }
        
        with patch('app.engine.face_match.rekognition') as mock_rekognition:
            mock_rekognition.compare_faces.return_value = mock_response
            
            result = compare_face_reference(
                bucket="test-bucket",
                reference_key="ref.jpg",
                live_key="live.jpg",
            )
        
        assert result["match"] is False
        assert result["similarity"] == 84.99

    def test_handles_access_denied_error(self):
        """Should handle IAM permission errors."""
        with patch('app.engine.face_match.rekognition') as mock_rekognition:
            mock_rekognition.compare_faces.side_effect = Exception(
                "AccessDeniedException: User is not authorized to perform rekognition:CompareFaces"
            )
            
            result = compare_face_reference(
                bucket="test-bucket",
                reference_key="ref.jpg",
                live_key="live.jpg",
            )
        
        assert result["match"] is None
        assert "AccessDeniedException" in result["error"]

    def test_handles_throttling_error(self):
        """Should handle API throttling errors."""
        with patch('app.engine.face_match.rekognition') as mock_rekognition:
            mock_rekognition.compare_faces.side_effect = Exception(
                "ProvisionedThroughputExceededException: Rate exceeded"
            )
            
            result = compare_face_reference(
                bucket="test-bucket",
                reference_key="ref.jpg",
                live_key="live.jpg",
            )
        
        assert result["match"] is None
        assert "Rate exceeded" in result["error"]


class TestCompareFaceReferenceResponseStructure:
    """Tests to verify response structure is always consistent."""

    def test_success_response_has_all_keys(self):
        """Successful response should have all required keys."""
        mock_response = {
            "FaceMatches": [{"Similarity": 90.0, "Face": {}}],
            "UnmatchedFaces": []
        }
        
        with patch('app.engine.face_match.rekognition') as mock_rekognition:
            mock_rekognition.compare_faces.return_value = mock_response
            
            result = compare_face_reference(
                bucket="test-bucket",
                reference_key="ref.jpg",
                live_key="live.jpg",
            )
        
        assert set(result.keys()) == {"enabled", "match", "similarity", "error"}

    def test_no_match_response_has_all_keys(self):
        """No-match response should have all required keys."""
        mock_response = {"FaceMatches": [], "UnmatchedFaces": []}
        
        with patch('app.engine.face_match.rekognition') as mock_rekognition:
            mock_rekognition.compare_faces.return_value = mock_response
            
            result = compare_face_reference(
                bucket="test-bucket",
                reference_key="ref.jpg",
                live_key="live.jpg",
            )
        
        assert set(result.keys()) == {"enabled", "match", "similarity", "error"}

    def test_error_response_has_all_keys(self):
        """Error response should have all required keys."""
        with patch('app.engine.face_match.rekognition') as mock_rekognition:
            mock_rekognition.compare_faces.side_effect = Exception("Test error")
            
            result = compare_face_reference(
                bucket="test-bucket",
                reference_key="ref.jpg",
                live_key="live.jpg",
            )
        
        assert set(result.keys()) == {"enabled", "match", "similarity", "error"}

    def test_enabled_is_always_true(self):
        """The 'enabled' field should always be True."""
        test_cases = [
            # Success case
            {"FaceMatches": [{"Similarity": 90.0}], "UnmatchedFaces": []},
            # No match case
            {"FaceMatches": [], "UnmatchedFaces": []},
        ]
        
        for mock_response in test_cases:
            with patch('app.engine.face_match.rekognition') as mock_rekognition:
                mock_rekognition.compare_faces.return_value = mock_response
                
                result = compare_face_reference(
                    bucket="test-bucket",
                    reference_key="ref.jpg",
                    live_key="live.jpg",
                )
                
                assert result["enabled"] is True

        # Error case
        with patch('app.engine.face_match.rekognition') as mock_rekognition:
            mock_rekognition.compare_faces.side_effect = Exception("Error")
            
            result = compare_face_reference(
                bucket="test-bucket",
                reference_key="ref.jpg",
                live_key="live.jpg",
            )
            
            assert result["enabled"] is True



from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from app.engine.blink import BlinkEngine


class TestBlinkEngineInit:

    def test_creates_face_cascade(self):
        engine = BlinkEngine()
        assert engine.face_cascade is not None

    def test_creates_eye_cascade(self):
        engine = BlinkEngine()
        assert engine.eye_cascade is not None

    def test_has_threshold_constants(self):
        assert hasattr(BlinkEngine, 'MIN_DROP_FROM_BASELINE')
        assert hasattr(BlinkEngine, 'MIN_LOCAL_VALLEY_DROP')
        assert hasattr(BlinkEngine, 'MAX_RECOVERY_DEVIATION_PCT')
        assert hasattr(BlinkEngine, 'MAX_BASELINE_STDDEV')


class TestExtractEyeBand:

    def test_extracts_eye_band_from_face(self, synthetic_face_image):
        engine = BlinkEngine()
        eye_band, info = engine.extract_eye_band(synthetic_face_image)
        
        assert isinstance(info, dict)
        assert "face_found" in info

    def test_returns_none_for_no_face(self, blank_image):
        engine = BlinkEngine()
        eye_band, info = engine.extract_eye_band(blank_image)
        
        assert eye_band is None
        assert info["face_found"] is False

    def test_handles_invalid_image(self):
        engine = BlinkEngine()
        
        invalid_img = np.zeros((100, 100), dtype=np.float32)
        eye_band, info = engine.extract_eye_band(invalid_img)
        
        assert eye_band is None


class TestFindBaselineAndValley:

    def test_detects_valid_blink_pattern(self):
        engine = BlinkEngine()
        
        brightness = [105.0, 104.5, 105.2, 104.8, 100.0, 104.9, 105.1, 105.0]
        
        result = engine._find_baseline_and_valley(brightness)
        
        assert result["valley_index"] == 4
        assert result["drop_from_baseline"] > 0
        assert result["local_drop"] > 0

    def test_rejects_no_valley_pattern(self):
        engine = BlinkEngine()
        
        brightness = [105.0, 105.1, 104.9, 105.0, 105.2, 104.8, 105.0, 105.1]
        
        result = engine._find_baseline_and_valley(brightness)
        
        if result["valley_index"] is not None:
            assert result["drop_from_baseline"] < engine.MIN_DROP_FROM_BASELINE or \
                   result["local_drop"] < engine.MIN_LOCAL_VALLEY_DROP

    def test_rejects_valley_at_edges(self):
        engine = BlinkEngine()
        
        brightness = [95.0, 105.0, 105.1, 104.9, 105.0, 105.2, 104.8, 105.0]
        
        result = engine._find_baseline_and_valley(brightness)
        
        if result["valley_index"] is not None:
            assert result["valley_index"] != 0

    def test_handles_zeros_in_brightness(self):
        engine = BlinkEngine()
        
        brightness = [105.0, 0.0, 105.2, 100.0, 105.0, 0.0, 104.8, 105.0]
        
        result = engine._find_baseline_and_valley(brightness)
        
        assert "baseline" in result
        assert "is_valid_blink" in result

    def test_insufficient_valid_frames(self):
        engine = BlinkEngine()
        
        brightness = [105.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 104.0]
        
        result = engine._find_baseline_and_valley(brightness)
        
        assert result["is_valid_blink"] is False
        assert result["rejection_reason"] == "insufficient_valid_frames"


class TestAnalyzeBlink:

    def test_returns_correct_structure(self, blink_sequence_valid):
        engine = BlinkEngine()
        result = engine.analyze_blink(blink_sequence_valid)
        
        assert "passed" in result
        assert "livenessScore" in result
        assert "reason" in result
        assert "stats" in result
        
        assert isinstance(result["passed"], bool)
        assert isinstance(result["livenessScore"], float)
        assert isinstance(result["stats"], dict)

    def test_rejects_too_few_frames(self):
        engine = BlinkEngine()
        
        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(2)]
        result = engine.analyze_blink(frames)
        
        assert result["passed"] is False
        assert "pocos frames" in result["reason"].lower()

    def test_stats_contain_debug_info(self, blink_sequence_valid):
        engine = BlinkEngine()
        result = engine.analyze_blink(blink_sequence_valid)
        
        stats = result["stats"]
        assert "framesCount" in stats
        assert "eyeBrightness" in stats

    def test_liveness_score_in_range(self, blink_sequence_valid):
        engine = BlinkEngine()
        result = engine.analyze_blink(blink_sequence_valid)
        
        assert 0.0 <= result["livenessScore"] <= 1.0


class TestBlinkEngineRun:

    def test_returns_error_when_no_frames_loaded(self):
        engine = BlinkEngine()
        
        with patch.object(engine, 'load_frame_from_s3', return_value=None):
            result = engine.run("fake-bucket", ["frame1.jpg", "frame2.jpg"])
        
        assert result["passed"] is False
        assert result["livenessScore"] == 0.0
        assert "cargar" in result["reason"].lower()

    def test_processes_valid_frames(self, blink_sequence_valid):
        engine = BlinkEngine()
        
        frame_iter = iter(blink_sequence_valid)
        
        def mock_load(bucket, key):
            try:
                return next(frame_iter)
            except StopIteration:
                return None
        
        with patch.object(engine, 'load_frame_from_s3', side_effect=mock_load):
            result = engine.run(
                "test-bucket",
                [f"frame_{i}.jpg" for i in range(len(blink_sequence_valid))]
            )
        
        assert "passed" in result
        assert "livenessScore" in result
        assert result["stats"]["framesCount"] == len(blink_sequence_valid)


class TestBlinkScenarios:

    def test_case_a_real_blink_pattern(self):

        engine = BlinkEngine()
        
        brightness = [103.12, 102.03, 102.83, 103.69, 105.06, 102.27, 104.79, 104.85]
        
        result = engine._find_baseline_and_valley(brightness)
        
        assert result["valley_index"] == 5
        assert result["valley_value"] == 102.27

    def test_case_b_no_blink_pattern(self):
        engine = BlinkEngine()
        
        brightness = [102.82, 103.24, 107.16, 105.31, 107.13, 106.05, 105.96, 105.45]
        
        result = engine._find_baseline_and_valley(brightness)
        
        if result["is_valid_blink"]:
            pass


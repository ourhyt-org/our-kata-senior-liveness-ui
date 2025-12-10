
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest

from app.engine.approach import ApproachEngine


class TestApproachEngineInit:

    def test_creates_face_cascade(self):
        engine = ApproachEngine()
        assert engine.face_cascade is not None

    def test_creates_alt_face_cascade(self):
        engine = ApproachEngine()
        assert engine.face_cascade_alt is not None

    def test_has_threshold_constant(self):
        assert hasattr(ApproachEngine, 'THRESHOLD_SCALE')
        assert ApproachEngine.THRESHOLD_SCALE > 1.0


class TestDetectFaceRobust:

    def test_returns_debug_info(self, blank_image):
        engine = ApproachEngine()
        gray = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY)
        
        face, debug_info = engine._detect_face_robust(gray, "test_frame")
        
        assert isinstance(debug_info, dict)
        assert debug_info["frame_label"] == "test_frame"
        assert "strategies_tried" in debug_info
        assert "faces_found" in debug_info

    def test_tries_multiple_strategies(self, synthetic_face_image):
        engine = ApproachEngine()
        gray = cv2.cvtColor(synthetic_face_image, cv2.COLOR_BGR2GRAY)
        
        face, debug_info = engine._detect_face_robust(gray, "test")
        
        assert len(debug_info["strategies_tried"]) >= 3

    def test_returns_largest_face(self):
        engine = ApproachEngine()
        
        img = np.ones((720, 1280), dtype=np.uint8) * 200
        
        mock_faces = np.array([
            [100, 100, 50, 50],
            [300, 200, 150, 150],
            [500, 300, 80, 80],
        ])
        
        with patch.object(engine.face_cascade, 'detectMultiScale', return_value=mock_faces):
            with patch.object(engine.face_cascade_alt, 'detectMultiScale', return_value=np.array([])):
                face, debug_info = engine._detect_face_robust(img, "test")
        
        if face is not None:
            assert face[2] == 150 or face[3] == 150


class TestEvaluateApproach:

    def test_returns_correct_structure(self, approach_sequence):
        engine = ApproachEngine()
        result = engine._evaluate_approach(approach_sequence)
        
        assert "passed" in result
        assert "livenessScore" in result
        assert "reason" in result
        assert "stats" in result

    def test_rejects_single_frame(self):
        engine = ApproachEngine()
        
        single_frame = [np.zeros((480, 640, 3), dtype=np.uint8)]
        result = engine._evaluate_approach(single_frame)
        
        assert result["passed"] is False
        assert "2 frames" in result["reason"]

    def test_detects_no_face_scenario(self, blank_image, white_image):
        engine = ApproachEngine()
        
        frames = [blank_image, white_image]
        result = engine._evaluate_approach(frames)
        
        assert result["passed"] is False
        assert "rostro" in result["reason"].lower()

    def test_calculates_scale_change(self):
        engine = ApproachEngine()
        
        face_small = (100, 100, 100, 100)
        face_large = (100, 100, 150, 150)
        
        with patch.object(engine, '_detect_face_robust') as mock_detect:
            mock_detect.side_effect = [
                (face_small, {"strategies_tried": [], "faces_found": 1}),
                (face_large, {"strategies_tried": [], "faces_found": 1}),
            ]
            
            frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(2)]
            result = engine._evaluate_approach(frames)
        
        assert result["passed"] is True
        assert result["stats"]["scaleChange"] == 2.25

    def test_fails_when_user_moves_away(self):
        engine = ApproachEngine()
        
        face_large = (100, 100, 150, 150)
        face_small = (100, 100, 100, 100)
        
        with patch.object(engine, '_detect_face_robust') as mock_detect:
            mock_detect.side_effect = [
                (face_large, {"strategies_tried": [], "faces_found": 1}),
                (face_small, {"strategies_tried": [], "faces_found": 1}),
            ]
            
            frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(2)]
            result = engine._evaluate_approach(frames)
        
        assert result["passed"] is False
        assert "alejaste" in result["reason"].lower()
        assert result["stats"]["scaleChange"] < 1.0

    def test_fails_when_approach_too_small(self):
        engine = ApproachEngine()
        
        face1 = (100, 100, 100, 100)
        face2 = (100, 100, 103, 103)
        
        with patch.object(engine, '_detect_face_robust') as mock_detect:
            mock_detect.side_effect = [
                (face1, {"strategies_tried": [], "faces_found": 1}),
                (face2, {"strategies_tried": [], "faces_found": 1}),
            ]
            
            frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(2)]
            result = engine._evaluate_approach(frames)
        
        assert result["passed"] is False
        assert "leve" in result["reason"].lower()


class TestApproachEngineRun:

    def test_returns_error_when_no_frames_loaded(self):
        engine = ApproachEngine()
        
        with patch.object(engine, 'load_frame_from_s3', return_value=None):
            result = engine.run("fake-bucket", ["frame1.jpg", "frame2.jpg"])
        
        assert result["passed"] is False
        assert result["livenessScore"] == 0.0
        assert "cargar" in result["reason"].lower()

    def test_loads_first_and_last_frames(self, approach_sequence):
        engine = ApproachEngine()
        
        loaded_keys = []
        
        def mock_load(bucket, key):
            loaded_keys.append(key)
            idx = int(key.split("_")[1].split(".")[0]) - 1
            if idx < len(approach_sequence):
                return approach_sequence[idx]
            return None
        
        with patch.object(engine, 'load_frame_from_s3', side_effect=mock_load):
            keys = [f"frame_{i+1:03d}.jpg" for i in range(8)]
            result = engine.run("test-bucket", keys)
        
        assert "frame_001.jpg" in loaded_keys
        assert "frame_008.jpg" in loaded_keys

    def test_uses_intermediate_frames_as_fallback(self):
        engine = ApproachEngine()
        
        frames = [
            np.ones((720, 1280, 3), dtype=np.uint8) * 200,
            np.ones((720, 1280, 3), dtype=np.uint8) * 200,
            np.ones((720, 1280, 3), dtype=np.uint8) * 200,
            np.zeros((720, 1280, 3), dtype=np.uint8),
        ]
        
        loaded_indices = []
        
        def mock_load(bucket, key):
            idx = int(key.split("_")[1].split(".")[0]) - 1
            loaded_indices.append(idx)
            return frames[idx] if idx < len(frames) else None
        
        with patch.object(engine, 'load_frame_from_s3', side_effect=mock_load):
            keys = [f"frame_{i+1:03d}.jpg" for i in range(4)]
            result = engine.run("test-bucket", keys)
        
        assert "passed" in result

    def test_requires_minimum_two_keys(self):
        engine = ApproachEngine()
        
        result = engine.run("test-bucket", ["single_frame.jpg"])
        
        assert result["passed"] is False
        assert "2 frame keys" in result["reason"]


class TestApproachScoreCalculation:

    def test_score_zero_for_no_change(self):
        engine = ApproachEngine()
        
        face = (100, 100, 100, 100)
        
        with patch.object(engine, '_detect_face_robust') as mock_detect:
            mock_detect.side_effect = [
                (face, {"strategies_tried": [], "faces_found": 1}),
                (face, {"strategies_tried": [], "faces_found": 1}),
            ]
            
            frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(2)]
            result = engine._evaluate_approach(frames)
        
        assert result["livenessScore"] == 0.0

    def test_score_one_for_large_approach(self):
        engine = ApproachEngine()
        
        face_small = (100, 100, 100, 100)
        face_large = (100, 100, 200, 200)
        
        with patch.object(engine, '_detect_face_robust') as mock_detect:
            mock_detect.side_effect = [
                (face_small, {"strategies_tried": [], "faces_found": 1}),
                (face_large, {"strategies_tried": [], "faces_found": 1}),
            ]
            
            frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(2)]
            result = engine._evaluate_approach(frames)
        
        assert result["livenessScore"] == 1.0

    def test_score_in_valid_range(self):
        engine = ApproachEngine()
        
        test_cases = [
            ((100, 100), (100, 100)),
            ((100, 100), (110, 110)),
            ((100, 100), (150, 150)),
            ((150, 150), (100, 100)),
        ]
        
        for (w1, h1), (w2, h2) in test_cases:
            face1 = (100, 100, w1, h1)
            face2 = (100, 100, w2, h2)
            
            with patch.object(engine, '_detect_face_robust') as mock_detect:
                mock_detect.side_effect = [
                    (face1, {"strategies_tried": [], "faces_found": 1}),
                    (face2, {"strategies_tried": [], "faces_found": 1}),
                ]
                
                frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(2)]
                result = engine._evaluate_approach(frames)
            
            assert 0.0 <= result["livenessScore"] <= 1.0


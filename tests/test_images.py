"""
Tests for app/utils/images.py

Tests image processing utility functions.
"""

import cv2
import numpy as np
import pytest

from app.utils.images import (
    to_gray,
    resize,
    frame_difference,
    center_crop,
    brightness,
    debug_image_info,
    normalize_score,
)


class TestToGray:
    """Tests for to_gray function."""

    def test_converts_bgr_to_gray(self, random_image):
        """Should convert BGR image to grayscale."""
        gray = to_gray(random_image)
        
        assert gray is not None
        assert len(gray.shape) == 2  # 2D array (no color channels)
        assert gray.shape[:2] == random_image.shape[:2]  # Same height/width

    def test_returns_none_for_invalid_input(self):
        """Should return None for invalid input."""
        assert to_gray(None) is None
        assert to_gray("not an image") is None
        assert to_gray(123) is None

    def test_handles_already_gray_image(self):
        """Should handle already grayscale images gracefully."""
        gray_input = np.zeros((100, 100), dtype=np.uint8)
        # This will fail because cvtColor expects BGR, but should return None gracefully
        result = to_gray(gray_input)
        assert result is None  # cv2.cvtColor will fail on 2D input


class TestResize:
    """Tests for resize function."""

    def test_resize_by_width(self, random_image):
        """Should resize image maintaining aspect ratio by width."""
        resized = resize(random_image, width=320)
        
        assert resized.shape[1] == 320  # Width is 320
        # Aspect ratio: 640/480 = 1.33, so 320 width -> 240 height
        assert resized.shape[0] == 240

    def test_resize_by_height(self, random_image):
        """Should resize image maintaining aspect ratio by height."""
        resized = resize(random_image, height=240)
        
        assert resized.shape[0] == 240  # Height is 240
        # Aspect ratio preserved
        assert resized.shape[1] == 320

    def test_no_resize_when_no_dimensions(self, random_image):
        """Should return original image when no dimensions specified."""
        result = resize(random_image)
        
        assert result is random_image  # Same object reference

    def test_resize_grayscale_image(self):
        """Should handle grayscale images."""
        gray = np.zeros((480, 640), dtype=np.uint8)
        resized = resize(gray, width=320)
        
        assert resized.shape == (240, 320)


class TestFrameDifference:
    """Tests for frame_difference function."""

    def test_identical_images_zero_difference(self, random_image):
        """Identical images should have zero difference."""
        diff = frame_difference(random_image, random_image.copy())
        
        assert diff == 0.0

    def test_different_images_positive_difference(self, blank_image, white_image):
        """Different images should have positive difference."""
        diff = frame_difference(blank_image, white_image)
        
        assert diff > 0.0
        assert diff <= 1.0  # Normalized to 0-1

    def test_max_difference_black_white(self):
        """Black and white images should have maximum difference."""
        black = np.zeros((480, 640, 3), dtype=np.uint8)
        white = np.ones((480, 640, 3), dtype=np.uint8) * 255
        
        diff = frame_difference(black, white)
        
        assert diff == 1.0  # Maximum difference

    def test_handles_invalid_input(self, random_image):
        """Should return 0.0 for invalid input."""
        assert frame_difference(None, random_image) == 0.0
        assert frame_difference(random_image, None) == 0.0
        assert frame_difference(None, None) == 0.0


class TestCenterCrop:
    """Tests for center_crop function."""

    def test_center_crop_square(self, random_image):
        """Should crop a square from center."""
        cropped = center_crop(random_image, size=256)
        
        assert cropped.shape[0] == 256
        assert cropped.shape[1] == 256

    def test_center_crop_larger_than_image(self):
        """Should handle crop size larger than image."""
        small_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cropped = center_crop(small_img, size=256)
        
        # Should return the full image (limited by image size)
        assert cropped.shape[0] <= 100
        assert cropped.shape[1] <= 100

    def test_center_crop_grayscale(self):
        """Should work with grayscale images."""
        gray = np.zeros((480, 640), dtype=np.uint8)
        cropped = center_crop(gray, size=200)
        
        assert cropped.shape == (200, 200)


class TestBrightness:
    """Tests for brightness function."""

    def test_black_image_zero_brightness(self, blank_image):
        """Black image should have zero brightness."""
        assert brightness(blank_image) == 0.0

    def test_white_image_max_brightness(self, white_image):
        """White image should have maximum brightness (255)."""
        assert brightness(white_image) == 255.0

    def test_gray_image_mid_brightness(self, gray_image):
        """Gray (128) image should have mid brightness."""
        assert brightness(gray_image) == 128.0

    def test_handles_invalid_input(self):
        """Should return 0.0 for invalid input."""
        assert brightness(None) == 0.0


class TestDebugImageInfo:
    """Tests for debug_image_info function."""

    def test_valid_image_info(self, random_image):
        """Should return correct info for valid image."""
        info = debug_image_info(random_image)
        
        assert info["valid"] is True
        assert info["width"] == 640
        assert info["height"] == 480
        assert "brightness" in info
        assert isinstance(info["brightness"], float)

    def test_invalid_image_info(self):
        """Should return valid=False for None."""
        info = debug_image_info(None)
        
        assert info == {"valid": False}

    def test_different_image_sizes(self):
        """Should correctly report different image sizes."""
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        info = debug_image_info(img)
        
        assert info["width"] == 1920
        assert info["height"] == 1080


class TestNormalizeScore:
    """Tests for normalize_score function."""

    def test_value_in_range(self):
        """Values in 0-1 range should be rounded."""
        assert normalize_score(0.5) == 0.5
        assert normalize_score(0.12345) == 0.123
        assert normalize_score(0.9999) == 1.0

    def test_value_below_zero(self):
        """Values below 0 should be clamped to 0."""
        assert normalize_score(-0.5) == 0.0
        assert normalize_score(-100) == 0.0

    def test_value_above_one(self):
        """Values above 1 should be clamped to 1."""
        assert normalize_score(1.5) == 1.0
        assert normalize_score(100) == 1.0

    def test_boundary_values(self):
        """Boundary values should be handled correctly."""
        assert normalize_score(0.0) == 0.0
        assert normalize_score(1.0) == 1.0


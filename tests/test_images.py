
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

    def test_converts_bgr_to_gray(self, random_image):
        gray = to_gray(random_image)
        
        assert gray is not None
        assert len(gray.shape) == 2
        assert gray.shape[:2] == random_image.shape[:2]

    def test_returns_none_for_invalid_input(self):
        assert to_gray(None) is None
        assert to_gray("not an image") is None
        assert to_gray(123) is None

    def test_handles_already_gray_image(self):
        gray_input = np.zeros((100, 100), dtype=np.uint8)
        result = to_gray(gray_input)
        assert result is None


class TestResize:

    def test_resize_by_width(self, random_image):
        resized = resize(random_image, width=320)
        
        assert resized.shape[1] == 320
        assert resized.shape[0] == 240

    def test_resize_by_height(self, random_image):
        resized = resize(random_image, height=240)
        
        assert resized.shape[0] == 240
        assert resized.shape[1] == 320

    def test_no_resize_when_no_dimensions(self, random_image):
        result = resize(random_image)
        
        assert result is random_image

    def test_resize_grayscale_image(self):
        gray = np.zeros((480, 640), dtype=np.uint8)
        resized = resize(gray, width=320)
        
        assert resized.shape == (240, 320)


class TestFrameDifference:

    def test_identical_images_zero_difference(self, random_image):
        diff = frame_difference(random_image, random_image.copy())
        
        assert diff == 0.0

    def test_different_images_positive_difference(self, blank_image, white_image):
        diff = frame_difference(blank_image, white_image)
        
        assert diff > 0.0
        assert diff <= 1.0

    def test_max_difference_black_white(self):
        black = np.zeros((480, 640, 3), dtype=np.uint8)
        white = np.ones((480, 640, 3), dtype=np.uint8) * 255
        
        diff = frame_difference(black, white)
        
        assert diff == 1.0

    def test_handles_invalid_input(self, random_image):
        assert frame_difference(None, random_image) == 0.0
        assert frame_difference(random_image, None) == 0.0
        assert frame_difference(None, None) == 0.0


class TestCenterCrop:

    def test_center_crop_square(self, random_image):
        cropped = center_crop(random_image, size=256)
        
        assert cropped.shape[0] == 256
        assert cropped.shape[1] == 256

    def test_center_crop_larger_than_image(self):
        small_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cropped = center_crop(small_img, size=256)
        
        assert cropped.shape[0] <= 100
        assert cropped.shape[1] <= 100

    def test_center_crop_grayscale(self):
        gray = np.zeros((480, 640), dtype=np.uint8)
        cropped = center_crop(gray, size=200)
        
        assert cropped.shape == (200, 200)


class TestBrightness:

    def test_black_image_zero_brightness(self, blank_image):
        assert brightness(blank_image) == 0.0

    def test_white_image_max_brightness(self, white_image):
        assert brightness(white_image) == 255.0

    def test_gray_image_mid_brightness(self, gray_image):
        assert brightness(gray_image) == 128.0

    def test_handles_invalid_input(self):
        assert brightness(None) == 0.0


class TestDebugImageInfo:

    def test_valid_image_info(self, random_image):
        info = debug_image_info(random_image)
        
        assert info["valid"] is True
        assert info["width"] == 640
        assert info["height"] == 480
        assert "brightness" in info
        assert isinstance(info["brightness"], float)

    def test_invalid_image_info(self):
        info = debug_image_info(None)
        
        assert info == {"valid": False}

    def test_different_image_sizes(self):
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        info = debug_image_info(img)
        
        assert info["width"] == 1920
        assert info["height"] == 1080


class TestNormalizeScore:

    def test_value_in_range(self):
        assert normalize_score(0.5) == 0.5
        assert normalize_score(0.12345) == 0.123
        assert normalize_score(0.9999) == 1.0

    def test_value_below_zero(self):
        assert normalize_score(-0.5) == 0.0
        assert normalize_score(-100) == 0.0

    def test_value_above_one(self):
        assert normalize_score(1.5) == 1.0
        assert normalize_score(100) == 1.0

    def test_boundary_values(self):
        assert normalize_score(0.0) == 0.0
        assert normalize_score(1.0) == 1.0


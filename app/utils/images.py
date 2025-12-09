import cv2
import numpy as np


def to_gray(img):
    try:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except Exception:
        return None


def resize(img, width=None, height=None):
    if width is None and height is None:
        return img

    h, w = img.shape[:2]

    if width is not None:
        scale = width / float(w)
        dim = (width, int(h * scale))
    else:
        scale = height / float(h)
        dim = (int(w * scale), height)

    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


def frame_difference(img1, img2):
    gray1 = to_gray(img1)
    gray2 = to_gray(img2)

    if gray1 is None or gray2 is None:
        return 0.0

    gray1 = resize(gray1, width=300)
    gray2 = resize(gray2, width=300)

    diff = cv2.absdiff(gray1, gray2)
    score = float(np.mean(diff)) / 255.0
    return round(score, 4)


def center_crop(img, size=256):
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2

    x1 = max(cx - size // 2, 0)
    y1 = max(cy - size // 2, 0)
    x2 = min(cx + size // 2, w)
    y2 = min(cy + size // 2, h)

    return img[y1:y2, x1:x2]


def brightness(img):
    gray = to_gray(img)
    if gray is None:
        return 0
    return float(np.mean(gray))


def debug_image_info(img):
    if img is None:
        return {"valid": False}

    h, w = img.shape[:2]
    return {
        "valid": True,
        "width": w,
        "height": h,
        "brightness": brightness(img)
    }
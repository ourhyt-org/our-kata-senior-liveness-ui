from typing import List
import cv2
import numpy as np

def load_images_from_s3(s3, bucket: str, keys: List[str]):
    frames = []

    for key in keys:
        obj = s3.get_object(Bucket=bucket, Key=key)
        img_bytes = obj["Body"].read()
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        frames.append(img)

    return frames
import sys
import types
import unittest

import numpy as np


ultralytics_stub = types.ModuleType("ultralytics")
ultralytics_stub.YOLO = object
sys.modules.setdefault("ultralytics", ultralytics_stub)

cv2_stub = types.ModuleType("cv2")
cv2_stub.COLOR_BGR2GRAY = 0


def cvt_color_to_gray(image, _flag):
    return image.mean(axis=2).astype(image.dtype)


cv2_stub.cvtColor = cvt_color_to_gray
sys.modules.setdefault("cv2", cv2_stub)

from tools.review.labeller.generate_layer1_annotations import estimate_uniform_bucket


class UniformBucketTest(unittest.TestCase):
    def test_estimate_uniform_bucket_detects_light_torso_region(self):
        frame = np.zeros((120, 80, 3), dtype=np.uint8)
        frame[20:80, 20:60] = 220
        result = estimate_uniform_bucket(frame, [10, 10, 70, 100])
        self.assertEqual(result["bucket"], "light")
        self.assertGreater(result["luma_mean"], 150.0)

    def test_estimate_uniform_bucket_detects_dark_from_keypoint_torso_crop(self):
        frame = np.full((120, 80, 3), 200, dtype=np.uint8)
        frame[30:70, 25:55] = 20
        keypoints_xy = [[0.0, 0.0] for _ in range(17)]
        keypoints_conf = [0.0 for _ in range(17)]
        for idx, pt in {
            5: [28.0, 35.0],
            6: [52.0, 35.0],
            11: [30.0, 65.0],
            12: [50.0, 65.0],
        }.items():
            keypoints_xy[idx] = pt
            keypoints_conf[idx] = 0.9

        result = estimate_uniform_bucket(
            frame,
            [10, 10, 70, 100],
            keypoints_xy=keypoints_xy,
            keypoints_conf=keypoints_conf,
        )
        self.assertEqual(result["bucket"], "dark")
        self.assertLess(result["luma_mean"], 95.0)


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest import mock

import numpy as np
import sys
import types

cv2_stub = types.ModuleType("cv2")
cv2_stub.COLOR_BGR2RGB = 0
cv2_stub.cvtColor = lambda image, _flag: image
sys.modules.setdefault("cv2", cv2_stub)

from tools.review.labeller.dinov3_bootstrap import (
    Dinov3Bootstrapper,
    foreground_mask_from_dense_features,
    foreground_prior_for_point,
    summarize_foreground_mask,
)


class Dinov3BootstrapMaskTest(unittest.TestCase):
    def test_foreground_mask_from_dense_features_prefers_central_cluster(self):
        features = np.zeros((4, 4, 3), dtype=np.float32)
        features[:, :] = [1.0, 1.0, 1.0]
        features[1:3, 1:3] = [5.0, 5.0, 5.0]
        mask = foreground_mask_from_dense_features(features)
        self.assertEqual(mask.shape, (4, 4))
        self.assertEqual(int(mask[1:3, 1:3].sum()), 4)

    def test_summarize_foreground_mask_includes_ratio_and_bbox(self):
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[1:3, 1:3] = 1
        summary = summarize_foreground_mask(mask)
        summary["image_width"] = 40
        summary["image_height"] = 40
        self.assertEqual(summary["mask_shape"], [4, 4])
        self.assertEqual(summary["foreground_bbox_xyxy"], [1, 1, 2, 2])
        self.assertAlmostEqual(summary["foreground_ratio"], 0.25, places=4)
        self.assertEqual(foreground_prior_for_point(summary, 15, 15), 1.0)
        self.assertEqual(foreground_prior_for_point(summary, 2, 2), 0.0)


class Dinov3BootstrapperTest(unittest.TestCase):
    @mock.patch("tools.review.labeller.dinov3_bootstrap._is_transformers_available", return_value=True)
    def test_run_on_frame_reports_cuda_unavailable_when_not_on_gpu(self, _available):
        bootstrapper = Dinov3Bootstrapper(device="cpu")
        result = bootstrapper.run_on_frame(np.zeros((32, 32, 3), dtype=np.uint8), frame_idx=0)
        payload = result.to_payload()
        self.assertFalse(payload["enabled"])
        self.assertEqual(payload["status"], "cuda_unavailable")


if __name__ == "__main__":
    unittest.main()

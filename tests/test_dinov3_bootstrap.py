import unittest
from unittest import mock

import numpy as np
import sys
import types
from types import SimpleNamespace

cv2_stub = types.ModuleType("cv2")
cv2_stub.COLOR_BGR2RGB = 0
cv2_stub.INTER_NEAREST = 0
cv2_stub.cvtColor = lambda image, _flag: image
cv2_stub.resize = lambda image, dsize, interpolation=None: np.resize(image, (dsize[1], dsize[0]))
sys.modules.setdefault("cv2", cv2_stub)

from tools.review.labeller.dinov3_bootstrap import (
    Dinov3Bootstrapper,
    foreground_mask_from_dense_features,
    foreground_prior_for_point,
    scale_summary_bbox_to_image,
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

    def test_scale_summary_bbox_to_image_projects_mask_bbox(self):
        summary = {
            "mask_shape": [14, 14],
            "foreground_bbox_xyxy": [0, 0, 13, 9],
        }
        scaled = scale_summary_bbox_to_image(summary, image_width=1920, image_height=1080)
        self.assertEqual(scaled["foreground_bbox_xyxy"], [0, 0, 1920, 771])


class Dinov3BootstrapperTest(unittest.TestCase):
    @mock.patch("tools.review.labeller.dinov3_bootstrap._is_transformers_available", return_value=True)
    def test_run_on_frame_reports_cuda_unavailable_when_not_on_gpu(self, _available):
        bootstrapper = Dinov3Bootstrapper(device="cpu")
        result = bootstrapper.run_on_frame(np.zeros((32, 32, 3), dtype=np.uint8), frame_idx=0)
        payload = result.to_payload()
        self.assertFalse(payload["enabled"])
        self.assertEqual(payload["status"], "cuda_unavailable")

    @mock.patch("tools.review.labeller.dinov3_bootstrap._is_transformers_available", return_value=True)
    @mock.patch.object(Dinov3Bootstrapper, "_gpu_ready", return_value=True)
    @mock.patch.object(Dinov3Bootstrapper, "load", side_effect=RuntimeError("401 Unauthorized gated repo"))
    def test_run_on_frame_reports_gated_repo_instead_of_raising(self, _load, _gpu_ready, _available):
        bootstrapper = Dinov3Bootstrapper(device="cuda:0")
        result = bootstrapper.run_on_frame(np.zeros((32, 32, 3), dtype=np.uint8), frame_idx=4)
        payload = result.to_payload()
        self.assertFalse(payload["enabled"])
        self.assertEqual(payload["status"], "gated_repo")
        self.assertEqual(payload["frame_idx"], 4)

    def test_dense_features_skips_cls_and_register_tokens(self):
        bootstrapper = Dinov3Bootstrapper(device="cuda:0")

        class _Tensor:
            def __init__(self, array):
                self.array = array

            def __getitem__(self, index):
                return _Tensor(self.array[index])

            def to(self, _device):
                return self

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.array

        class _Model:
            device = "cpu"
            config = SimpleNamespace(num_register_tokens=4)

            def __call__(self, **kwargs):
                tokens = np.zeros((1, 201, 8), dtype=np.float32)
                return SimpleNamespace(last_hidden_state=_Tensor(tokens))

        bootstrapper.model = _Model()
        bootstrapper.processor = lambda images, return_tensors: {
            "pixel_values": _Tensor(np.zeros((1, 3, 224, 224), dtype=np.float32))
        }

        class _InferenceMode:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        fake_torch = types.SimpleNamespace(inference_mode=lambda: _InferenceMode())
        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            features = bootstrapper.dense_features(np.zeros((32, 32, 3), dtype=np.uint8))

        self.assertEqual(features.shape, (14, 14, 8))


if __name__ == "__main__":
    unittest.main()

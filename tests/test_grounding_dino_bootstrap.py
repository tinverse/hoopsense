import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from types import SimpleNamespace

cv2_stub = types.ModuleType("cv2")
cv2_stub.COLOR_BGR2RGB = 0
cv2_stub.INTER_NEAREST = 0
cv2_stub.cvtColor = lambda image, _flag: image
cv2_stub.resize = lambda image, dsize, interpolation=None: np.resize(image, (dsize[1], dsize[0]))
sys.modules.setdefault("cv2", cv2_stub)

from tools.review.labeller.grounding_dino_bootstrap import (
    GroundingDinoBootstrapper,
    bootstrap_summary_is_informative,
    foreground_mask_from_regions,
    foreground_prior_for_point,
    scale_summary_bbox_to_image,
    summarize_foreground_mask,
)


class GroundingDinoBootstrapMaskTest(unittest.TestCase):
    def test_foreground_mask_from_regions_rasterizes_boxes(self):
        regions = [{"bbox_xyxy": [0, 0, 32, 32], "confidence": 0.8, "text_label": "basketball player"}]
        mask = foreground_mask_from_regions(regions, image_width=32, image_height=32, long_side=8)
        self.assertEqual(mask.shape, (8, 8))
        self.assertGreater(int(mask.sum()), 0)

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

    def test_bootstrap_summary_is_informative_rejects_near_full_frame_mask(self):
        summary = {
            "foreground_ratio": 0.96,
            "foreground_bbox_xyxy": [0, 0, 1919, 1079],
            "image_width": 1920,
            "image_height": 1080,
        }
        self.assertFalse(bootstrap_summary_is_informative(summary))

    def test_bootstrap_summary_is_informative_accepts_localized_mask(self):
        summary = {
            "foreground_ratio": 0.28,
            "foreground_bbox_xyxy": [220, 180, 1460, 900],
            "image_width": 1920,
            "image_height": 1080,
        }
        self.assertTrue(bootstrap_summary_is_informative(summary))


class GroundingDinoBootstrapperTest(unittest.TestCase):
    @mock.patch("tools.review.labeller.grounding_dino_bootstrap._is_hf_available", return_value=True)
    @mock.patch("tools.review.labeller.grounding_dino_bootstrap._is_transformers_available", return_value=True)
    def test_run_on_frame_reports_cuda_unavailable_when_not_on_gpu(self, _available, _hf):
        bootstrapper = GroundingDinoBootstrapper(device="cpu")
        result = bootstrapper.run_on_frame(np.zeros((32, 32, 3), dtype=np.uint8), frame_idx=0)
        payload = result.to_payload()
        self.assertFalse(payload["enabled"])
        self.assertEqual(payload["status"], "cuda_unavailable")

    @mock.patch("tools.review.labeller.grounding_dino_bootstrap._is_hf_available", return_value=True)
    @mock.patch("tools.review.labeller.grounding_dino_bootstrap._is_transformers_available", return_value=True)
    @mock.patch.object(GroundingDinoBootstrapper, "_gpu_ready", return_value=True)
    @mock.patch.object(GroundingDinoBootstrapper, "load", side_effect=RuntimeError("401 Unauthorized gated repo"))
    def test_run_on_frame_reports_gated_repo_instead_of_raising(self, _load, _gpu_ready, _available, _hf):
        bootstrapper = GroundingDinoBootstrapper(device="cuda:0")
        result = bootstrapper.run_on_frame(np.zeros((32, 32, 3), dtype=np.uint8), frame_idx=4)
        payload = result.to_payload()
        self.assertFalse(payload["enabled"])
        self.assertEqual(payload["status"], "gated_repo")
        self.assertEqual(payload["frame_idx"], 4)

    @mock.patch("tools.review.labeller.grounding_dino_bootstrap._is_hf_available", return_value=True)
    @mock.patch("tools.review.labeller.grounding_dino_bootstrap._is_transformers_available", return_value=True)
    @mock.patch.object(GroundingDinoBootstrapper, "_gpu_ready", return_value=True)
    @mock.patch.object(GroundingDinoBootstrapper, "_resolve_model_source", return_value="/tmp/grounding-dino-snapshot")
    def test_load_uses_resolved_local_model_source(self, _resolve_model_source, _gpu_ready, _available, _hf):
        bootstrapper = GroundingDinoBootstrapper(device="cuda:0")
        fake_model = mock.MagicMock()
        processor_loader = mock.MagicMock(return_value=mock.MagicMock())
        model_loader = mock.MagicMock(return_value=fake_model)
        fake_transformers = types.SimpleNamespace(
            AutoProcessor=types.SimpleNamespace(from_pretrained=processor_loader),
            AutoModelForZeroShotObjectDetection=types.SimpleNamespace(from_pretrained=model_loader),
        )
        with mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
            bootstrapper.load()

        processor_loader.assert_called_once_with("/tmp/grounding-dino-snapshot", local_files_only=True)
        model_loader.assert_called_once_with("/tmp/grounding-dino-snapshot", local_files_only=True)
        fake_model.to.assert_called_once_with("cuda:0")
        fake_model.eval.assert_called_once()

    def test_load_is_idempotent_once_model_and_processor_are_present(self):
        bootstrapper = GroundingDinoBootstrapper(device="cuda:0")
        bootstrapper.model = mock.MagicMock()
        bootstrapper.processor = mock.MagicMock()
        result = bootstrapper.load()
        self.assertIs(result, bootstrapper)

    @mock.patch("tools.review.labeller.grounding_dino_bootstrap._is_hf_available", return_value=True)
    @mock.patch("tools.review.labeller.grounding_dino_bootstrap._is_transformers_available", return_value=True)
    @mock.patch.object(GroundingDinoBootstrapper, "_gpu_ready", return_value=True)
    @mock.patch.object(GroundingDinoBootstrapper, "load", return_value=None)
    @mock.patch.object(
        GroundingDinoBootstrapper,
        "predict_regions",
        return_value=[{"bbox_xyxy": [0.0, 0.0, 1920.0, 1080.0], "confidence": 0.91, "text_label": "basketball court"}],
    )
    def test_run_on_frame_rejects_broad_foreground_mask(
        self,
        _predict_regions,
        _load,
        _gpu_ready,
        _available,
        _hf,
    ):
        bootstrapper = GroundingDinoBootstrapper(device="cuda:0")
        result = bootstrapper.run_on_frame(np.zeros((1080, 1920, 3), dtype=np.uint8), frame_idx=12)
        payload = result.to_payload()
        self.assertFalse(payload["enabled"])
        self.assertEqual(payload["status"], "foreground_too_broad")
        self.assertGreater(payload["foreground_ratio"], 0.9)


if __name__ == "__main__":
    unittest.main()

import sys
import types
import unittest
from unittest import mock

import numpy as np

from pipelines.frame_quality import (
    annotate_frame_quality,
    detector_policy_for_quality,
    summarize_detection_misses_by_quality,
)


def _cv2_stub():
    module = types.ModuleType("cv2")
    module.COLOR_BGR2GRAY = 0
    module.CV_64F = 0
    module.INTER_AREA = 0

    def cvt_color(image, _flag):
        return image.mean(axis=2).astype(np.float32)

    def resize(image, dsize, interpolation=None):
        out_w, out_h = dsize
        in_h, in_w = image.shape[:2]
        ys = np.clip((np.arange(out_h) * in_h / max(out_h, 1)).astype(int), 0, in_h - 1)
        xs = np.clip((np.arange(out_w) * in_w / max(out_w, 1)).astype(int), 0, in_w - 1)
        if image.ndim == 2:
            return image[ys][:, xs]
        return image[ys][:, xs, :]

    module.cvtColor = cvt_color
    module.resize = resize
    module.Laplacian = lambda image, _dtype: image.astype(np.float32)
    module.phaseCorrelate = lambda previous, current: ((3.0, 4.0), 1.0)
    return module


class FrameQualityTest(unittest.TestCase):
    def test_annotates_quality_labels_and_detector_policy(self):
        frames = []
        for idx in range(20):
            if idx < 3:
                image = np.zeros((32, 32, 3), dtype=np.uint8)
            else:
                pattern = ((np.indices((32, 32)).sum(axis=0) + idx) % 2) * 255
                image = np.repeat(pattern[:, :, None], 3, axis=2).astype(np.uint8)
            frames.append({"frame_idx": idx, "_frame_bgr": image})

        with mock.patch.dict(sys.modules, {"cv2": _cv2_stub()}):
            summary = annotate_frame_quality(frames, long_side=32)

        self.assertEqual(summary["frame_count"], 20)
        self.assertEqual(frames[0]["frame_quality"]["quality_label"], "severe_blur")
        self.assertEqual(frames[0]["frame_quality"]["detector_policy"], "bridge_measurements")
        self.assertEqual(frames[-1]["frame_quality"]["quality_label"], "sharp")

    def test_summarizes_detection_misses_by_quality(self):
        frames = [
            {
                "frame_quality": {"quality_label": "severe_blur"},
                "ball_state": {"state": "missing"},
                "detections": [{"active_player_candidate": True}],
            },
            {
                "frame_quality": {"quality_label": "sharp"},
                "ball_state": {"state": "observed"},
                "detections": [{"active_player_candidate": False}, {"active_player_candidate": True}],
            },
        ]

        summary = summarize_detection_misses_by_quality(frames)

        self.assertEqual(summary["severe_blur"]["ball_missing_rate"], 1.0)
        self.assertEqual(summary["sharp"]["ball_missing_rate"], 0.0)
        self.assertEqual(summary["sharp"]["avg_player_detections"], 2.0)

    def test_detector_policy_for_quality(self):
        self.assertEqual(detector_policy_for_quality("severe_blur"), "bridge_measurements")
        self.assertEqual(detector_policy_for_quality("blurred"), "downweight_measurements")
        self.assertEqual(detector_policy_for_quality("sharp"), "normal")


if __name__ == "__main__":
    unittest.main()

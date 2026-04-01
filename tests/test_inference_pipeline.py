import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pipelines.inference import (
    BALL_CLASS_ID,
    CalibrationResolver,
    FrameResultAdapter,
)


class _RecordingWriter:
    def __init__(self):
        self.rows = []

    def write(self, payload):
        self.rows.append(payload)


class CalibrationResolverTest(unittest.TestCase):
    def test_load_for_clip_prefers_sequence_when_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            calibration_path = tmp / "camera_calibration.json"
            calibration_path.write_text(
                json.dumps(
                    {
                        "demo_clip": {
                            "h_sequence": {
                                "3": [[1, 0, 10], [0, 1, 20], [0, 0, 1]],
                            }
                        }
                    }
                )
            )
            resolver = CalibrationResolver(calibration_path, tmp / "missing.json").load_for_clip("demo_clip")
            h_matrix = resolver.homography_for_frame(3)
            self.assertTrue(np.array_equal(h_matrix, np.array([[1, 0, 10], [0, 1, 20], [0, 0, 1]])))

    def test_load_for_clip_falls_back_to_legacy_global_matrix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            legacy_path = tmp / "calibration.json"
            legacy_path.write_text(json.dumps({"h_matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 1]]}))
            resolver = CalibrationResolver(tmp / "missing.json", legacy_path).load_for_clip("demo_clip")
            h_matrix = resolver.homography_for_frame(1)
            self.assertTrue(np.array_equal(h_matrix, np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]])))


class FrameResultAdapterTest(unittest.TestCase):
    def test_extract_ball_state_updates_last_ball_and_writes_event(self):
        boxes_xywh = [
            np.array([100.0, 120.0, 20.0, 20.0]),
            np.array([300.0, 220.0, 12.0, 12.0]),
        ]
        cls = [0, BALL_CLASS_ID]
        conf = [0.88, 0.67]
        writer = _RecordingWriter()
        last_ball_2d, ball_3d = FrameResultAdapter.extract_ball_state(
            boxes_xywh=boxes_xywh,
            cls=cls,
            conf=conf,
            last_ball_2d=np.array([0.0, 0.0]),
            h_matrix=np.eye(3),
            t_ms=250,
            writer=writer,
        )
        self.assertTrue(np.array_equal(last_ball_2d, np.array([300.0, 220.0])))
        self.assertIsNotNone(ball_3d)
        self.assertEqual(len(writer.rows), 1)
        self.assertEqual(writer.rows[0]["kind"], "ball")
        self.assertEqual(writer.rows[0]["t_ms"], 250)


if __name__ == "__main__":
    unittest.main()

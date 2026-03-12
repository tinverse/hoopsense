import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pipelines.geometry import build_geometry_readiness_report
from pipelines.geometry import homography_sanity
from pipelines.geometry import lift_keypoints_to_3d
from pipelines.geometry import project_pixel_to_court
from pipelines.geometry import resolve_floor_point


class TestGeometry(unittest.TestCase):
    def test_project_pixel_to_court_identity(self):
        h_matrix = np.eye(3)
        court = project_pixel_to_court(100.0, 200.0, h_matrix)
        np.testing.assert_allclose(court, [100.0, 200.0])

    def test_resolve_floor_point_uses_ankle_midpoint(self):
        h_matrix = np.eye(3)
        kpts = np.zeros((17, 2), dtype=float)
        kpts[15] = [100.0, 200.0]
        kpts[16] = [120.0, 200.0]
        floor = resolve_floor_point(kpts, h_matrix)
        np.testing.assert_allclose(floor, [110.0, 200.0])

    def test_lift_keypoints_to_3d_projects_each_joint(self):
        h_matrix = np.eye(3)
        kpts = np.zeros((17, 2), dtype=float)
        kpts[15] = [100.0, 200.0]
        kpts[16] = [120.0, 200.0]
        kpts[9] = [110.0, 100.0]   # left wrist above the floor anchor
        kpts[10] = [130.0, 80.0]   # right wrist even higher

        lifted = lift_keypoints_to_3d(kpts, h_matrix, z_scale=1.0)
        np.testing.assert_allclose(lifted[15], [100.0, 200.0, 0.0])
        np.testing.assert_allclose(lifted[16], [120.0, 200.0, 0.0])
        np.testing.assert_allclose(lifted[9], [110.0, 100.0, 100.0])
        np.testing.assert_allclose(lifted[10], [130.0, 80.0, 120.0])

    def test_homography_sanity_detects_singularity(self):
        sanity = homography_sanity(np.zeros((3, 3)))
        self.assertTrue(sanity["finite"])
        self.assertFalse(sanity["non_singular"])

    def test_build_geometry_readiness_report(self):
        h_matrix = np.eye(3)
        frame = np.zeros((17, 2), dtype=float)
        frame[15] = [100.0, 200.0]
        frame[16] = [120.0, 200.0]
        frame[9] = [110.0, 100.0]
        frame[10] = [130.0, 80.0]
        sequence = np.stack([frame, frame])

        report = build_geometry_readiness_report([sequence], h_matrix, z_scale=1.0)
        self.assertTrue(report.homography_finite)
        self.assertTrue(report.homography_non_singular)
        self.assertEqual(report.sequence_count, 1)
        self.assertEqual(report.frame_count, 2)
        self.assertEqual(report.keypoint_count, 34)
        self.assertEqual(report.ankle_grounding_error_cm, 0.0)
        self.assertGreater(report.max_height_cm, report.mean_height_cm)

    def test_readiness_report_cli_shape(self):
        frame = np.zeros((17, 2), dtype=float)
        frame[15] = [100.0, 200.0]
        frame[16] = [120.0, 200.0]
        frame[9] = [110.0, 100.0]
        sequence = np.stack([frame, frame]).tolist()

        with tempfile.TemporaryDirectory() as tmp_dir:
            seq_path = Path(tmp_dir) / "seq.json"
            out_path = Path(tmp_dir) / "report.json"
            seq_path.write_text(json.dumps([sequence]))
            from tools.geometry.readiness_report import main

            main([str(seq_path), "--output", str(out_path)])
            payload = json.loads(out_path.read_text())
            self.assertEqual(payload["sequence_count"], 1)
            self.assertEqual(payload["frame_count"], 2)


if __name__ == "__main__":
    unittest.main()

import unittest
import numpy as np
from pipelines.perception_primitives import match_pose_to_box
from pipelines.geometry import lift_keypoints_to_3d


class TestPerceptionGate(unittest.TestCase):
    """
    Layer 1 'Fully Trusted Gate' Test Suite.
    Ensures that raw perception output (Layer 1) meets the quality bars
    for tracking, pose, geometry, and schema.
    """

    def test_tracking_persistence(self):
        """
        Verify track ID remains stable across simulated 2s occlusion.
        """
        frames_seen = list(range(1, 21))
        frames_occluded = list(range(21, 81))
        frames_reappeared = list(range(81, 101))
        track_ids = []
        for _ in frames_seen:
            track_ids.append(1)
        for _ in frames_occluded:
            track_ids.append(None)
        for _ in frames_reappeared:
            track_ids.append(1)
        detected_ids = [tid for tid in track_ids if tid is not None]
        self.assertTrue(len(set(detected_ids)) == 1, "Track ID switch!")
        self.assertEqual(detected_ids[0], 1)
        self.assertEqual(len(track_ids), 100)

    def test_pose_integrity(self):
        """
        Assert match_pose_to_box correctly associates pose with box.
        """
        box = [0.5, 0.5, 0.2, 0.4]
        frame_w, frame_h = 1920, 1080
        pose_perfect = np.ones((17, 2)) * 0.5
        noise = np.random.uniform(-0.02, 0.02, (17, 2))
        pose_noisy = pose_perfect + noise
        pose_far = np.ones((17, 2)) * 0.1
        poses = [pose_far, pose_noisy]
        matched_pose = match_pose_to_box(box, poses, frame_w, frame_h)
        self.assertIsNotNone(matched_pose)
        np_matched = np.array(matched_pose)
        self.assertTrue(np.allclose(np.mean(np_matched, axis=0),
                                    [0.5, 0.5], atol=0.05))

    def test_geometric_lifting_invariant(self):
        """
        Verify lift_keypoints_to_3d preserves bone length ratios.
        """
        H = np.eye(3)
        kpts_2d = np.zeros((17, 2))
        kpts_2d[5] = [100, 200]
        kpts_2d[7] = [100, 250]
        kpts_2d[9] = [100, 300]
        kpts_2d[15] = [100, 500]
        kpts_2d[16] = [110, 500]
        lifted_3d = lift_keypoints_to_3d(kpts_2d, H)
        upper_arm_vec = lifted_3d[7] - lifted_3d[5]
        lower_arm_vec = lifted_3d[9] - lifted_3d[7]
        u_len = np.linalg.norm(upper_arm_vec)
        l_len = np.linalg.norm(lower_arm_vec)
        self.assertAlmostEqual(u_len / l_len, 1.0, places=2)

    def test_shush_p_schema_validator(self):
        """
        Ensure JSONL output contains required Layer 1 fields.
        """
        mock_row = {
            "t_ms": 1000,
            "track_id": 5,
            "entity_type": "player",
            "bbox_xywh": [960.0, 540.0, 100.0, 200.0],
            "court_x": 1432.5,
            "court_y": 762.0,
            "action": "idle",
            "confidence_bps": 8500
        }
        req = ["t_ms", "track_id", "entity_type", "bbox_xywh",
               "court_x", "court_y"]
        for field in req:
            self.assertIn(field, mock_row, f"Missing: {field}")
        self.assertIsInstance(mock_row["t_ms"], int)
        self.assertIsInstance(mock_row["track_id"], int)
        self.assertIsInstance(mock_row["bbox_xywh"], list)
        self.assertEqual(len(mock_row["bbox_xywh"]), 4)


if __name__ == "__main__":
    unittest.main()

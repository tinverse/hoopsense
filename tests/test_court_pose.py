import unittest

import numpy as np

from pipelines.court_pose import CourtPoseTracker, annotate_court_pose


class CourtPoseTest(unittest.TestCase):
    def test_calibrated_pose_uses_homography_and_foot_support(self):
        frame = {
            "frame_idx": 10,
            "_h_matrix": np.eye(3),
            "scene_prior": {
                "prior_status": "ready",
                "region_mask_shape": [2, 2],
                "region_mask_grid": [[0, 0], [1, 1]],
            },
            "detections": [
                {
                    "class_id": 0,
                    "bbox_xyxy": [40.0, 20.0, 80.0, 90.0],
                    "keypoints_xy": [[0.0, 0.0] for _ in range(17)],
                    "keypoints_conf": [0.0 for _ in range(17)],
                }
            ],
        }
        frame["detections"][0]["keypoints_xy"][15] = [55.0, 80.0]
        frame["detections"][0]["keypoints_xy"][16] = [65.0, 82.0]
        frame["detections"][0]["keypoints_conf"][15] = 0.9
        frame["detections"][0]["keypoints_conf"][16] = 0.8

        pose = CourtPoseTracker().update(frame, video_meta={"width": 100, "height": 100})

        payload = pose.to_payload()
        self.assertEqual(payload["state"], "calibrated")
        self.assertGreater(payload["confidence"], 0.75)
        self.assertEqual(payload["foot_anchor_count"], 1)
        self.assertEqual(payload["supported_foot_anchor_count"], 1)
        self.assertIn("calibrated_homography", payload["visible_features"])

    def test_weak_scene_pose_without_homography(self):
        frame = {
            "frame_idx": 1,
            "_h_matrix": None,
            "scene_prior": {
                "prior_status": "ready",
                "region_mask_shape": [1, 1],
                "region_mask_grid": [[1]],
            },
            "detections": [{"class_id": 0, "bbox_xyxy": [10.0, 20.0, 40.0, 80.0]}],
        }

        pose = CourtPoseTracker().update(frame, video_meta={"width": 100, "height": 100})

        self.assertEqual(pose.state, "weak_scene_pose")
        self.assertGreater(pose.confidence, 0.3)
        self.assertEqual(pose.supported_foot_anchor_count, 1)

    def test_carries_previous_pose_when_features_disappear(self):
        tracker = CourtPoseTracker(carry_max_age_frames=10)
        first = tracker.update(
            {"frame_idx": 1, "_h_matrix": np.eye(3), "detections": []},
            video_meta={"width": 100, "height": 100},
        )
        second = tracker.update(
            {"frame_idx": 2, "_h_matrix": None, "detections": []},
            video_meta={"width": 100, "height": 100},
        )

        self.assertEqual(first.state, "calibrated")
        self.assertEqual(second.state, "carried")
        self.assertLess(second.confidence, first.confidence)
        self.assertEqual(second.temporal_age_frames, 1)

    def test_annotate_court_pose_adds_summary_and_frame_payload(self):
        frames = [
            {"frame_idx": 1, "_h_matrix": np.eye(3), "detections": []},
            {"frame_idx": 2, "_h_matrix": None, "detections": []},
        ]

        summary = annotate_court_pose(frames, video_meta={"width": 100, "height": 100})

        self.assertEqual(summary["frame_count"], 2)
        self.assertIn("court_pose", frames[0])
        self.assertEqual(frames[1]["court_pose"]["state"], "carried")


if __name__ == "__main__":
    unittest.main()

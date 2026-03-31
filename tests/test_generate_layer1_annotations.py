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

from tools.review.labeller.generate_layer1_annotations import (
    _score_identity_link,
    annotate_active_players,
    estimate_uniform_bucket,
    repair_short_track_gaps,
    score_active_player,
    smooth_track_motion,
)


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


class ActivePlayerScoreTest(unittest.TestCase):
    def test_score_active_player_accepts_central_persistent_detection(self):
        score_info = score_active_player(
            {
                "confidence": 0.91,
                "bbox_xyxy": [200.0, 120.0, 280.0, 320.0],
                "court_xy": [1200.0, 700.0],
            },
            frame_width=640,
            frame_height=480,
            track_frame_count=4,
        )
        self.assertTrue(score_info["candidate"])
        self.assertGreater(score_info["score"], 0.55)
        self.assertTrue(score_info["reasons"]["court_in_bounds"])

    def test_score_active_player_rejects_tiny_edge_detection(self):
        score_info = score_active_player(
            {
                "confidence": 0.28,
                "bbox_xyxy": [2.0, 30.0, 18.0, 60.0],
                "court_xy": [3100.0, 1700.0],
            },
            frame_width=640,
            frame_height=480,
            track_frame_count=1,
        )
        self.assertFalse(score_info["candidate"])
        self.assertLess(score_info["score"], 0.55)
        self.assertFalse(score_info["reasons"]["court_in_bounds"])

    def test_score_active_player_rewards_motion(self):
        static = score_active_player(
            {
                "confidence": 0.7,
                "bbox_xyxy": [200.0, 120.0, 280.0, 320.0],
                "court_xy": [1200.0, 700.0],
                "motion_speed_px": 0.0,
            },
            frame_width=640,
            frame_height=480,
            track_frame_count=2,
        )
        moving = score_active_player(
            {
                "confidence": 0.7,
                "bbox_xyxy": [200.0, 120.0, 280.0, 320.0],
                "court_xy": [1200.0, 700.0],
                "motion_speed_px": 12.0,
            },
            frame_width=640,
            frame_height=480,
            track_frame_count=2,
        )
        self.assertGreater(moving["score"], static["score"])


class TrackRepairTest(unittest.TestCase):
    def test_repair_short_track_gaps_interpolates_missing_frame(self):
        frames = [
            {
                "frame_idx": 0,
                "t_ms": 0,
                "detections": [{
                    "track_id": 7,
                    "class_id": 0,
                    "class_name": "person",
                    "confidence": 0.9,
                    "bbox_xyxy": [100.0, 80.0, 160.0, 260.0],
                    "bbox_xywh": [130.0, 170.0, 60.0, 180.0],
                    "court_xy": [1000.0, 600.0],
                    "active_player_candidate": True,
                    "uniform_bucket": "dark",
                }],
            },
            {"frame_idx": 1, "t_ms": 33, "detections": []},
            {
                "frame_idx": 2,
                "t_ms": 66,
                "detections": [{
                    "track_id": 7,
                    "class_id": 0,
                    "class_name": "person",
                    "confidence": 0.8,
                    "bbox_xyxy": [140.0, 90.0, 200.0, 270.0],
                    "bbox_xywh": [170.0, 180.0, 60.0, 180.0],
                    "court_xy": [1100.0, 650.0],
                    "active_player_candidate": True,
                    "uniform_bucket": "dark",
                }],
            },
        ]
        repaired = repair_short_track_gaps(frames, max_gap=2)
        inserted = repaired[1]["detections"]
        self.assertEqual(len(inserted), 1)
        self.assertTrue(inserted[0]["synthesized"])
        self.assertEqual(inserted[0]["track_id"], 7)
        self.assertEqual(inserted[0]["repair_source"]["kind"], "short_gap_interpolation")
        self.assertAlmostEqual(inserted[0]["bbox_xywh"][0], 150.0)
        self.assertAlmostEqual(inserted[0]["court_xy"][0], 1050.0)

    def test_repair_short_track_gaps_skips_non_candidates(self):
        frames = [
            {
                "frame_idx": 0,
                "t_ms": 0,
                "detections": [{
                    "track_id": 7,
                    "class_id": 0,
                    "class_name": "person",
                    "confidence": 0.9,
                    "bbox_xyxy": [100.0, 80.0, 160.0, 260.0],
                    "bbox_xywh": [130.0, 170.0, 60.0, 180.0],
                    "active_player_candidate": False,
                }],
            },
            {"frame_idx": 1, "t_ms": 33, "detections": []},
            {
                "frame_idx": 2,
                "t_ms": 66,
                "detections": [{
                    "track_id": 7,
                    "class_id": 0,
                    "class_name": "person",
                    "confidence": 0.8,
                    "bbox_xyxy": [140.0, 90.0, 200.0, 270.0],
                    "bbox_xywh": [170.0, 180.0, 60.0, 180.0],
                    "active_player_candidate": True,
                }],
            },
        ]
        repaired = repair_short_track_gaps(frames, max_gap=2)
        self.assertEqual(repaired[1]["detections"], [])

    def test_score_identity_link_rejects_conflicting_uniforms(self):
        link = _score_identity_link(
            {
                "active_player_candidate": True,
                "uniform_bucket": "dark",
                "bbox_xywh": [100.0, 120.0, 50.0, 150.0],
                "smoothed_center_xy": [100.0, 120.0],
                "smoothed_velocity_xy": [20.0, 0.0],
                "confidence": 0.8,
            },
            {
                "active_player_candidate": True,
                "uniform_bucket": "light",
                "bbox_xywh": [102.0, 120.0, 50.0, 150.0],
                "smoothed_center_xy": [102.0, 120.0],
                "smoothed_velocity_xy": [18.0, 0.0],
                "confidence": 0.8,
            },
            gap_frames=1,
            fps=30.0,
        )
        self.assertIsNone(link)

    def test_repair_short_track_gaps_bridges_short_identity_switch(self):
        frames = [
            {
                "frame_idx": 0,
                "t_ms": 0,
                "detections": [{
                    "track_id": 3,
                    "class_id": 0,
                    "class_name": "person",
                    "confidence": 0.92,
                    "bbox_xyxy": [100.0, 80.0, 160.0, 260.0],
                    "bbox_xywh": [130.0, 170.0, 60.0, 180.0],
                    "smoothed_center_xy": [130.0, 170.0],
                    "smoothed_velocity_xy": [120.0, 0.0],
                    "smoothed_bbox_xywh": [130.0, 170.0, 60.0, 180.0],
                    "court_xy": [1000.0, 600.0],
                    "active_player_candidate": True,
                    "uniform_bucket": "dark",
                }],
            },
            {"frame_idx": 1, "t_ms": 33, "detections": []},
            {
                "frame_idx": 2,
                "t_ms": 66,
                "detections": [{
                    "track_id": 19,
                    "class_id": 0,
                    "class_name": "person",
                    "confidence": 0.9,
                    "bbox_xyxy": [108.0, 82.0, 168.0, 262.0],
                    "bbox_xywh": [138.0, 172.0, 60.0, 180.0],
                    "smoothed_center_xy": [138.0, 172.0],
                    "smoothed_velocity_xy": [118.0, 1.0],
                    "smoothed_bbox_xywh": [138.0, 172.0, 60.0, 180.0],
                    "court_xy": [1015.0, 605.0],
                    "active_player_candidate": True,
                    "uniform_bucket": "dark",
                }],
            },
        ]
        repaired = repair_short_track_gaps(
            frames,
            {"fps": 30.0, "frame_count": 3, "width": 640, "height": 480},
            max_gap=2,
        )
        inserted = repaired[1]["detections"]
        self.assertEqual(len(inserted), 1)
        self.assertTrue(inserted[0]["synthesized"])
        self.assertEqual(inserted[0]["identity_track_id"], 3)
        self.assertEqual(inserted[0]["identity_repair"]["kind"], "short_gap_identity_bridge")
        successor = repaired[2]["detections"][0]
        self.assertEqual(successor["track_id"], 19)
        self.assertEqual(successor["identity_track_id"], 3)
        self.assertEqual(successor["identity_track_source"], "repaired")
        self.assertEqual(successor["identity_repair"]["predecessor_track_id"], 3)
        self.assertEqual(successor["identity_repair"]["successor_track_id"], 19)


class ActivePlayerAnnotationTest(unittest.TestCase):
    def test_annotate_active_players_adds_score_fields(self):
        frames = [{
            "frame_idx": 0,
            "t_ms": 0,
            "detections": [{
                "track_id": 3,
                "confidence": 0.9,
                "bbox_xyxy": [220.0, 100.0, 300.0, 340.0],
                "court_xy": [1200.0, 800.0],
            }],
        }]
        annotated = annotate_active_players(
            frames,
            {"width": 640, "height": 480, "fps": 30.0, "frame_count": 1},
        )
        detection = annotated[0]["detections"][0]
        self.assertIn("active_player_score", detection)
        self.assertIn("active_player_candidate", detection)
        self.assertIn("active_player_reasons", detection)


class KalmanMotionTest(unittest.TestCase):
    def test_smooth_track_motion_adds_smoothed_fields(self):
        frames = [
            {
                "frame_idx": 0,
                "t_ms": 0,
                "detections": [{
                    "track_id": 3,
                    "bbox_xywh": [100.0, 120.0, 50.0, 150.0],
                }],
            },
            {
                "frame_idx": 1,
                "t_ms": 33,
                "detections": [{
                    "track_id": 3,
                    "bbox_xywh": [110.0, 122.0, 50.0, 150.0],
                }],
            },
        ]
        smoothed = smooth_track_motion(frames, {"fps": 30.0, "frame_count": 2, "width": 640, "height": 480})
        first = smoothed[0]["detections"][0]
        second = smoothed[1]["detections"][0]
        self.assertIn("smoothed_center_xy", first)
        self.assertIn("smoothed_velocity_xy", second)
        self.assertIn("motion_speed_px", second)
        self.assertGreaterEqual(second["motion_speed_px"], 0.0)


if __name__ == "__main__":
    unittest.main()

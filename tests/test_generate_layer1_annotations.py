import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np


ultralytics_stub = types.ModuleType("ultralytics")
ultralytics_stub.YOLO = object
sys.modules.setdefault("ultralytics", ultralytics_stub)

cv2_stub = types.ModuleType("cv2")
cv2_stub.COLOR_BGR2GRAY = 0
cv2_stub.CV_64F = 0
cv2_stub.INTER_AREA = 0


def cvt_color_to_gray(image, _flag):
    return image.mean(axis=2).astype(image.dtype)


cv2_stub.cvtColor = cvt_color_to_gray
cv2_stub.Laplacian = lambda image, _dtype: image.astype(float)
cv2_stub.resize = lambda image, dsize, interpolation=None: np.resize(image, (dsize[1], dsize[0]))
sys.modules.setdefault("cv2", cv2_stub)

from tools.review.labeller.generate_layer1_annotations import (
    _normalize_jersey_text,
    _resolve_identity_jersey_consensus,
    _score_identity_link,
    annotate_team_appearance_consistency,
    annotate_active_players,
    annotate_continuity_segments,
    annotate_identity_jersey_numbers,
    annotate_live_play,
    estimate_uniform_bucket,
    estimate_torso_color_histogram,
    histogram_intersection_distance,
    load_layer1_identity_policy,
    repair_short_track_gaps,
    score_active_player,
    score_live_play_frame,
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

    def test_estimate_torso_color_histogram_is_normalized(self):
        crop = np.zeros((10, 10, 3), dtype=np.uint8)
        crop[:, :] = [255, 255, 255]
        hist = estimate_torso_color_histogram(crop)
        self.assertIsNotNone(hist)
        self.assertEqual(len(hist), 8)
        self.assertAlmostEqual(sum(hist), 1.0, places=5)

    def test_histogram_intersection_distance_distinguishes_colors(self):
        dark = estimate_torso_color_histogram(np.zeros((8, 8, 3), dtype=np.uint8))
        bright = estimate_torso_color_histogram(np.full((8, 8, 3), 255, dtype=np.uint8))
        self.assertAlmostEqual(histogram_intersection_distance(dark, dark), 0.0, places=5)
        self.assertGreater(histogram_intersection_distance(dark, bright), 0.9)


class IdentityPolicyTest(unittest.TestCase):
    def test_load_layer1_identity_policy_loads_default_contract(self):
        policy = load_layer1_identity_policy()
        self.assertEqual(policy["kind"], "layer1_identity_policy")
        self.assertEqual(policy["assumptions"]["disappearance_default"], "occlusion_or_detector_miss")
        self.assertTrue(policy["continuity"]["reset_identity_on_discontinuity"])

    def test_load_layer1_identity_policy_rejects_missing_sections(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad_policy.yaml"
            path.write_text("version: '1.0'\nkind: layer1_identity_policy\ncontinuity: {}\n")
            with self.assertRaises(ValueError):
                load_layer1_identity_policy(path)


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

    def test_score_active_player_penalizes_edge_low_motion_appearance_mismatch(self):
        baseline = score_active_player(
            {
                "confidence": 0.7,
                "bbox_xyxy": [4.0, 100.0, 90.0, 320.0],
                "court_xy": [1200.0, 700.0],
                "motion_speed_px": 0.0,
                "appearance_team_distance": 0.7,
            },
            frame_width=640,
            frame_height=480,
            track_frame_count=2,
        )
        moving = score_active_player(
            {
                "confidence": 0.7,
                "bbox_xyxy": [4.0, 100.0, 90.0, 320.0],
                "court_xy": [1200.0, 700.0],
                "motion_speed_px": 9.0,
                "appearance_team_distance": 0.7,
            },
            frame_width=640,
            frame_height=480,
            track_frame_count=2,
        )
        self.assertGreater(baseline["reasons"]["appearance_penalty"], 0.0)
        self.assertEqual(moving["reasons"]["appearance_penalty"], 0.0)


class AppearanceConsistencyTest(unittest.TestCase):
    def test_annotate_team_appearance_consistency_assigns_distance(self):
        dark_hist = estimate_torso_color_histogram(np.zeros((8, 8, 3), dtype=np.uint8))
        bright_hist = estimate_torso_color_histogram(np.full((8, 8, 3), 255, dtype=np.uint8))
        frames = [
            {
                "frame_idx": 0,
                "detections": [
                    {
                        "track_id": 1,
                        "appearance_histogram_rgbq": dark_hist,
                        "active_player_candidate": True,
                        "motion_speed_px": 8.0,
                        "uniform_bucket": "dark",
                        "active_player_reasons": {"court_in_bounds": True, "edge_penalty": 0.0},
                    },
                    {
                        "track_id": 2,
                        "appearance_histogram_rgbq": bright_hist,
                        "active_player_candidate": False,
                        "motion_speed_px": 0.0,
                        "uniform_bucket": "dark",
                        "active_player_reasons": {"court_in_bounds": True, "edge_penalty": 0.2},
                    },
                ],
            }
        ]
        summary = annotate_team_appearance_consistency(frames)
        self.assertEqual(summary["prototype_count"], 1)
        self.assertEqual(frames[0]["detections"][0]["appearance_team_bucket"], "dark")
        self.assertAlmostEqual(frames[0]["detections"][0]["appearance_team_distance"], 0.0, places=5)
        self.assertGreater(frames[0]["detections"][1]["appearance_team_distance"], 0.9)


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

    def test_repair_short_track_gaps_does_not_bridge_across_discontinuity_segments(self):
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
                    "continuity_segment_id": 0,
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
                    "continuity_segment_id": 1,
                }],
            },
        ]
        repaired = repair_short_track_gaps(
            frames,
            {"fps": 30.0, "frame_count": 3, "width": 640, "height": 480},
            max_gap=2,
        )
        self.assertEqual(repaired[1]["detections"], [])
        successor = repaired[2]["detections"][0]
        self.assertEqual(successor["identity_track_id"], 19)
        self.assertIsNone(successor["identity_repair"])


class JerseyIdentityTest(unittest.TestCase):
    def test_normalize_jersey_text_handles_common_ocr_confusions(self):
        self.assertEqual(_normalize_jersey_text("1O"), "10")
        self.assertEqual(_normalize_jersey_text("O8"), "8")
        self.assertIsNone(_normalize_jersey_text("ABC"))
        self.assertIsNone(_normalize_jersey_text("000"))

    def test_resolve_identity_jersey_consensus_requires_votes_and_share(self):
        consensus = _resolve_identity_jersey_consensus(
            [
                {"candidate": "24", "ocr_confidence": 0.9, "sharpness": 90.0},
                {"candidate": "24", "ocr_confidence": 0.8, "sharpness": 80.0},
                {"candidate": "84", "ocr_confidence": 0.2, "sharpness": 30.0},
            ]
        )
        self.assertIsNotNone(consensus)
        self.assertEqual(consensus["number"], "24")
        self.assertGreaterEqual(consensus["confidence"], 0.62)

    def test_annotate_identity_jersey_numbers_applies_consensus_to_identity(self):
        import tools.review.labeller.generate_layer1_annotations as gla

        original_reader = gla.get_easyocr_reader
        original_collect = gla._collect_jersey_evidence
        try:
            gla.get_easyocr_reader = lambda: object()

            def fake_collect(_reader, detection):
                mapping = {
                    3: {"candidate": "24", "raw_text": "24", "ocr_confidence": 0.92, "sharpness": 70.0, "source": "stub"},
                    19: {"candidate": "24", "raw_text": "24", "ocr_confidence": 0.81, "sharpness": 65.0, "source": "stub"},
                }
                evidence = mapping.get(detection.get("track_id"))
                if evidence:
                    detection["jersey_number_evidence"] = evidence
                return evidence

            gla._collect_jersey_evidence = fake_collect
            frames = [
                {"frame_idx": 0, "detections": [{"track_id": 3, "identity_track_id": 3, "_jersey_crop_bgr": np.zeros((24, 24, 3), dtype=np.uint8), "_jersey_crop_sharpness": 80.0}]},
                {"frame_idx": 2, "detections": [{"track_id": 19, "identity_track_id": 3, "_jersey_crop_bgr": np.zeros((24, 24, 3), dtype=np.uint8), "_jersey_crop_sharpness": 80.0}]},
            ]
            summary = annotate_identity_jersey_numbers(frames)
            self.assertTrue(summary["reader_available"])
            self.assertEqual(summary["identity_count_with_consensus"], 1)
            self.assertEqual(frames[0]["detections"][0]["identity_jersey_number"], "24")
            self.assertEqual(frames[1]["detections"][0]["identity_jersey_number"], "24")
            self.assertEqual(frames[1]["detections"][0]["identity_jersey_evidence_count"], 2)
        finally:
            gla.get_easyocr_reader = original_reader
            gla._collect_jersey_evidence = original_collect


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


class LivePlayGateTest(unittest.TestCase):
    def test_score_live_play_frame_prefers_multi_player_motion(self):
        live_info = score_live_play_frame(
            {
                "frame_idx": 10,
                "detections": [
                    {
                        "track_id": 1,
                        "active_player_candidate": True,
                        "motion_speed_px": 85.0,
                        "active_player_reasons": {"court_in_bounds": True, "edge_penalty": 0.0},
                    },
                    {
                        "track_id": 2,
                        "active_player_candidate": True,
                        "motion_speed_px": 62.0,
                        "active_player_reasons": {"court_in_bounds": True, "edge_penalty": 0.0},
                    },
                    {
                        "track_id": 3,
                        "active_player_candidate": True,
                        "motion_speed_px": 41.0,
                        "active_player_reasons": {"court_in_bounds": True, "edge_penalty": 0.0},
                    },
                ],
            }
        )
        self.assertGreaterEqual(live_info["score"], 0.62)
        self.assertEqual(live_info["label"], "live_play")

    def test_score_live_play_frame_marks_idle_sparse_frame_dead_ball(self):
        live_info = score_live_play_frame(
            {
                "frame_idx": 3,
                "detections": [
                    {
                        "track_id": 6,
                        "active_player_candidate": False,
                        "motion_speed_px": 0.0,
                        "active_player_reasons": {"court_in_bounds": False, "edge_penalty": 0.2},
                    },
                    {
                        "track_id": 7,
                        "active_player_candidate": True,
                        "motion_speed_px": 0.0,
                        "active_player_reasons": {"court_in_bounds": False, "edge_penalty": 0.2},
                    },
                ],
            }
        )
        self.assertLessEqual(live_info["score"], 0.38)
        self.assertEqual(live_info["label"], "dead_ball")

    def test_score_live_play_frame_records_ball_signal_and_lifts_score_conservatively(self):
        baseline = score_live_play_frame(
            {
                "frame_idx": 12,
                "detections": [
                    {
                        "track_id": 1,
                        "active_player_candidate": True,
                        "motion_speed_px": 24.0,
                        "active_player_reasons": {"court_in_bounds": True, "edge_penalty": 0.0},
                    },
                    {
                        "track_id": 2,
                        "active_player_candidate": True,
                        "motion_speed_px": 19.0,
                        "active_player_reasons": {"court_in_bounds": True, "edge_penalty": 0.0},
                    },
                ],
            }
        )
        with_ball = score_live_play_frame(
            {
                "frame_idx": 12,
                "detections": [
                    {
                        "track_id": 1,
                        "active_player_candidate": True,
                        "motion_speed_px": 24.0,
                        "active_player_reasons": {"court_in_bounds": True, "edge_penalty": 0.0},
                    },
                    {
                        "track_id": 2,
                        "active_player_candidate": True,
                        "motion_speed_px": 19.0,
                        "active_player_reasons": {"court_in_bounds": True, "edge_penalty": 0.0},
                    },
                ],
                "ball_detection": {
                    "confidence": 0.81,
                    "bbox_xyxy": [310.0, 180.0, 326.0, 196.0],
                    "bbox_xywh": [318.0, 188.0, 16.0, 16.0],
                    "center_xy": [318.0, 188.0],
                },
            }
        )
        self.assertFalse(baseline["reasons"]["ball_signal_present"])
        self.assertTrue(with_ball["reasons"]["ball_signal_present"])
        self.assertGreater(with_ball["score"], baseline["score"])

    def test_annotate_live_play_stitches_segment_with_hysteresis(self):
        live_frame = {
            "detections": [
                {
                    "track_id": 1,
                    "active_player_candidate": True,
                    "motion_speed_px": 75.0,
                    "active_player_reasons": {"court_in_bounds": True, "edge_penalty": 0.0},
                },
                {
                    "track_id": 2,
                    "active_player_candidate": True,
                    "motion_speed_px": 68.0,
                    "active_player_reasons": {"court_in_bounds": True, "edge_penalty": 0.0},
                },
                {
                    "track_id": 3,
                    "active_player_candidate": True,
                    "motion_speed_px": 54.0,
                    "active_player_reasons": {"court_in_bounds": True, "edge_penalty": 0.0},
                },
            ],
        }
        dip_frame = {
            "detections": [
                {
                    "track_id": 1,
                    "active_player_candidate": True,
                    "motion_speed_px": 4.0,
                    "active_player_reasons": {"court_in_bounds": True, "edge_penalty": 0.0},
                },
                {
                    "track_id": 2,
                    "active_player_candidate": True,
                    "motion_speed_px": 3.0,
                    "active_player_reasons": {"court_in_bounds": True, "edge_penalty": 0.0},
                },
            ],
        }
        frames = []
        for frame_idx in range(5):
            frame = {"frame_idx": frame_idx, "t_ms": frame_idx * 33, **live_frame}
            frames.append(frame)
        frames.append({"frame_idx": 5, "t_ms": 165, **dip_frame})
        summary = annotate_live_play(frames, {"fps": 30.0})
        self.assertEqual(frames[4]["live_play_label"], "live_play")
        self.assertEqual(frames[5]["live_play_label"], "uncertain")
        self.assertGreaterEqual(len(summary["segments"]), 1)
        self.assertIn("live_play", [segment["label"] for segment in summary["segments"]])


class ContinuitySegmentationTest(unittest.TestCase):
    def test_annotate_continuity_segments_marks_visual_cut(self):
        dark_sig = np.zeros((18, 32), dtype=np.float32).tolist()
        bright_sig = np.ones((18, 32), dtype=np.float32).tolist()
        frames = [
            {
                "frame_idx": 0,
                "t_ms": 0,
                "_frame_visual_signature": dark_sig,
                "detections": [{"track_id": 1}],
            },
            {
                "frame_idx": 1,
                "t_ms": 33,
                "_frame_visual_signature": dark_sig,
                "detections": [{"track_id": 1}],
            },
            {
                "frame_idx": 2,
                "t_ms": 66,
                "_frame_visual_signature": bright_sig,
                "detections": [{"track_id": 7}],
            },
        ]
        summary = annotate_continuity_segments(frames)
        self.assertEqual(len(summary["segments"]), 2)
        self.assertEqual(frames[1]["continuity_segment_id"], 0)
        self.assertEqual(frames[2]["continuity_segment_id"], 1)
        self.assertEqual(frames[2]["discontinuity_label"], "discontinuity")


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

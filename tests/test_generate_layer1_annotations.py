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
cv2_stub.INTER_NEAREST = 0


def cvt_color_to_gray(image, _flag):
    return image.mean(axis=2).astype(image.dtype)


cv2_stub.cvtColor = cvt_color_to_gray
cv2_stub.Laplacian = lambda image, _dtype: image.astype(float)
def resize_nearest(image, dsize, interpolation=None):
    out_w, out_h = dsize
    in_h, in_w = image.shape[:2]
    ys = np.clip((np.arange(out_h) * in_h / max(out_h, 1)).astype(int), 0, in_h - 1)
    xs = np.clip((np.arange(out_w) * in_w / max(out_w, 1)).astype(int), 0, in_w - 1)
    if image.ndim == 2:
        return image[ys][:, xs]
    return image[ys][:, xs, :]

cv2_stub.resize = resize_nearest
sys.modules.setdefault("cv2", cv2_stub)

import tools.review.labeller.generate_layer1_annotations as gla
import tools.review.labeller.sam_refiner as sam_refiner
from tools.review.labeller.sam_refiner import SamRefineResult

from tools.review.labeller.generate_layer1_annotations import (
    ON_COURT_SCORE_THRESHOLD,
    _ball_detection_needs_search,
    _normalize_jersey_text,
    _player_ball_search_rois,
    _run_ball_predictive_search,
    _run_ball_roi_fallback,
    _serialize_ball_candidates,
    _update_ball_predictive_state,
    _resolve_identity_jersey_consensus,
    _score_identity_link,
    _apply_grounding_mask,
    annotate_team_appearance_consistency,
    annotate_active_players,
    annotate_ball_state,
    annotate_bootstrap_contexts,
    annotate_continuity_segments,
    annotate_identity_jersey_numbers,
    annotate_live_play,
    annotate_scene_discovery_contracts,
    mark_tracking_collapse_reground,
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
        self.assertTrue(score_info["on_court_candidate"])
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
        self.assertFalse(score_info["on_court_candidate"])
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

    def test_score_active_player_keeps_near_camera_player_when_pose_and_persistence_are_strong(self):
        keypoints_xy = [[0.0, 0.0] for _ in range(17)]
        keypoints_conf = [0.0 for _ in range(17)]
        for idx, pt in {
            5: [42.0, 120.0],
            6: [140.0, 118.0],
            11: [55.0, 260.0],
            12: [128.0, 258.0],
            13: [62.0, 360.0],
            14: [126.0, 362.0],
            15: [64.0, 468.0],
            16: [124.0, 470.0],
        }.items():
            keypoints_xy[idx] = pt
            keypoints_conf[idx] = 0.92

        score_info = score_active_player(
            {
                "confidence": 0.84,
                "bbox_xyxy": [0.0, 80.0, 170.0, 476.0],
                "court_xy": [1150.0, 730.0],
                "court_foot_xy": [1110.0, 760.0],
                "motion_speed_px": 8.5,
                "keypoints_xy": keypoints_xy,
                "keypoints_conf": keypoints_conf,
            },
            frame_width=640,
            frame_height=480,
            track_frame_count=5,
        )
        self.assertTrue(score_info["on_court_candidate"])
        self.assertGreater(score_info["reasons"]["pose_coherence"], 0.55)
        self.assertLess(score_info["reasons"]["spectator_risk"], 0.5)

    def test_score_active_player_penalizes_wide_sparse_merged_detection(self):
        score_info = score_active_player(
            {
                "confidence": 0.74,
                "bbox_xyxy": [120.0, 140.0, 390.0, 340.0],
                "court_xy": [1240.0, 780.0],
                "court_foot_xy": [1240.0, 860.0],
                "motion_speed_px": 1.0,
                "keypoints_xy": [[0.0, 0.0] for _ in range(17)],
                "keypoints_conf": [0.0 for _ in range(17)],
            },
            frame_width=640,
            frame_height=480,
            track_frame_count=1,
        )
        self.assertGreater(score_info["reasons"]["merge_risk"], 0.45)
        self.assertLess(score_info["on_court_score"], 0.6)

    def test_score_active_player_discounts_shared_pan_motion_without_grounding(self):
        score_info = score_active_player(
            {
                "confidence": 0.82,
                "bbox_xyxy": [190.0, 500.0, 296.0, 686.0],
                "motion_speed_px": 166.0,
                "smoothed_velocity_xy": [166.0, 0.0],
                "keypoints_xy": [[10.0, 10.0] for _ in range(17)],
                "keypoints_conf": [0.95 for _ in range(17)],
            },
            frame_width=640,
            frame_height=480,
            track_frame_count=37,
            frame_motion_context={
                "coherent_velocity_xy": [160.0, 0.0],
                "coherent_speed_px": 160.0,
                "shared_motion": True,
            },
        )
        self.assertTrue(score_info["reasons"]["frame_shared_motion"])
        self.assertGreater(score_info["reasons"]["ungrounded_shared_motion_penalty"], 0.0)
        self.assertLess(score_info["on_court_score"], ON_COURT_SCORE_THRESHOLD)


class BallStateTest(unittest.TestCase):
    def test_serialize_ball_candidates_preserves_sources_and_sort_order(self):
        detections = [
            {"class_id": 32, "class_name": "sports_ball", "confidence": 0.11, "bbox_xyxy": [0, 0, 4, 4], "bbox_xywh": [2, 2, 4, 4], "ball_detection_source": "full_frame"},
            {"class_id": 32, "class_name": "sports_ball", "confidence": 0.41, "bbox_xyxy": [5, 5, 9, 9], "bbox_xywh": [7, 7, 4, 4], "ball_detection_source": "player_local_rois_v2"},
        ]
        raw = _serialize_ball_candidates(detections)
        self.assertEqual(len(raw), 2)
        self.assertEqual(raw[0]["source"], "player_local_rois_v2")
        self.assertGreater(raw[0]["confidence"], raw[1]["confidence"])

    def test_player_ball_search_rois_build_local_windows(self):
        rois = _player_ball_search_rois(
            (100, 200, 3),
            [
                {"bbox_xyxy": [20, 30, 40, 80]},
                {"bbox_xyxy": [100, 25, 130, 78]},
            ],
        )
        self.assertEqual(len(rois), 2)
        self.assertLessEqual(rois[0][0], 20)
        self.assertLessEqual(rois[0][1], 30)
        self.assertGreaterEqual(rois[0][2], 40)
        self.assertGreaterEqual(rois[0][3], 78)

    def test_run_ball_roi_fallback_remaps_detections(self):
        class _FakeBoxes:
            cls = np.array([32], dtype=np.float32)
            conf = np.array([0.55], dtype=np.float32)
            id = None
            xyxy = np.array([[3.0, 4.0, 7.0, 8.0]], dtype=np.float32)
            xywh = np.array([[5.0, 6.0, 4.0, 4.0]], dtype=np.float32)

            def __len__(self):
                return 1

        class _FakeResult:
            boxes = _FakeBoxes()
            keypoints = None

        class _FakeBallModel:
            names = {32: "sports_ball"}

            def predict(self, crop, classes, conf, device, verbose):
                return [_FakeResult()]

        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        detections, summary = _run_ball_roi_fallback(
            _FakeBallModel(),
            frame,
            [{"bbox_xyxy": [20, 30, 40, 80]}],
            device="cpu",
            h_matrix=None,
        )
        self.assertTrue(summary["triggered"])
        self.assertEqual(summary["candidate_count"], 1)
        self.assertEqual(detections[0]["ball_detection_source"], "player_local_rois_v2")
        self.assertGreater(detections[0]["bbox_xyxy"][0], 0.0)
        self.assertEqual(len(summary["roi_bboxes_xyxy"]), 1)

    def test_update_ball_predictive_state_assigns_dribble_like_mode(self):
        state = _update_ball_predictive_state(
            {
                "center_xy": [50.0, 70.0],
                "last_seen_frame_idx": 0,
            },
            {
                "confidence": 0.62,
                "center_xy": [54.0, 116.0],
            },
            [
                {
                    "track_id": 17,
                    "bbox_xyxy": [20.0, 30.0, 80.0, 120.0],
                },
            ],
            1,
        )
        self.assertEqual(state["motion_mode"], "dribble_like")
        self.assertEqual(state["nearby_player_track_id"], 17)
        self.assertGreater(state["velocity_xy"][1], 0.0)

    def test_run_ball_predictive_search_remaps_detections(self):
        class _FakeBoxes:
            cls = np.array([32], dtype=np.float32)
            conf = np.array([0.58], dtype=np.float32)
            id = None
            xyxy = np.array([[4.0, 5.0, 10.0, 11.0]], dtype=np.float32)
            xywh = np.array([[7.0, 8.0, 6.0, 6.0]], dtype=np.float32)

            def __len__(self):
                return 1

        class _FakeResult:
            boxes = _FakeBoxes()
            keypoints = None

        class _FakeBallModel:
            names = {32: "sports_ball"}

            def predict(self, crop, classes, conf, device, verbose):
                return [_FakeResult()]

        frame = np.zeros((120, 220, 3), dtype=np.uint8)
        detections, summary = _run_ball_predictive_search(
            _FakeBallModel(),
            frame,
            {
                "center_xy": [80.0, 60.0],
                "velocity_xy": [8.0, 4.0],
                "last_seen_frame_idx": 10,
                "confidence": 0.73,
                "motion_mode": "pass_or_loose",
                "nearby_player_track_id": 9,
                "nearby_player_bbox_xyxy": [50.0, 20.0, 110.0, 120.0],
            },
            frame_idx=11,
            device="cpu",
            h_matrix=None,
        )
        self.assertTrue(summary["triggered"])
        self.assertEqual(summary["candidate_count"], 1)
        self.assertEqual(detections[0]["ball_detection_source"], "predictive_roi_v1")
        self.assertEqual(detections[0]["ball_motion_mode"], "pass_or_loose")
        self.assertGreater(detections[0]["bbox_xyxy"][0], 0.0)

    def test_ball_detection_needs_search_uses_confidence_threshold(self):
        self.assertTrue(_ball_detection_needs_search(None))
        self.assertTrue(_ball_detection_needs_search({"confidence": 0.19}))
        self.assertFalse(_ball_detection_needs_search({"confidence": 0.21}))

    def test_annotate_ball_state_marks_observed_and_predicted_short_gap(self):
        frames = [
            {
                "frame_idx": 0,
                "ball_detection": {
                    "confidence": 0.72,
                    "bbox_xyxy": [290.0, 210.0, 302.0, 222.0],
                    "bbox_xywh": [296.0, 216.0, 12.0, 12.0],
                    "center_xy": [296.0, 216.0],
                },
            },
            {
                "frame_idx": 1,
                "ball_detection": None,
            },
        ]
        summary = annotate_ball_state(frames)
        self.assertEqual(summary["kind"], "single_candidate_short_gap_v1")
        self.assertEqual(frames[0]["ball_state"]["state"], "observed")
        self.assertEqual(frames[1]["ball_state"]["state"], "predicted_short_gap")
        self.assertEqual(frames[1]["ball_state"]["missing_gap_frames"], 1)

    def test_annotate_ball_state_rejects_implausible_jump(self):
        frames = [
            {
                "frame_idx": 0,
                "ball_detection": {
                    "confidence": 0.72,
                    "bbox_xyxy": [290.0, 210.0, 302.0, 222.0],
                    "bbox_xywh": [296.0, 216.0, 12.0, 12.0],
                    "center_xy": [296.0, 216.0],
                },
            },
            {
                "frame_idx": 1,
                "ball_detection": {
                    "confidence": 0.31,
                    "bbox_xyxy": [790.0, 610.0, 802.0, 622.0],
                    "bbox_xywh": [796.0, 616.0, 12.0, 12.0],
                    "center_xy": [796.0, 616.0],
                },
            },
        ]
        annotate_ball_state(frames)
        self.assertEqual(frames[1]["ball_state"]["state"], "predicted_short_gap")


class BootstrapContextTest(unittest.TestCase):
    def test_annotate_bootstrap_contexts_reuses_context_within_segment(self):
        frames = [
            {
                "frame_idx": 0,
                "continuity_segment_id": 0,
                "_frame_bgr": np.zeros((8, 8, 3), dtype=np.uint8),
            },
            {
                "frame_idx": 1,
                "continuity_segment_id": 0,
                "_frame_bgr": np.zeros((8, 8, 3), dtype=np.uint8),
            },
            {
                "frame_idx": 2,
                "continuity_segment_id": 1,
                "_frame_bgr": np.zeros((8, 8, 3), dtype=np.uint8),
            },
        ]

        class _FakeResult:
            def __init__(self, frame_idx):
                self.frame_idx = frame_idx

            def to_payload(self):
                return {
                    "enabled": True,
                    "status": "ready",
                    "backend": "dinov3",
                    "model_name": "fake",
                    "frame_idx": self.frame_idx,
                    "foreground_ratio": 0.5,
                    "foreground_bbox_xyxy": [80, 90, 520, 430],
                    "mask_shape": [2, 2],
                    "mask_grid": [[1, 1], [0, 0]],
                    "image_width": 640,
                    "image_height": 480,
                }

        class _FakeBootstrapper:
            def __init__(self, model_name, device):
                self.model_name = model_name
                self.device = device

            def run_on_frame(self, frame, frame_idx=0):
                return _FakeResult(frame_idx)

        import tools.review.labeller.generate_layer1_annotations as module

        original = module.Dinov3Bootstrapper
        module.Dinov3Bootstrapper = _FakeBootstrapper
        try:
            summary = annotate_bootstrap_contexts(
                frames,
                bootstrap_foreground_backend="dinov3",
                bootstrap_foreground_model="fake",
                device="cuda:0",
            )
        finally:
            module.Dinov3Bootstrapper = original

        self.assertEqual(len(summary["contexts"]), 2)
        self.assertEqual(frames[0]["bootstrap_context"]["frame_idx"], 0)
        self.assertEqual(frames[1]["bootstrap_context"]["frame_idx"], 0)
        self.assertEqual(frames[2]["bootstrap_context"]["frame_idx"], 2)
        self.assertEqual(frames[0]["grounding_context"]["pipeline_order"], ["dino", "sam", "yolo"])
        self.assertEqual(frames[0]["grounding_context"]["trigger_reason"], "initial")
        self.assertEqual(frames[2]["grounding_context"]["trigger_reason"], "discontinuity")
        self.assertEqual(
            frames[0]["grounding_context"]["yolo_search_policy"],
            "mask_outside_play_region",
        )
        self.assertTrue(frames[0]["grounding_context"]["should_rerun_yolo"])
        self.assertEqual(frames[0]["grounding_context"]["yolo_search_region_mask_shape"], [2, 2])
        self.assertTrue(frames[0]["grounding_context"]["proposal_regions"])


    def test_annotate_scene_discovery_contracts_materializes_scene_prior(self):
        frames = [
            {
                "frame_idx": 0,
                "continuity_segment_id": 0,
                "bootstrap_context": {
                    "enabled": True,
                    "status": "ready",
                    "backend": "dinov3",
                    "model_name": "fake",
                    "mask_shape": [2, 2],
                    "mask_grid": [[1, 0], [1, 1]],
                },
                "grounding_context": {
                    "enabled": True,
                    "trigger_reason": "initial",
                    "proposal_regions": [
                        {"kind": "dino_play_region_component", "bbox_xyxy": [80, 90, 520, 430], "confidence": 0.5},
                    ],
                    "yolo_search_policy": "mask_outside_play_region",
                    "yolo_search_region_mask_shape": [2, 2],
                    "yolo_search_region_mask_grid": [[1, 0], [1, 1]],
                    "yolo_search_region_bbox_xyxy": [80, 90, 520, 430],
                    "should_rerun_yolo": True,
                },
                "detections": [],
            }
        ]
        summary = annotate_scene_discovery_contracts(frames)

        self.assertEqual(summary["scene_prior_frame_count"], 1)
        self.assertEqual(summary["scene_prior_ready_frame_count"], 1)
        self.assertEqual(summary["scene_prior_trigger_reasons"], ["initial"])
        scene_prior = frames[0]["scene_prior"]
        self.assertEqual(scene_prior["prior_status"], "ready")
        self.assertEqual(scene_prior["source_model"], "fake")
        self.assertEqual(scene_prior["region_mask_shape"], [2, 2])
        self.assertTrue(scene_prior["proposal_regions"])
        self.assertEqual(scene_prior["yolo_search_policy"], "mask_outside_play_region")

    def test_apply_grounding_mask_zeros_pixels_outside_search_region(self):
        frame = np.arange(6 * 6 * 3, dtype=np.uint8).reshape(6, 6, 3)
        masked, enabled = _apply_grounding_mask(
            frame,
            {
                "yolo_search_policy": "mask_outside_play_region",
                "yolo_search_region_mask_grid": [[0, 1, 1], [0, 1, 1], [0, 0, 0]],
            },
        )
        self.assertTrue(enabled)
        self.assertTrue(np.all(masked[:, 0:2] == 0))
        self.assertTrue(np.all(masked[4:, :] == 0))
        self.assertTrue(np.any(masked[:, 2:5] != 0))


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
        self.assertIn("on_court_score", detection)
        self.assertIn("on_court_candidate", detection)
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


class SamPlayerRecoveryTest(unittest.TestCase):
    def test_build_sam_recovery_rois_marks_unexplained_blobs_and_ambiguous_yolo(self):
        original_component_boxes = sam_refiner._connected_component_boxes
        try:
            sam_refiner._connected_component_boxes = lambda *args, **kwargs: [
                {"bbox_xyxy": [10, 10, 80, 140], "area_px": 7000, "area_ratio": 0.02},
                {"bbox_xyxy": [110, 20, 170, 150], "area_px": 9000, "area_ratio": 0.03},
            ]
            frame = {
                "bootstrap_context": {
                    "enabled": True,
                    "mask_grid": [[1, 1], [1, 1]],
                    "image_width": 320,
                    "image_height": 240,
                },
                "grounding_context": {
                    "proposal_regions": [
                        {"kind": "dino_play_region", "bbox_xyxy": [110, 20, 170, 150], "confidence": 0.72},
                    ],
                    "trigger_reason": "tracking_collapse",
                },
                "detections": [
                    {"track_id": 1, "bbox_xyxy": [12, 12, 78, 138], "active_player_reasons": {"merge_risk": 0.1}, "on_court_candidate": True},
                    {"track_id": 9, "bbox_xyxy": [200, 30, 310, 220], "active_player_reasons": {"merge_risk": 0.7}, "on_court_candidate": True},
                ],
        }
            rois = gla.build_sam_recovery_rois(frame)
        finally:
            sam_refiner._connected_component_boxes = original_component_boxes

        kinds = [roi["kind"] for roi in rois]
        self.assertIn("unexplained_dino_blob", kinds)
        self.assertIn("ambiguous_yolo_detection", kinds)
        self.assertIn("grounding_anchor_region", kinds)

    def test_annotate_sam_player_recovery_wires_refinement_and_recovered_detection(self):
        original_refiner = gla.Sam3RoiRefiner
        original_component_boxes = sam_refiner._connected_component_boxes

        class FakeRefiner:
            def __init__(self, model_name=None, text_prompt=None, device=None):
                self.model_name = model_name
                self.text_prompt = text_prompt
                self.device = device

            def refine(self, _frame_bgr, _rois):
                return SamRefineResult(
                    enabled=True,
                    status="ready",
                    backend="sam3_repo_v1",
                    model_name=self.model_name,
                    proposals=[
                        {
                            "kind": "ambiguous_yolo_detection",
                            "bbox_xyxy": [102.0, 54.0, 144.0, 176.0],
                            "bbox_xywh": [123.0, 115.0, 42.0, 122.0],
                            "mask_area_px": 5100,
                            "mask_area_ratio": 0.021,
                            "sam_score": 0.82,
                            "source_kind": "ambiguous_yolo_detection",
                            "source_track_id": 4,
                            "source_merge_risk": 0.66,
                        },
                        {
                            "kind": "unexplained_dino_blob",
                            "bbox_xyxy": [210.0, 40.0, 258.0, 172.0],
                            "bbox_xywh": [234.0, 106.0, 48.0, 132.0],
                            "mask_area_px": 6300,
                            "mask_area_ratio": 0.026,
                            "sam_score": 0.77,
                            "source_kind": "unexplained_dino_blob",
                            "source_iou": 0.04,
                            "source_roi_bbox_xyxy": [205.0, 35.0, 262.0, 178.0],
                        },
                        {
                            "kind": "grounding_anchor_region",
                            "bbox_xyxy": [42.0, 30.0, 90.0, 160.0],
                            "bbox_xywh": [66.0, 95.0, 48.0, 130.0],
                            "mask_area_px": 6200,
                            "mask_area_ratio": 0.025,
                            "sam_score": 0.75,
                            "source_kind": "grounding_anchor_region",
                            "source_iou": 0.01,
                            "source_trigger_reason": "tracking_collapse",
                            "source_roi_bbox_xyxy": [40.0, 28.0, 94.0, 164.0],
                        },
                    ],
                )

        try:
            gla.Sam3RoiRefiner = FakeRefiner
            sam_refiner._connected_component_boxes = lambda *args, **kwargs: [
                {"bbox_xyxy": [205, 35, 262, 178], "area_px": 6300, "area_ratio": 0.026},
            ]
            frames = [
                {
                    "frame_idx": 0,
                    "_frame_bgr": np.zeros((240, 320, 3), dtype=np.uint8),
                    "_h_matrix": None,
                    "bootstrap_context": {
                        "enabled": True,
                        "mask_grid": [[1, 1], [1, 1]],
                        "image_width": 320,
                        "image_height": 240,
                    },
                    "grounding_context": {
                        "proposal_regions": [
                            {"kind": "dino_play_region", "bbox_xyxy": [40, 28, 94, 164], "confidence": 0.81},
                        ],
                        "trigger_reason": "tracking_collapse",
                    },
                    "detections": [
                        {
                            "track_id": 4,
                            "bbox_xyxy": [100.0, 50.0, 150.0, 180.0],
                            "bbox_xywh": [125.0, 115.0, 50.0, 130.0],
                            "on_court_candidate": True,
                            "active_player_reasons": {"merge_risk": 0.66},
                        },
                    ],
                }
            ]
            summary = gla.annotate_sam_player_recovery(
                frames,
                player_recovery_backend="sam3",
                player_recovery_model="facebook/sam3.1",
                player_recovery_prompt="basketball player",
                device="cuda:0",
            )
        finally:
            gla.Sam3RoiRefiner = original_refiner
            sam_refiner._connected_component_boxes = original_component_boxes

        self.assertEqual(summary["status"], "ready")
        self.assertEqual(summary["refined_detection_count"], 1)
        self.assertEqual(summary["recovered_detection_count"], 2)
        detections = frames[0]["detections"]
        self.assertIn("sam3_refinement", detections[0])
        self.assertTrue(any(d.get("recovered_candidate_source") == "sam3_unexplained_dino_blob" for d in detections))
        self.assertTrue(any(d.get("recovered_candidate_source") == "sam3_grounding_anchor_region" for d in detections))
        recovered_anchor = next(d for d in detections if d.get("recovered_candidate_source") == "sam3_grounding_anchor_region")
        self.assertEqual(recovered_anchor["sam3_refinement"]["source_trigger_reason"], "tracking_collapse")
        self.assertEqual(detections[0]["sam3_refinement"]["model_name"], "facebook/sam3.1")
        self.assertEqual(detections[0]["sam3_refinement"]["text_prompt"], "basketball player")

        contract_summary = annotate_scene_discovery_contracts(frames)
        self.assertEqual(contract_summary["discovery_proposal_count"], 3)
        self.assertEqual(contract_summary["discovery_refinement_count"], 1)
        self.assertEqual(contract_summary["discovery_recovery_count"], 2)
        proposals = frames[0]["discovery_proposals"]
        self.assertEqual(proposals[0]["source_prompt"], "basketball player")
        self.assertEqual(proposals[0]["source_model"], "facebook/sam3.1")
        self.assertEqual(proposals[0]["proposal_role"], "refinement")
        self.assertTrue(any(p.get("proposal_role") == "recovery" for p in proposals))
        self.assertTrue(any((p.get("source_region") or {}).get("kind") == "grounding_anchor_region" for p in proposals))


class GroundingCollapseTest(unittest.TestCase):
    def test_mark_tracking_collapse_reground_flags_segment_after_streak(self):
        frames = []
        for frame_idx in range(22):
            frames.append(
                {
                    "frame_idx": frame_idx,
                    "continuity_segment_id": 0,
                    "discontinuity_label": "continuous",
                    "live_play_score": 0.82,
                    "live_play_reasons": {"on_court_active_candidate_count": 2},
                    "grounding_context": {
                        "enabled": True,
                        "proposal_regions": [{"kind": "dino_play_region", "bbox_xyxy": [40, 20, 220, 200]}],
                        "trigger_reason": "initial",
                        "should_rerun_yolo": False,
                    },
                }
            )
        summary = mark_tracking_collapse_reground(frames)
        self.assertEqual(summary["triggered_segment_ids"], [0])
        self.assertTrue(frames[-1]["grounding_context"]["collapse_triggered"])
        self.assertEqual(frames[-1]["grounding_context"]["collapse_trigger_frame_idx"], 19)
        self.assertTrue(frames[-1]["grounding_context"]["should_rerun_yolo"])
        self.assertEqual(frames[-1]["grounding_context"]["trigger_reason"], "initial")


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

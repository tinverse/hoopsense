import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pipelines.inference import (
    BALL_CLASS_ID,
    BallStateTracker,
    CalibrationResolver,
    FrameResultAdapter,
    GameDNAExtractor,
    InferenceConfig,
    MvpEventAdapter,
    RuntimeDiscontinuityDetector,
    RuntimePanDriftDetector,
)
from pipelines.ball_tracking import BallSearchPlan, BallSearchRoi
from pipelines.geometry import geometry_evidence_gate
from pipelines.mvp_stat_accumulator import MvpStatAccumulator


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

    def test_load_for_clip_falls_back_to_global_matrix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            fallback_path = tmp / "calibration.json"
            fallback_path.write_text(json.dumps({"h_matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 1]]}))
            resolver = CalibrationResolver(tmp / "missing.json", fallback_path).load_for_clip("demo_clip")
            h_matrix = resolver.homography_for_frame(1)
            self.assertTrue(np.array_equal(h_matrix, np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]])))


class FrameResultAdapterTest(unittest.TestCase):
    def test_extract_ball_state_updates_ball_tracker_and_writes_observed_event(self):
        boxes_xywh = [
            np.array([100.0, 120.0, 20.0, 20.0]),
            np.array([300.0, 220.0, 12.0, 12.0]),
        ]
        cls = [0, BALL_CLASS_ID]
        conf = [0.88, 0.67]
        writer = _RecordingWriter()
        tracker = BallStateTracker()
        ball_state, last_ball_2d, ball_3d = FrameResultAdapter.extract_ball_state(
            boxes_xywh=boxes_xywh,
            cls=cls,
            conf=conf,
            ball_tracker=tracker,
            h_matrix=np.eye(3),
            t_ms=250,
            writer=writer,
        )
        self.assertEqual(ball_state["state"], "observed")
        self.assertTrue(np.array_equal(last_ball_2d, np.array([300.0, 220.0])))
        self.assertIsNotNone(ball_3d)
        self.assertEqual(len(writer.rows), 1)
        self.assertEqual(writer.rows[0]["kind"], "ball")
        self.assertEqual(writer.rows[0]["t_ms"], 250)
        self.assertEqual(writer.rows[0]["state"], "observed")
        self.assertIsNone(writer.rows[0]["search_plan"])

    def test_extract_ball_state_predicts_short_gap_after_missing_frame(self):
        writer = _RecordingWriter()
        tracker = BallStateTracker()
        FrameResultAdapter.extract_ball_state(
            boxes_xywh=[np.array([300.0, 220.0, 12.0, 12.0])],
            cls=[BALL_CLASS_ID],
            conf=[0.67],
            ball_tracker=tracker,
            h_matrix=np.eye(3),
            t_ms=250,
            writer=writer,
        )
        ball_state, last_ball_2d, ball_3d = FrameResultAdapter.extract_ball_state(
            boxes_xywh=[],
            cls=[],
            conf=[],
            ball_tracker=tracker,
            h_matrix=np.eye(3),
            t_ms=283,
            writer=writer,
        )
        self.assertEqual(ball_state["state"], "predicted_short_gap")
        self.assertTrue(np.array_equal(last_ball_2d, np.array([300.0, 220.0], dtype=np.float32)))
        self.assertIsNotNone(ball_3d)
        self.assertEqual(writer.rows[-1]["state"], "predicted_short_gap")

    def test_extract_ball_state_uses_bootstrap_foreground_prior(self):
        writer = _RecordingWriter()
        tracker = BallStateTracker(min_observed_confidence=0.09)
        bootstrap_context = {
            "enabled": True,
            "mask_shape": [2, 2],
            "mask_grid": [[1, 0], [0, 0]],
            "image_width": 200,
            "image_height": 200,
        }
        ball_state, _, _ = FrameResultAdapter.extract_ball_state(
            boxes_xywh=[np.array([20.0, 20.0, 12.0, 12.0])],
            cls=[BALL_CLASS_ID],
            conf=[0.05],
            ball_tracker=tracker,
            h_matrix=np.eye(3),
            t_ms=100,
            writer=writer,
            bootstrap_context=bootstrap_context,
            bootstrap_segment_id=3,
        )
        self.assertEqual(ball_state["state"], "observed")
        self.assertEqual(writer.rows[-1]["bootstrap_segment_id"], 3)
        self.assertTrue(writer.rows[-1]["bootstrap_enabled"])
        self.assertGreater(ball_state["candidate_scores"][0]["score_parts"]["foreground_prior"], 0.0)
        self.assertGreater(writer.rows[-1]["play_region_prior"], 0.0)
        self.assertTrue(writer.rows[-1]["geometry_region_ok"])

    def test_extract_ball_state_writes_search_plan_provenance(self):
        writer = _RecordingWriter()
        tracker = BallStateTracker()
        search_plan = {
            "kind": "runtime_ball_search_plan_v1",
            "run_full_frame": False,
            "reason": "roi_cadence",
            "roi_count": 1,
        }
        FrameResultAdapter.extract_ball_state(
            boxes_xywh=[np.array([300.0, 220.0, 12.0, 12.0])],
            cls=[BALL_CLASS_ID],
            conf=[0.67],
            ball_tracker=tracker,
            h_matrix=np.eye(3),
            t_ms=250,
            writer=writer,
            ball_search_plan=search_plan,
        )
        self.assertEqual(writer.rows[-1]["search_plan"], search_plan)



class DetectorFirstRuntimeTest(unittest.TestCase):
    class _Tensor:
        def __init__(self, array):
            self.array = np.array(array, dtype=float)

        def cpu(self):
            return self

        def numpy(self):
            return self.array

    class _Boxes:
        def __init__(self, xyxy):
            self.xyxy = DetectorFirstRuntimeTest._Tensor(xyxy)

        def __len__(self):
            return len(self.xyxy.array)

    class _Keypoints:
        def __init__(self, xyn):
            self.xyn = DetectorFirstRuntimeTest._Tensor(xyn)

    class _Result:
        def __init__(self, xyxy, keypoints=None):
            self.boxes = DetectorFirstRuntimeTest._Boxes(xyxy)
            self.keypoints = None if keypoints is None else DetectorFirstRuntimeTest._Keypoints(keypoints)

    def test_match_pose_keypoints_projects_pose_onto_tracked_detector_boxes(self):
        tracked_result = [self._Result([[100.0, 40.0, 150.0, 180.0]])]
        pose_result = [self._Result([[102.0, 42.0, 148.0, 182.0]], keypoints=[[[0.5, 0.5]] * 17])]
        matched = FrameResultAdapter.match_pose_keypoints(tracked_result, pose_result)
        self.assertEqual(len(matched), 1)
        self.assertIsNotNone(matched[0])
        self.assertEqual(matched[0].shape, (17, 2))

    def test_update_tracks_records_runtime_tracklet_payload(self):
        extractor = GameDNAExtractor(
            InferenceConfig(video_path="dummy.mp4", reid_sample_interval_frames=2),
            models=type("Models", (), {"brain": None, "device": None})(),
            calibration=type("Calibration", (), {"homography_for_frame": lambda self, frame_idx: np.eye(3)})(),
        )
        extractor.frame_idx = 1
        frame = np.full((96, 96, 3), 64, dtype=np.uint8)

        player_map = extractor._update_tracks(
            boxes_xywh=np.array([[48.0, 48.0, 20.0, 40.0]], dtype=np.float32),
            conf=np.array([0.86], dtype=np.float32),
            tids=np.array([5]),
            classes=np.array([0]),
            keypoints=[None],
            h_matrix=np.eye(3),
            frame=frame,
        )

        tracklet = player_map[5]["tracklet"]
        self.assertEqual(tracklet["track_id"], 5)
        self.assertEqual(tracklet["observation_count"], 1)
        self.assertEqual(tracklet["reid"]["source"], "torso_color_histogram_v1")
        self.assertEqual(tracklet["reid"]["sample_count"], 1)

    def test_ball_search_player_detections_include_pixel_pose_keypoints(self):
        extractor = GameDNAExtractor(
            InferenceConfig(video_path="dummy.mp4"),
            models=type("Models", (), {"brain": None, "device": None})(),
            calibration=type("Calibration", (), {"homography_for_frame": lambda self, frame_idx: np.eye(3)})(),
        )
        keypoints = np.zeros((1, 17, 2), dtype=np.float32)
        keypoints[0, 9] = [0.5, 0.25]

        players = extractor._ball_search_player_detections(
            boxes_xywh=np.array([[320.0, 240.0, 80.0, 160.0]], dtype=np.float32),
            tids=np.array([12]),
            classes=np.array([0]),
            keypoints=keypoints,
            frame_shape=(720, 1280, 3),
        )

        self.assertEqual(players[0]["track_id"], 12)
        self.assertEqual(players[0]["keypoints_xy"][9], [640.0, 180.0])
        self.assertEqual(players[0]["bbox_xyxy"], [280.0, 160.0, 360.0, 320.0])
        self.assertEqual(players[0]["class_id"], 0)
        self.assertEqual(players[0]["keypoints_conf"][9], 1.0)

    def test_write_court_pose_event_emits_runtime_pose_state(self):
        extractor = GameDNAExtractor(
            InferenceConfig(video_path="dummy.mp4"),
            models=type("Models", (), {"brain": None, "device": None})(),
            calibration=type("Calibration", (), {"homography_for_frame": lambda self, frame_idx: np.eye(3)})(),
        )
        extractor.frame_idx = 7
        extractor.bootstrap_state.context = {
            "enabled": True,
            "status": "ready",
            "mask_shape": [1, 1],
            "mask_grid": [[1]],
            "backend": "grounding_dino",
            "model_name": "fake",
            "reason": "initial",
        }
        extractor.bootstrap_state.segment_id = 3
        writer = _RecordingWriter()

        extractor._write_court_pose_event(
            np.zeros((100, 100, 3), dtype=np.uint8),
            291,
            np.eye(3),
            [{"class_id": 0, "bbox_xyxy": [10.0, 20.0, 40.0, 80.0]}],
            writer,
        )

        self.assertEqual(writer.rows[0]["kind"], "court_pose")
        self.assertEqual(writer.rows[0]["state"], "calibrated")
        self.assertEqual(writer.rows[0]["bootstrap_segment_id"], 3)
        self.assertIn("grounding_scene_prior", writer.rows[0]["visible_features"])

    def test_run_ball_search_skips_tiny_roi_crops(self):
        class _BallModel:
            def __init__(self):
                self.called = False

            def predict(self, *args, **kwargs):
                self.called = True
                raise AssertionError("tiny crops should not be sent to the ball model")

        ball_model = _BallModel()
        extractor = GameDNAExtractor(
            InferenceConfig(video_path="dummy.mp4"),
            models=type("Models", (), {"brain": None, "device": "cpu", "ball_model": ball_model})(),
            calibration=type("Calibration", (), {"homography_for_frame": lambda self, frame_idx: np.eye(3)})(),
        )
        extractor.ball_search_scheduler = type(
            "Scheduler",
            (),
            {
                "plan": lambda self, *args, **kwargs: BallSearchPlan(
                    run_full_frame=False,
                    reason="roi_cadence",
                    rois=[BallSearchRoi([5, 5, 6, 40], "last_ball_state")],
                )
            },
        )()

        boxes, classes, confidences, payload, metadata = extractor._run_ball_search(
            np.zeros((64, 64, 3), dtype=np.uint8),
            [],
        )

        self.assertFalse(ball_model.called)
        self.assertEqual(boxes, [])
        self.assertEqual(classes, [])
        self.assertEqual(confidences, [])
        self.assertEqual(metadata, [])
        self.assertEqual(payload["roi_count"], 1)


class RuntimeBootstrapTest(unittest.TestCase):
    def test_discontinuity_detector_triggers_on_large_brightness_change(self):
        detector = RuntimeDiscontinuityDetector(threshold=10.0)
        dark = np.zeros((16, 16, 3), dtype=np.uint8)
        bright = np.full((16, 16, 3), 255, dtype=np.uint8)
        first = detector.update(dark)
        second = detector.update(bright)
        self.assertFalse(first["triggered"])
        self.assertTrue(second["triggered"])
        self.assertGreater(second["score"], 10.0)

    def test_game_dna_extractor_rebootstraps_on_discontinuity(self):
        import pipelines.inference as inference_module

        original_bootstrapper = inference_module.GroundingDinoBootstrapper

        class _FakeBootstrapper:
            def __init__(self, model_name, text_prompt=None, device=None):
                self.model_name = model_name
                self.text_prompt = text_prompt
                self.device = device

            def run_on_frame(self, frame, frame_idx=0):
                payload = {
                    "enabled": True,
                    "status": "ready",
                    "backend": "fake_grounding_dino",
                    "model_name": self.model_name,
                    "frame_idx": frame_idx,
                    "foreground_ratio": 0.25,
                }
                return type("BootstrapResult", (), {"to_payload": lambda self: payload})()

        inference_module.GroundingDinoBootstrapper = _FakeBootstrapper
        try:
            extractor = GameDNAExtractor(
                InferenceConfig(
                    video_path="dummy.mp4",
                    bootstrap_foreground_backend="grounding_dino",
                    bootstrap_discontinuity_threshold=10.0,
                ),
                models=type("Models", (), {"brain": None, "device": "cuda:0"})(),
                calibration=type("Calibration", (), {"homography_for_frame": lambda self, frame_idx: np.eye(3)})(),
            )
            writer = _RecordingWriter()
            extractor.frame_idx = 1
            extractor._maybe_refresh_bootstrap_context(np.zeros((16, 16, 3), dtype=np.uint8), 33, writer)
            extractor.frame_idx = 2
            extractor._maybe_refresh_bootstrap_context(np.full((16, 16, 3), 255, dtype=np.uint8), 66, writer)
        finally:
            inference_module.GroundingDinoBootstrapper = original_bootstrapper

        bootstrap_rows = [row for row in writer.rows if row["kind"] == "bootstrap_context"]
        self.assertEqual(len(bootstrap_rows), 2)
        self.assertEqual(bootstrap_rows[0]["reason"], "initial")
        self.assertEqual(bootstrap_rows[1]["reason"], "discontinuity")
        self.assertEqual(extractor.bootstrap_state.segment_id, 2)
        self.assertEqual(extractor.bootstrap_state.context["status"], "ready")

    def test_pan_drift_detector_triggers_on_coherent_track_shift(self):
        detector = RuntimePanDriftDetector(
            min_shared_tracks=2,
            median_displacement_px=20.0,
            alignment_spread_px=5.0,
        )
        first = detector.update(
            boxes_xywh=np.array([[100.0, 100.0, 10.0, 10.0], [200.0, 200.0, 10.0, 10.0]]),
            tids=np.array([1, 2]),
            classes=np.array([0, 0]),
        )
        second = detector.update(
            boxes_xywh=np.array([[140.0, 100.0, 10.0, 10.0], [240.0, 201.0, 10.0, 10.0]]),
            tids=np.array([1, 2]),
            classes=np.array([0, 0]),
        )
        self.assertFalse(first["triggered"])
        self.assertTrue(second["triggered"])
        self.assertGreaterEqual(second["median_displacement_px"], 20.0)

    def test_game_dna_extractor_rebootstraps_on_pan_drift(self):
        import pipelines.inference as inference_module

        original_bootstrapper = inference_module.GroundingDinoBootstrapper

        class _FakeBootstrapper:
            def __init__(self, model_name, text_prompt=None, device=None):
                self.model_name = model_name
                self.text_prompt = text_prompt
                self.device = device

            def run_on_frame(self, frame, frame_idx=0):
                payload = {
                    "enabled": True,
                    "status": "ready",
                    "backend": "fake_grounding_dino",
                    "model_name": self.model_name,
                    "frame_idx": frame_idx,
                    "foreground_ratio": 0.25,
                }
                return type("BootstrapResult", (), {"to_payload": lambda self: payload})()

        inference_module.GroundingDinoBootstrapper = _FakeBootstrapper
        try:
            extractor = GameDNAExtractor(
                InferenceConfig(
                    video_path="dummy.mp4",
                    bootstrap_foreground_backend="grounding_dino",
                    bootstrap_pan_recompute_cooldown_frames=0,
                ),
                models=type("Models", (), {"brain": None, "device": "cuda:0"})(),
                calibration=type("Calibration", (), {"homography_for_frame": lambda self, frame_idx: np.eye(3)})(),
            )
            writer = _RecordingWriter()
            extractor.frame_idx = 1
            extractor._maybe_refresh_bootstrap_context(np.zeros((16, 16, 3), dtype=np.uint8), 33, writer)
            extractor._maybe_refresh_bootstrap_context_after_tracking(
                np.zeros((16, 16, 3), dtype=np.uint8),
                33,
                writer,
                np.array([[100.0, 100.0, 10.0, 10.0], [200.0, 200.0, 10.0, 10.0], [300.0, 300.0, 10.0, 10.0], [400.0, 400.0, 10.0, 10.0]]),
                np.array([1, 2, 3, 4]),
                np.array([0, 0, 0, 0]),
            )
            extractor.frame_idx = 2
            extractor._maybe_refresh_bootstrap_context_after_tracking(
                np.zeros((16, 16, 3), dtype=np.uint8),
                66,
                writer,
                np.array([[170.0, 100.0, 10.0, 10.0], [270.0, 200.0, 10.0, 10.0], [370.0, 300.0, 10.0, 10.0], [470.0, 400.0, 10.0, 10.0]]),
                np.array([1, 2, 3, 4]),
                np.array([0, 0, 0, 0]),
            )
        finally:
            inference_module.GroundingDinoBootstrapper = original_bootstrapper

        bootstrap_rows = [row for row in writer.rows if row["kind"] == "bootstrap_context"]
        self.assertEqual(len(bootstrap_rows), 2)
        self.assertEqual(bootstrap_rows[1]["reason"], "pan_drift")
        self.assertEqual(extractor.bootstrap_state.segment_id, 2)

    def test_pan_drift_marks_context_stale_during_cooldown(self):
        import pipelines.inference as inference_module

        original_bootstrapper = inference_module.GroundingDinoBootstrapper

        class _FakeBootstrapper:
            def __init__(self, model_name, text_prompt=None, device=None):
                self.model_name = model_name
                self.text_prompt = text_prompt
                self.device = device

            def run_on_frame(self, frame, frame_idx=0):
                payload = {
                    "enabled": True,
                    "status": "ready",
                    "backend": "fake_grounding_dino",
                    "model_name": self.model_name,
                    "frame_idx": frame_idx,
                    "foreground_ratio": 0.25,
                }
                return type("BootstrapResult", (), {"to_payload": lambda self: payload})()

        inference_module.GroundingDinoBootstrapper = _FakeBootstrapper
        try:
            extractor = GameDNAExtractor(
                InferenceConfig(video_path="dummy.mp4", bootstrap_foreground_backend="grounding_dino"),
                models=type("Models", (), {"brain": None, "device": "cuda:0"})(),
                calibration=type("Calibration", (), {"homography_for_frame": lambda self, frame_idx: np.eye(3)})(),
            )
            writer = _RecordingWriter()
            extractor.frame_idx = 1
            extractor._maybe_refresh_bootstrap_context(np.zeros((16, 16, 3), dtype=np.uint8), 33, writer)
            extractor.frame_idx = 2
            extractor._maybe_refresh_bootstrap_context_after_tracking(
                np.zeros((16, 16, 3), dtype=np.uint8),
                66,
                writer,
                np.array([[100.0, 100.0, 10.0, 10.0], [200.0, 200.0, 10.0, 10.0], [300.0, 300.0, 10.0, 10.0], [400.0, 400.0, 10.0, 10.0]]),
                np.array([1, 2, 3, 4]),
                np.array([0, 0, 0, 0]),
            )
            extractor.frame_idx = 3
            extractor._maybe_refresh_bootstrap_context_after_tracking(
                np.zeros((16, 16, 3), dtype=np.uint8),
                99,
                writer,
                np.array([[170.0, 100.0, 10.0, 10.0], [270.0, 200.0, 10.0, 10.0], [370.0, 300.0, 10.0, 10.0], [470.0, 400.0, 10.0, 10.0]]),
                np.array([1, 2, 3, 4]),
                np.array([0, 0, 0, 0]),
            )
        finally:
            inference_module.GroundingDinoBootstrapper = original_bootstrapper

        bootstrap_rows = [row for row in writer.rows if row["kind"] == "bootstrap_context"]
        self.assertEqual(len(bootstrap_rows), 1)
        self.assertTrue(extractor.bootstrap_state.context["bootstrap_stale"])


class GeometryGateTest(unittest.TestCase):
    def test_geometry_evidence_gate_marks_points_inside_play_region(self):
        summary = {
            "mask_shape": [2, 2],
            "mask_grid": [[1, 0], [0, 0]],
            "image_width": 200,
            "image_height": 200,
        }
        gate = geometry_evidence_gate(summary, 10, 10, min_prior=0.5)
        self.assertTrue(gate["geometry_region_ok"])
        self.assertEqual(gate["play_region_prior"], 1.0)


class MvpEventAdapterTest(unittest.TestCase):
    def setUp(self):
        self.adapter = MvpEventAdapter()
        self.player_map = {
            7: {"team": 1},
            9: {"team": 1},
            14: {"team": 2},
        }

    def test_adapt_pass_emits_attributed_event(self):
        rows = self.adapter.adapt(
            {"kind": "pass", "from": 7, "to": 9, "team_id": 1, "t_ms": 500},
            self.player_map,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["kind"], "attributed_event")
        self.assertEqual(rows[0]["event_type"], "pass")
        self.assertEqual(rows[0]["actor_id"], 7)
        self.assertEqual(rows[0]["secondary_actor_id"], 9)
        self.assertEqual(rows[0]["stat_deltas"], {})
        self.assertEqual(rows[0]["rule_validation"], [])

    def test_adapt_steal_emits_turnover_and_steal(self):
        rows = self.adapter.adapt(
            {"kind": "steal", "player_id": 14, "from": 7, "team_id": 2, "t_ms": 800},
            self.player_map,
        )
        self.assertEqual([row["event_type"] for row in rows], ["turnover", "steal"])
        turnover, steal = rows
        self.assertEqual(turnover["stat_deltas"]["TOs"], 1)
        self.assertEqual(steal["stat_deltas"]["Steals"], 1)
        self.assertEqual(turnover["rule_validation"], [])
        self.assertEqual(steal["rule_validation"], [])


class ShotAttemptCandidateTest(unittest.TestCase):
    def _extractor(self):
        extractor = GameDNAExtractor(
            InferenceConfig(video_path="dummy.mp4"),
            models=type("Models", (), {"brain": None, "device": None})(),
            calibration=type("Calibration", (), {"homography_for_frame": lambda self, frame_idx: np.eye(3)})(),
        )
        extractor.tracks = {
            4: type("Track", (), {"team": 1, "court_x": 0.0, "court_y": 0.0})(),
        }
        extractor.last_ball_2d = np.array([320.0, 180.0])
        extractor.possession_engine.current_handler = 4
        return extractor

    def test_maybe_build_shot_attempt_candidate_emits_unresolved_payload(self):
        extractor = self._extractor()
        payload = extractor._maybe_build_shot_attempt_candidate(
            tid=4,
            learned_label="jump_shot",
            has_possession=True,
            t_ms=1200,
        )
        self.assertIsNotNone(payload)
        self.assertEqual(payload["event_type"], "shot_attempt")
        self.assertTrue(payload["candidate_only"])
        self.assertIn("shot_value_known", payload["rule_validation"])
        self.assertEqual(payload["stat_deltas"], {})

    def test_maybe_build_shot_attempt_candidate_debounces_repeated_labels(self):
        extractor = self._extractor()
        first = extractor._maybe_build_shot_attempt_candidate(
            tid=4,
            learned_label="jump_shot",
            has_possession=True,
            t_ms=1200,
        )
        second = extractor._maybe_build_shot_attempt_candidate(
            tid=4,
            learned_label="jump_shot",
            has_possession=True,
            t_ms=1800,
        )
        self.assertIsNotNone(first)
        self.assertIsNone(second)


class RunningStatUpdateTest(unittest.TestCase):
    def test_accumulator_emits_stat_update_for_counted_event(self):
        accumulator = MvpStatAccumulator()
        update = accumulator.apply_attributed_event(
            {
                "kind": "attributed_event",
                "event_type": "turnover",
                "actor_id": 7,
                "team_id": 1,
                "t_ms": 800,
                "stat_deltas": {"TOs": 1},
            }
        )
        self.assertEqual(update["kind"], "stat_update")
        self.assertEqual(update["applied_deltas"]["TOs"], 1)
        self.assertEqual(update["running_totals"]["TOs"], 1)

    def test_accumulator_snapshot_matches_running_totals(self):
        accumulator = MvpStatAccumulator()
        update = accumulator.apply_attributed_event(
            {
                "kind": "attributed_event",
                "event_type": "steal",
                "actor_id": 14,
                "team_id": 2,
                "t_ms": 900,
                "stat_deltas": {"Steals": 1},
            }
        )
        snapshot = accumulator.snapshot_for_player(
            update["player_id"],
            team_id=update["team_id"],
            t_ms=update["t_ms"],
        )
        self.assertEqual(snapshot["kind"], "stat_snapshot")
        self.assertEqual(snapshot["totals"], update["running_totals"])

    def test_terminal_game_snapshot_contains_accumulated_player_rows(self):
        accumulator = MvpStatAccumulator()
        accumulator.apply_attributed_event(
            {
                "kind": "attributed_event",
                "event_type": "turnover",
                "actor_id": 7,
                "team_id": 1,
                "t_ms": 800,
                "stat_deltas": {"TOs": 1},
            }
        )
        sheet = accumulator.terminal_game_snapshot(game_id="demo_clip", t_ms=800)
        self.assertEqual(sheet["kind"], "game_stat_sheet")
        self.assertEqual(sheet["players"][0]["player_id"], 7)
        self.assertEqual(sheet["players"][0]["totals"]["TOs"], 1)


if __name__ == "__main__":
    unittest.main()

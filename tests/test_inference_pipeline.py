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
)
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
        self.assertGreater(ball_state["candidate_scores"][0]["foreground_prior"], 0.0)


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

        original_bootstrapper = inference_module.Dinov3Bootstrapper

        class _FakeBootstrapper:
            def __init__(self, model_name, device):
                self.model_name = model_name
                self.device = device

            def run_on_frame(self, frame, frame_idx=0):
                payload = {
                    "enabled": True,
                    "status": "ready",
                    "backend": "fake_dino",
                    "model_name": self.model_name,
                    "frame_idx": frame_idx,
                    "foreground_ratio": 0.25,
                }
                return type("BootstrapResult", (), {"to_payload": lambda self: payload})()

        inference_module.Dinov3Bootstrapper = _FakeBootstrapper
        try:
            extractor = GameDNAExtractor(
                InferenceConfig(
                    video_path="dummy.mp4",
                    bootstrap_foreground_backend="dinov3",
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
            inference_module.Dinov3Bootstrapper = original_bootstrapper

        bootstrap_rows = [row for row in writer.rows if row["kind"] == "bootstrap_context"]
        self.assertEqual(len(bootstrap_rows), 2)
        self.assertEqual(bootstrap_rows[0]["reason"], "initial")
        self.assertEqual(bootstrap_rows[1]["reason"], "discontinuity")
        self.assertEqual(extractor.bootstrap_state.segment_id, 2)
        self.assertEqual(extractor.bootstrap_state.context["status"], "ready")


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

import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_perception_regressions import build_report, summarize_frame


class AuditPerceptionRegressionsTest(unittest.TestCase):
    def test_summarize_frame_reports_identity_ambiguity_metrics(self):
        summary = summarize_frame({
            "detections": [
                {
                    "track_id": 19,
                    "identity_track_id": 19,
                    "identity_option_track_id": 19,
                    "identity_is_ambiguous": True,
                    "identity_best_canonical_track_id": 3,
                    "motion_speed_px": 4.0,
                    "bbox_xyxy": [0, 0, 10, 10],
                },
                {
                    "track_id": 3,
                    "identity_track_id": 3,
                    "identity_option_track_id": 3,
                    "identity_is_ambiguous": False,
                    "identity_best_canonical_track_id": 3,
                    "motion_speed_px": 9.0,
                    "bbox_xyxy": [20, 0, 30, 10],
                },
            ]
        })
        self.assertEqual(summary["identity_option_track_ids"], ["19", "3"])
        self.assertEqual(summary["ambiguous_identity_detection_count"], 1)
        self.assertEqual(summary["best_canonical_delta_detection_count"], 1)

    def test_build_report_summarizes_case_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fixture_path = root / "fixture.json"
            artifact_dir = root / "artifacts"
            artifact_dir.mkdir()

            fixture = {
                "schema_version": "1.0.0",
                "cases": [
                    {
                        "case_id": "clip_a:0",
                        "clip_id": "clip_a",
                        "frame_idx": 0,
                        "categories": ["spectator_false_positive", "uniform_confusion"],
                        "scene_context": None,
                        "notes": ["P6 erroneously detects seated spectator."],
                    }
                ],
            }
            fixture_path.write_text(json.dumps(fixture))

            artifact = {
                "clip_id": "clip_a",
                "live_play_segments": [
                    {
                        "start_frame": 0,
                        "end_frame": 0,
                        "start_t_ms": 0,
                        "end_t_ms": 0,
                        "label": "dead_ball",
                        "mean_score": 0.2,
                        "peak_score": 0.2,
                        "frame_count": 1,
                        "reasons_summary": {"dominant_signal": "low_on_court_motion"},
                    }
                ],
                "video": {"width": 1280, "height": 720, "fps": 30.0},
                "frames": [
                    {
                        "frame_idx": 0,
                        "t_ms": 0,
                        "live_play_score": 0.2,
                        "live_play_label": "dead_ball",
                        "live_play_segment_label": "dead_ball",
                        "live_play_segment_duration_ms": 0,
                        "detections": [
                            {
                                "track_id": 6,
                                "identity_track_id": 6,
                                "identity_option_track_id": 6,
                                "identity_is_ambiguous": True,
                                "identity_best_canonical_track_id": 2,
                                "bbox_xyxy": [0, 0, 100, 200],
                                "uniform_bucket": "dark",
                                "motion_speed_px": 0.0,
                                "active_player_candidate": False,
                            },
                            {
                                "track_id": 3,
                                "bbox_xyxy": [10, 10, 120, 220],
                                "uniform_bucket": "light",
                                "motion_speed_px": 12.0,
                                "active_player_candidate": True,
                            },
                        ],
                    }
                ],
            }
            (artifact_dir / "clip_a.perception.json").write_text(json.dumps(artifact))

            report = build_report(fixture, artifact_dir, fixture_path)

            self.assertEqual(report["case_count"], 1)
            case = report["cases"][0]
            self.assertTrue(case["artifact_found"])
            self.assertTrue(case["frame_found"])
            self.assertEqual(case["frame_metrics"]["detection_count"], 2)
            self.assertEqual(case["frame_metrics"]["active_candidate_count"], 1)
            self.assertEqual(case["frame_metrics"]["uniform_bucket_counts"]["dark"], 1)
            self.assertEqual(case["frame_metrics"]["uniform_bucket_counts"]["light"], 1)
            self.assertIn("low_motion_detection_count", case["category_signals"])
            self.assertIn("active_uniform_bucket_counts", case["category_signals"])
            self.assertEqual(case["frame_metrics"]["ambiguous_identity_detection_count"], 1)
            self.assertEqual(case["frame_metrics"]["best_canonical_delta_detection_count"], 1)
            self.assertEqual(case["frame_metrics"]["live_play_label"], "dead_ball")

            summary = report["category_summary"]
            self.assertEqual(summary["spectator_false_positive"]["case_count"], 1)
            self.assertEqual(summary["spectator_false_positive"]["detection_count_total"], 2)
            self.assertEqual(summary["uniform_confusion"]["active_candidate_count_total"], 1)

    def test_build_report_marks_missing_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fixture_path = root / "fixture.json"
            artifact_dir = root / "artifacts"
            artifact_dir.mkdir()
            fixture = {
                "schema_version": "1.0.0",
                "cases": [
                    {
                        "case_id": "clip_missing:5",
                        "clip_id": "clip_missing",
                        "frame_idx": 5,
                        "categories": ["id_switch"],
                        "scene_context": None,
                        "notes": [],
                    }
                ],
            }
            fixture_path.write_text(json.dumps(fixture))

            report = build_report(fixture, artifact_dir, fixture_path)

            case = report["cases"][0]
            self.assertFalse(case["artifact_found"])
            self.assertFalse(case["frame_found"])
            self.assertNotIn("frame_metrics", case)


if __name__ == "__main__":
    unittest.main()

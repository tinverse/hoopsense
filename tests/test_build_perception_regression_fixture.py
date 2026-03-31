import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_perception_regression_fixture import build_fixture, load_jsonl


class PerceptionRegressionFixtureTest(unittest.TestCase):
    def test_build_fixture_groups_and_categorizes_feedback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "perception_feedback.jsonl"
            src.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "clip_id": "clip_a",
                                "frame_idx": 0,
                                "t_ms": 0,
                                "issue_type": "false_positive",
                                "note": "P6 erroneously detects seated spectator.",
                                "timestamp": "2026-03-31T01:16:55.020Z",
                            }
                        ),
                        json.dumps(
                            {
                                "clip_id": "clip_a",
                                "frame_idx": 0,
                                "t_ms": 0,
                                "issue_type": "general_note",
                                "note": "Missed 3 players clustered together by P3.",
                                "timestamp": "2026-03-31T01:19:54.752Z",
                            }
                        ),
                        json.dumps(
                            {
                                "clip_id": "clip_b",
                                "frame_idx": 28,
                                "t_ms": 1167,
                                "issue_type": "general_note",
                                "note": "Player numbers keep changing. P3 and P23 are the same.",
                                "timestamp": "2026-03-31T01:36:36.891Z",
                            }
                        ),
                        json.dumps(
                            {
                                "clip_id": "clip_c",
                                "frame_idx": 120,
                                "t_ms": 5005,
                                "issue_type": "general_note",
                                "note": "All the players from one team are getting subbed out.",
                                "timestamp": "2026-03-31T03:04:01.418Z",
                            }
                        ),
                    ]
                )
                + "\n"
            )

            entries = load_jsonl(src)
            fixture = build_fixture(entries, src)

            self.assertEqual(fixture["case_count"], 3)
            by_case_id = {case["case_id"]: case for case in fixture["cases"]}

            self.assertIn("spectator_false_positive", by_case_id["clip_a:0"]["categories"])
            self.assertIn("cluster_miss", by_case_id["clip_a:0"]["categories"])
            self.assertEqual(by_case_id["clip_a:0"]["source_count"], 2)

            self.assertIn("id_switch", by_case_id["clip_b:28"]["categories"])
            self.assertIsNone(by_case_id["clip_b:28"]["scene_context"])

            self.assertIn("dead_ball_context", by_case_id["clip_c:120"]["categories"])
            self.assertEqual(by_case_id["clip_c:120"]["scene_context"], "dead_ball")


if __name__ == "__main__":
    unittest.main()

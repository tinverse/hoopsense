import tempfile
import unittest
from pathlib import Path

from pipelines.mvp_rules import load_mvp_event_rules


class MvpEventRulesTest(unittest.TestCase):
    def test_load_mvp_event_rules_loads_default_contract(self):
        spec = load_mvp_event_rules()
        self.assertEqual(spec["kind"], "mvp_event_rules")
        self.assertIn("shot_attempt", spec["events"])
        self.assertEqual(
            spec["events"]["made_shot"]["counting"]["stat_deltas"]["three_point"]["PTS"],
            3,
        )

    def test_load_mvp_event_rules_rejects_missing_events(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad_rules.yaml"
            path.write_text(
                "kind: mvp_event_rules\n"
                "assumptions:\n"
                "  deterministic_counting: true\n"
                "events:\n"
                "  pass:\n"
                "    prerequisites: []\n"
                "    actors: {primary: passer}\n"
                "    temporal: {}\n"
                "    counting: {stat_deltas: {}}\n"
            )
            with self.assertRaises(ValueError):
                load_mvp_event_rules(path)


if __name__ == "__main__":
    unittest.main()

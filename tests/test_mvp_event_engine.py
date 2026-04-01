import unittest

from pipelines.mvp_event_engine import MvpEventRuleEngine


class MvpEventRuleEngineTest(unittest.TestCase):
    def setUp(self):
        self.engine = MvpEventRuleEngine()

    def test_stat_deltas_for_two_point_make(self):
        event = {
            "event_type": "made_shot",
            "actor_id": 7,
            "live_play": True,
            "shot_value": "two_point",
            "preceding_event_type": "shot_attempt",
            "evidence": {"shot_result": "made"},
        }
        deltas = self.engine.stat_deltas_for_event(event)
        self.assertEqual(deltas["2P FGM"], 1)
        self.assertEqual(deltas["PTS"], 2)

    def test_validate_event_reports_missing_constraints(self):
        event = {
            "event_type": "assist_candidate",
            "actor_id": 3,
            "secondary_actor_id": None,
            "live_play": True,
            "preceding_event_type": "pass",
            "terminal_event_type": "made_shot",
            "evidence": {},
        }
        unmet = self.engine.validate_event(event)
        self.assertIn("receiver_identity_known", unmet)
        self.assertIn("secondary_actor_missing", unmet)
        self.assertIn("pass_then_made_shot_sequence", unmet)

    def test_stat_deltas_for_event_rejects_invalid_payload(self):
        event = {
            "event_type": "turnover",
            "actor_id": 5,
            "live_play": False,
            "evidence": {},
        }
        with self.assertRaises(ValueError):
            self.engine.stat_deltas_for_event(event)


if __name__ == "__main__":
    unittest.main()

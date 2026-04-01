import unittest

from pipelines.mvp_stat_accumulator import MvpStatAccumulator


class MvpStatAccumulatorTest(unittest.TestCase):
    def test_apply_attributed_event_emits_running_totals(self):
        accumulator = MvpStatAccumulator()
        first = accumulator.apply_attributed_event(
            {
                "kind": "attributed_event",
                "event_type": "steal",
                "actor_id": 14,
                "team_id": 2,
                "t_ms": 800,
                "stat_deltas": {"Steals": 1},
            }
        )
        second = accumulator.apply_attributed_event(
            {
                "kind": "attributed_event",
                "event_type": "steal",
                "actor_id": 14,
                "team_id": 2,
                "t_ms": 1200,
                "stat_deltas": {"Steals": 1},
            }
        )
        self.assertEqual(first["kind"], "stat_update")
        self.assertEqual(first["running_totals"]["Steals"], 1)
        self.assertEqual(second["running_totals"]["Steals"], 2)

    def test_apply_attributed_event_ignores_uncounted_payloads(self):
        accumulator = MvpStatAccumulator()
        update = accumulator.apply_attributed_event(
            {
                "kind": "attributed_event",
                "event_type": "shot_attempt",
                "actor_id": 7,
                "team_id": 1,
                "t_ms": 1000,
                "stat_deltas": {},
            }
        )
        self.assertIsNone(update)

    def test_snapshot_for_player_returns_current_totals(self):
        accumulator = MvpStatAccumulator()
        accumulator.apply_attributed_event(
            {
                "kind": "attributed_event",
                "event_type": "turnover",
                "actor_id": 7,
                "team_id": 1,
                "t_ms": 1000,
                "stat_deltas": {"TOs": 1},
            }
        )
        snapshot = accumulator.snapshot_for_player(7, team_id=1, t_ms=1000)
        self.assertEqual(snapshot["kind"], "stat_snapshot")
        self.assertEqual(snapshot["player_id"], 7)
        self.assertEqual(snapshot["totals"]["TOs"], 1)


if __name__ == "__main__":
    unittest.main()

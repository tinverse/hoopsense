import unittest
import numpy as np
from pipelines.behavior_engine import BehaviorStateMachine, EntityState, PossessionEngine

class TestBehaviorEngine(unittest.TestCase):
    def setUp(self):
        self.player_fsm = BehaviorStateMachine(is_ref=False)
        self.ref_fsm = BehaviorStateMachine(is_ref=True)

    def test_idle_state(self):
        # Static history should stay IDLE and pick the idle_bystander label.
        mock_kpts = [np.zeros((17, 2)) for _ in range(30)]
        self.player_fsm.update(mock_kpts)
        self.assertEqual(self.player_fsm.state, EntityState.IDLE)
        self.assertEqual(self.player_fsm.get_label(), "idle_bystander")

    def test_jump_shot_recognition(self):
        # Simulate a jump shot: rising hips and wrists above head with possession context.
        mock_kpts = []
        for i in range(20):
            k = np.zeros((17, 2))
            # Hips (11, 12) rising: Y decreases from 0.5 to 0.3
            y_val = 0.5 - (i * 0.01)
            k[11] = [0.4, y_val]
            k[12] = [0.6, y_val]
            # Wrists (9, 10) above head (0): Head at 0.4, Wrists at 0.2
            k[0] = [0.5, 0.4]
            k[9] = [0.4, 0.2]
            k[10] = [0.6, 0.2]
            mock_kpts.append(k)
        
        self.player_fsm.update(mock_kpts, context={"has_possession": True})
        self.assertEqual(self.player_fsm.state, EntityState.SHOOTING)
        self.assertEqual(self.player_fsm.get_label(), "jump_shot")

    def test_jump_shot_without_possession_falls_back(self):
        mock_kpts = []
        for i in range(20):
            k = np.zeros((17, 2))
            y_val = 0.5 - (i * 0.01)
            k[11] = [0.4, y_val]
            k[12] = [0.6, y_val]
            k[0] = [0.5, 0.4]
            k[9] = [0.4, 0.2]
            k[10] = [0.6, 0.2]
            mock_kpts.append(k)

        self.player_fsm.update(mock_kpts, context={"has_possession": False})
        self.assertEqual(self.player_fsm.state, EntityState.SHOOTING)
        self.assertNotEqual(self.player_fsm.get_label(), "jump_shot")

    def test_ref_without_signal_rules_stays_idle(self):
        # The current DSL has no signal-type referee rules, so refs stay idle.
        mock_kpts = []
        for _ in range(15):
            k = np.zeros((17, 2))
            k[5] = [0.4, 0.4]
            k[6] = [0.6, 0.4]
            k[9] = [0.3, 0.2]
            k[10] = [0.7, 0.2]
            mock_kpts.append(k)
            
        self.ref_fsm.update(mock_kpts)
        self.assertEqual(self.ref_fsm.state, EntityState.IDLE)

    def test_possession_engine_catch_pass_and_steal(self):
        engine = PossessionEngine()
        ball = np.array([0.0, 0.0, 0.0])
        team_a1 = {"pos_3d": np.array([0.0, 0.0, 0.0]), "team": 1}
        team_a2 = {"pos_3d": np.array([10.0, 0.0, 0.0]), "team": 1}
        team_b1 = {"pos_3d": np.array([10.0, 0.0, 0.0]), "team": 2}

        events = engine.update({7: team_a1}, ball, 100)
        self.assertEqual(events, [{"kind": "catch", "actor": 7, "t_ms": 100}])
        self.assertEqual(engine.current_handler, 7)

        events = engine.update({7: {"pos_3d": np.array([200.0, 0.0, 0.0]), "team": 1}, 9: team_a2}, ball, 200)
        self.assertEqual(events, [
            {"kind": "pass", "from": 7, "to": 9, "t_ms": 200},
            {"kind": "catch", "actor": 9, "t_ms": 200},
        ])
        self.assertEqual(engine.current_handler, 9)

        events = engine.update({9: {"pos_3d": np.array([200.0, 0.0, 0.0]), "team": 1}, 14: team_b1}, ball, 300)
        self.assertEqual(events, [
            {"kind": "steal", "actor": 14, "t_ms": 300},
            {"kind": "catch", "actor": 14, "t_ms": 300},
        ])
        self.assertEqual(engine.current_handler, 14)

    def test_possession_engine_releases_when_ball_is_loose(self):
        engine = PossessionEngine()
        ball = np.array([0.0, 0.0, 0.0])
        engine.update({7: {"pos_3d": np.array([0.0, 0.0, 0.0]), "team": 1}}, ball, 100)
        self.assertEqual(engine.current_handler, 7)

        events = engine.update({7: {"pos_3d": np.array([500.0, 0.0, 0.0]), "team": 1}}, ball, 200)
        self.assertEqual(events, [])
        self.assertIsNone(engine.current_handler)

if __name__ == "__main__":
    unittest.main()

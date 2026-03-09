import unittest
import numpy as np
from pipelines.behavior_engine import BehaviorStateMachine, EntityState

class TestBehaviorEngine(unittest.TestCase):
    def setUp(self):
        self.player_fsm = BehaviorStateMachine(is_ref=False)
        self.ref_fsm = BehaviorStateMachine(is_ref=True)

    def test_idle_state(self):
        # Empty or static history should be IDLE
        mock_kpts = [np.zeros((17, 2)) for _ in range(30)]
        self.player_fsm.update(mock_kpts)
        self.assertEqual(self.player_fsm.state, EntityState.IDLE)

    def test_jump_shot_recognition(self):
        # Simulate a jump shot: rising hips and wrists above head
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
        
        self.player_fsm.update(mock_kpts)
        self.assertEqual(self.player_fsm.state, EntityState.SHOOTING)

    def test_ref_signal_recognition(self):
        # Simulate Ref with both arms up
        mock_kpts = []
        for _ in range(15):
            k = np.zeros((17, 2))
            # Shoulders (5, 6) at Y=0.4
            k[5] = [0.4, 0.4]
            k[6] = [0.6, 0.4]
            # Wrists (9, 10) at Y=0.2 (Above shoulders)
            k[9] = [0.3, 0.2]
            k[10] = [0.7, 0.2]
            mock_kpts.append(k)
            
        self.ref_fsm.update(mock_kpts)
        self.assertEqual(self.ref_fsm.state, EntityState.OFFICIAL_SIGNALING)

if __name__ == "__main__":
    unittest.main()

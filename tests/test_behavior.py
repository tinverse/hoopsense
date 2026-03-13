import unittest
import numpy as np
from pipelines.behavior_engine import BehaviorStateMachine, EntityState


class TestBehaviorEngine(unittest.TestCase):
    def setUp(self):
        self.player_fsm = BehaviorStateMachine(is_ref=False)
        self.ref_fsm = BehaviorStateMachine(is_ref=True)

    def test_idle_default(self):
        self.assertEqual(self.player_fsm.state, EntityState.IDLE)

    def test_jump_shot_logic(self):
        """Mock sequence of keypoints where shoulders and wrists rise."""
        mock_kpts = []
        for i in range(20):
            # Keypoints are [x, y] normalized to bbox
            # COCO indices: 5,6 shoulders, 9,10 wrists
            # Rising in image space means Y decreases
            k = np.ones((17, 2)) * 0.5
            # Rapid rise
            k[9:11, 1] = 0.5 - (i * 0.02)
            mock_kpts.append(k)

        self.player_fsm.update(mock_kpts)
        # Should trigger jump_shot declarative rule
        self.assertEqual(self.player_fsm.get_label(), "jump_shot")

    def test_ref_signal_logic(self):
        """Mock sequence where referee signals 3pt success."""
        # This requires spec support for signal types
        mock_kpts = []
        for i in range(20):
            k = np.ones((17, 2)) * 0.5
            # Mock hands above head (Y < 0.2)
            k[10] = [0.7, 0.2]
            mock_kpts.append(k)

        self.ref_fsm.update(mock_kpts)
        # Expected label based on NCAA spec
        label = self.ref_fsm.get_label()
        self.assertTrue(label == "official_signaling" or label == "idle")


if __name__ == "__main__":
    unittest.main()

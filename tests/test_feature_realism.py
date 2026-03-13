import unittest
import numpy as np
from tools.synthetic.generate_data import MoveLibrary, compute_features_v2


class TestFeatureRealism(unittest.TestCase):
    def test_jump_shot_physics(self):
        """Verify that jump shot features show expected Z-velocity."""
        data_3d = [MoveLibrary.jump_shot(t) for t in np.linspace(0, 1, 30)]
        skel_3d = np.array([d[0] for d in data_3d])
        ball_3d = np.array([d[1] for d in data_3d])
        # Normalized 2D mock (assuming identity projection for test)
        skel_2d_norm = np.ones((30, 17, 2)) * 0.5
        feats = compute_features_v2(skel_2d_norm, skel_3d, ball_3d)
        feats_np = np.array(feats)
        # Check ball proximity (D=68,69 in features_v2)
        # Wrist index 10, check distance
        self.assertLess(feats_np[15, 69], 1.0)  # Close to ball at apex

    def test_court_context_normalization(self):
        """Verify court coordinates are correctly scaled."""
        skel_3d = np.zeros((30, 17, 3))
        # Center of court
        skel_3d[:, 11:13] = [1432.5, 762.0, 100.0]
        ball_3d = np.zeros((30, 3))
        skel_2d_norm = np.zeros((30, 17, 2))
        feats = compute_features_v2(skel_2d_norm, skel_3d, ball_3d)
        feats_np = np.array(feats)
        # Last two dims are court context
        self.assertAlmostEqual(feats_np[0, -2], 1.4325, places=2)
        self.assertAlmostEqual(feats_np[0, -1], 0.762, places=2)


if __name__ == "__main__":
    unittest.main()

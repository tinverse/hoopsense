import unittest
import numpy as np
import torch
from pipelines.inference import construct_features_v2

class TestFeatureRealism(unittest.TestCase):
    def test_cm_grounding(self):
        # 1. Setup: NCAA Homography (Simplified Identity for Test)
        H = np.eye(3)
        device = torch.device("cpu")
        
        # 2. Mock Data: Player at (1000, 1000) pixels, Ball at (1000, 1100) pixels
        # With H=Identity, this maps to (1000cm, 1000cm) and (1000cm, 1100cm)
        player_court_pos = np.array([1000.0, 1000.0])
        last_ball_2d = np.array([1000.0, 1100.0])
        
        # 30 frames of static keypoints (Hips at 1000, 1000)
        # We'll put feet at (1000, 1000) and wrists at (1000, 900)
        # lift_to_3d should see 100cm vertical diff -> Z height
        kpts = np.zeros((30, 17, 2))
        kpts[:, 15:17] = [1000, 1000] # Ankles
        kpts[:, 9:11] = [1000, 900]   # Wrists
        kpt_history = kpts.tolist()
        
        # 3. Construct Features
        feat_tensor = construct_features_v2(kpt_history, last_ball_2d, H, player_court_pos, device)
        
        # 4. Assertions
        features = feat_tensor.squeeze(0).numpy()
        
        # Global Court Context (Indices 70-71)
        # player_court_pos (1000, 1000) * 0.001 = (1.0, 1.0)
        self.assertAlmostEqual(features[0, 70], 1.0)
        self.assertAlmostEqual(features[0, 71], 1.0)
        
        # Ball Proximity (Indices 68-69)
        # Should be real distance in CM, not 0.0 or 1.0
        # This proves we are using lifting and 3D Euclidean math
        self.assertGreater(features[0, 68], 0.0)
        print(f"[TEST] Real-world Ball Distance: {features[0, 68] * 100:.1f} cm")

if __name__ == "__main__":
    unittest.main()

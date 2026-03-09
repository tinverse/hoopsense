import unittest
import numpy as np
from tools.synthetic.generate_data import MoveLibrary, get_look_at_matrix, project_to_2d

class TestSyntheticGenerator(unittest.TestCase):
    def test_look_at_matrix(self):
        cam_pos = np.array([0.0, -500.0, 200.0])
        target_pos = np.array([0.0, 0.0, 100.0])
        R = get_look_at_matrix(cam_pos, target_pos)
        
        # Matrix should be 3x3
        self.assertEqual(R.shape, (3, 3))
        # Matrix should be orthogonal (R * R^T = I)
        np.testing.assert_array_almost_equal(R @ R.T, np.eye(3))

    def test_projection_logic(self):
        K = np.array([[1000, 0, 960], [0, 1000, 540], [0, 0, 1]])
        R = np.eye(3)
        t = np.array([0, 0, -500]) # Camera 5m back
        
        # Point at origin (0,0,0)
        skel_3d = np.zeros((1, 17, 3))
        skel_2d = project_to_2d(skel_3d, K, R, t, noise_std=0.0)
        
        # (0,0,0) in world should map to center (960, 540) if looking straight
        self.assertAlmostEqual(skel_2d[0, 0, 0], 960.0)
        self.assertAlmostEqual(skel_2d[0, 0, 1], 540.0)

if __name__ == "__main__":
    unittest.main()

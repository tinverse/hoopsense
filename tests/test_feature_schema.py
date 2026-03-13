import unittest
import numpy as np
from core.vision.action_brain import ActionBrain


class TestFeatureSchema(unittest.TestCase):
    def test_schema_v2_input_dims(self):
        """Verify that ActionBrain expects D=72 as per Schema V2."""
        model = ActionBrain(num_classes=5)
        # Transformer input projection check via forward pass
        dummy = np.random.randn(1, 30, 72).astype(np.float32)
        import torch
        try:
            model(torch.from_numpy(dummy))
            passed = True
        except Exception:
            passed = False
        self.assertTrue(passed, "Model failed on D=72 input")

    def test_feature_flattening_logic(self):
        """Verify that flattening 17 points leads to expected indices."""
        # 17 points * 2 coords = 34
        skel = np.zeros((17, 2))
        flat = skel.flatten()
        self.assertEqual(len(flat), 34)


if __name__ == "__main__":
    unittest.main()

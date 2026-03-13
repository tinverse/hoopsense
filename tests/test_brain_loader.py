import unittest
import torch
from core.vision.action_brain import ActionBrain


class TestBrainLoader(unittest.TestCase):
    def test_model_initialization(self):
        """Verify that ActionBrain can be initialized."""
        model = ActionBrain(num_classes=5)
        self.assertIsNotNone(model)

    def test_forward_pass_dummy(self):
        """Verify that a dummy forward pass works."""
        model = ActionBrain(num_classes=5)
        # Input shape: (Batch, Frames, Dims) -> (1, 30, 72)
        dummy_input = torch.randn(1, 30, 72)
        output = model(dummy_input)
        self.assertEqual(output.shape, (1, 5))

    def test_parameter_count(self):
        """Verify model has expected parameter scale."""
        model = ActionBrain(num_classes=5)
        total_params = sum(p.numel() for p in model.parameters())
        # Temporal Transformer should be around ~100k-500k params
        self.assertGreater(total_params, 50000)


if __name__ == "__main__":
    unittest.main()

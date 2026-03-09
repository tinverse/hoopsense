import torch
import os
import unittest
from core.vision.action_brain import ActionBrain, save_model

class TestBrainLoader(unittest.TestCase):
    def test_lifecycle(self):
        # 1. Initialize with Schema V2 dimensions
        model = ActionBrain(input_dim=72, num_classes=5)
        
        # 2. Verify Forward Pass
        dummy_input = torch.randn(1, 30, 72)
        output = model(dummy_input)
        self.assertEqual(output.shape, (1, 5))
        
        # 3. Verify Save (Tests the 'os' import fix)
        test_path = "data/test_models/test_brain.pt"
        if os.path.exists(test_path): os.remove(test_path)
        
        save_model(model, path=test_path)
        self.assertTrue(os.path.exists(test_path))
        
        # 4. Verify Load
        new_model = ActionBrain(input_dim=72, num_classes=5)
        new_model.load_state_dict(torch.load(test_path))
        new_model.eval()
        
        new_output = new_model(dummy_input)
        self.assertTrue(torch.allclose(output, new_output))
        print("[SUCCESS] Action Brain lifecycle verified.")

if __name__ == "__main__":
    unittest.main()

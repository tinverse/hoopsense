import torch
import unittest
from core.vision.action_brain import ActionBrain

class TestModelHardware(unittest.TestCase):
    def test_device_movement(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[TEST] Testing model on: {device}")
        
        model = ActionBrain(num_classes=5).to(device)
        
        # Verify all parameters are on the correct device
        for param in model.parameters():
            self.assertEqual(param.device.type, device.type)
            
        # Run a dummy inference to verify forward pass on device
        dummy_input = torch.randn(1, 30, 72).to(device)
        output = model(dummy_input)
        self.assertEqual(output.device.type, device.type)
        self.assertEqual(output.shape, (1, 5))
        print("[SUCCESS] Model hardware awareness verified.")

if __name__ == "__main__":
    unittest.main()

import json
import os
import unittest

class TestFeatureSchema(unittest.TestCase):
    def test_v2_schema_completeness(self):
        data_path = "data/training/synthetic_dataset_v2.jsonl"
        self.assertTrue(os.path.exists(data_path), "v2 dataset missing")
        
        with open(data_path, 'r') as f:
            for line in f:
                sample = json.loads(line)
                self.assertEqual(sample["schema_version"], "2.0.0")
                
                features = sample["features_v2"]
                # Must be 30 frames
                self.assertEqual(len(features), 30)
                
                for frame in features:
                    # Each frame must have exactly 72 dimensions
                    self.assertEqual(len(frame), 72, f"Feature dimension mismatch in {sample['label']}")
                    
                    # Basic normalization checks
                    # Pose (0-33) should be ~0.0 to 1.0
                    for val in frame[:34]:
                        self.assertTrue(-0.5 <= val <= 1.5, f"Pose normalization failure: {val}")

if __name__ == "__main__":
    unittest.main()

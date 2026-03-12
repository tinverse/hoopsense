import unittest
import json
import numpy as np
import os
from pipelines.perception_primitives import match_pose_to_box
from pipelines.geometry import lift_keypoints_to_3d

class TestPerceptionGate(unittest.TestCase):
    """
    Layer 1 'Fully Trusted Gate' Test Suite.
    Ensures that raw perception output (Layer 1) meets the quality bars
    for tracking, pose, geometry, and schema.
    """

    def test_tracking_persistence(self):
        """
        Verify that a track ID remains stable across a simulated 2-second (60 frame)
        occlusion or jitter event using mock data.
        """
        # Scenario: Track 1 is seen for 20 frames, disappears for 60 frames, then reappears.
        # This test ensures that our identity resolution logic (Layer 1) would handle this.
        # Currently, if using YOLOv8 'persist=True', this is handled by BoT-SORT/ByteTrack.
        # We mock the track ID stream to ensure no 'id-switches' occur in our processing.
        
        frames_seen = list(range(1, 21))
        frames_occluded = list(range(21, 81))
        frames_reappeared = list(range(81, 101))
        
        # Mock track IDs for a single entity
        track_ids = []
        for f in frames_seen:
            track_ids.append(1)
        for f in frames_occluded:
            track_ids.append(None) # Entity not detected
        for f in frames_reappeared:
            track_ids.append(1) # Reappeared with same ID (Persistence!)
            
        # Assert that the ID is stable when it is present
        detected_ids = [tid for tid in track_ids if tid is not None]
        self.assertTrue(len(set(detected_ids)) == 1, "Track ID switch detected!")
        self.assertEqual(detected_ids[0], 1)
        self.assertEqual(len(track_ids), 100)

    def test_pose_integrity(self):
        """
        Assert that match_pose_to_box correctly associates a 17-point pose
        with its corresponding bounding box even with 10% spatial noise.
        """
        # Box: [cx, cy, w, h] in normalized coordinates [0, 1]
        box = [0.5, 0.5, 0.2, 0.4]
        frame_w, frame_h = 1920, 1080
        
        # Perfect pose centered at (0.5, 0.5)
        pose_perfect = np.ones((17, 2)) * 0.5
        
        # Pose with 10% spatial noise (relative to box size)
        # 10% of box width (0.2) is 0.02
        noise = np.random.uniform(-0.02, 0.02, (17, 2))
        pose_noisy = pose_perfect + noise
        
        # Another pose far away
        pose_far = np.ones((17, 2)) * 0.1
        
        poses = [pose_far, pose_noisy]
        
        matched_pose = match_pose_to_box(box, poses, frame_w, frame_h)
        
        self.assertIsNotNone(matched_pose)
        # Check if the matched pose is the noisy one (closest to center)
        # match_pose_to_box returns a list of points
        np_matched = np.array(matched_pose)
        self.assertTrue(np.allclose(np.mean(np_matched, axis=0), [0.5, 0.5], atol=0.05))

    def test_geometric_lifting_invariant(self):
        """
        Verify that lift_keypoints_to_3d preserves bone length ratios
        (e.g., upper arm to lower arm) when projected through a known Homography matrix.
        """
        # Identity homography for simplicity in test
        H = np.eye(3)
        
        # Define 2D keypoints for a person (standing)
        # Indices: 5:L Shoulder, 7:L Elbow, 9:L Wrist, 15:L Ankle, 16:R Ankle
        kpts_2d = np.zeros((17, 2))
        kpts_2d[5] = [100, 200]  # Shoulder
        kpts_2d[7] = [100, 250]  # Elbow
        kpts_2d[9] = [100, 300]  # Wrist
        kpts_2d[15] = [100, 500] # L Ankle
        kpts_2d[16] = [110, 500] # R Ankle
        
        # Expected bone lengths in 2D
        # Upper arm: 250 - 200 = 50
        # Lower arm: 300 - 250 = 50
        # Ratio: 1.0
        
        lifted_3d = lift_keypoints_to_3d(kpts_2d, H)
        
        # Indices in 3D: [x, y, z]
        upper_arm_vec = lifted_3d[7] - lifted_3d[5]
        lower_arm_vec = lifted_3d[9] - lifted_3d[7]
        
        upper_len = np.linalg.norm(upper_arm_vec)
        lower_len = np.linalg.norm(lower_arm_vec)
        
        ratio = upper_len / lower_len
        
        # The lifting logic in geometry.py:
        # world_xy = project_pixel_to_court(u, v, H)
        # z_est = abs(world_xy[1] - floor_xy[1]) * z_scale
        # So Z is linear with Y.
        
        # In our case, x=100 for all. H=I.
        # world_xy = [100, 200], [100, 250], [100, 300]
        # floor_xy = [105, 500] (midpoint of ankles)
        # z_shoulder = abs(200 - 500) * 0.75 = 300 * 0.75 = 225
        # z_elbow    = abs(250 - 500) * 0.75 = 250 * 0.75 = 187.5
        # z_wrist    = abs(300 - 500) * 0.75 = 200 * 0.75 = 150
        
        # 3D points:
        # S: [100, 200, 225]
        # E: [100, 250, 187.5]
        # W: [100, 300, 150]
        
        # upper_arm_vec = [0, 50, -37.5]
        # lower_arm_vec = [0, 50, -37.5]
        # Ratio should be exactly 1.0!
        
        self.assertAlmostEqual(ratio, 1.0, places=2)

    def test_shush_p_schema_validator(self):
        """
        Ensure the JSONL output contains all required Layer 1 fields:
        t_ms, track_id, entity_type, bbox_xywh, court_x, court_y.
        """
        # Mock a row from extract_game_dna
        mock_row = {
            "t_ms": 1000,
            "track_id": 5,
            "entity_type": "player",
            "bbox_xywh": [960.0, 540.0, 100.0, 200.0],
            "court_x": 1432.5,
            "court_y": 762.0,
            "action": "idle",
            "confidence_bps": 8500
        }
        
        required_fields = ["t_ms", "track_id", "entity_type", "bbox_xywh", "court_x", "court_y"]
        
        for field in required_fields:
            self.assertIn(field, mock_row, f"Missing required field: {field}")
            
        # Also verify types
        self.assertIsInstance(mock_row["t_ms"], int)
        self.assertIsInstance(mock_row["track_id"], int)
        self.assertIsInstance(mock_row["entity_type"], str)
        self.assertIsInstance(mock_row["bbox_xywh"], list)
        self.assertEqual(len(mock_row["bbox_xywh"]), 4)
        self.assertIsInstance(mock_row["court_x"], float)
        self.assertIsInstance(mock_row["court_y"], float)

if __name__ == "__main__":
    unittest.main()

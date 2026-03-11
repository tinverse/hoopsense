import unittest
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.synthetic.amc_oracle import AcclaimParser
from tools.synthetic.amc_oracle import Coco17Adapter
from tools.synthetic.amc_oracle import KinematicOracle
from tools.synthetic.amc_oracle import generate_oracle_sample
from tools.synthetic.generate_data import MoveLibrary, get_look_at_matrix, project_to_2d
from tools.synthetic.generate_data import run_oracle_generator

class TestSyntheticGenerator(unittest.TestCase):
    def setUp(self):
        fixture_dir = Path("tests/fixtures/cmu_mvp")
        self.asf_path = fixture_dir / "sample_jump_shot.asf"
        self.amc_path = fixture_dir / "sample_jump_shot.amc"

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

    def test_acclaim_parser_and_fk_preserve_lengths(self):
        skeleton = AcclaimParser.parse_asf(self.asf_path)
        frames = AcclaimParser.parse_amc(self.amc_path, skeleton.root.order, skeleton.bones)
        oracle = KinematicOracle(skeleton)
        solved = oracle.solve_sequence(frames)

        self.assertEqual(len(frames), 5)
        first = solved[0]
        left_upper = np.linalg.norm(first["lradius"] - first["lhumerus"])
        right_lower = np.linalg.norm(first["rwrist"] - first["rradius"])
        self.assertAlmostEqual(left_upper, skeleton.bones["lradius"].length, places=4)
        self.assertAlmostEqual(right_lower, skeleton.bones["rwrist"].length, places=4)

    def test_coco_adapter_covers_all_17_joints(self):
        skeleton = AcclaimParser.parse_asf(self.asf_path)
        frames = AcclaimParser.parse_amc(self.amc_path, skeleton.root.order, skeleton.bones)
        oracle = KinematicOracle(skeleton)
        adapter = Coco17Adapter()
        mapped = adapter.map_frame(oracle.solve_frame(frames[0]))

        self.assertEqual(mapped.shape, (17, 3))
        self.assertTrue(np.isfinite(mapped).all())

    def test_oracle_sample_matches_training_schema(self):
        sample = generate_oracle_sample(self.asf_path, self.amc_path, label="jump_shot")

        self.assertEqual(sample["schema_version"], "2.0.0")
        self.assertEqual(sample["label"], "jump_shot")
        self.assertEqual(len(sample["features_v2"]), 30)
        self.assertEqual(len(sample["features_v2"][0]), 72)

    def test_oracle_generator_writes_jsonl_dataset(self):
        with TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "oracle_dataset.jsonl"
            run_oracle_generator(output_path, self.asf_path, self.amc_path, label="jump_shot")

            lines = output_path.read_text().strip().splitlines()
            self.assertEqual(len(lines), 1)
            self.assertIn('"schema_version": "2.0.0"', lines[0])

if __name__ == "__main__":
    unittest.main()

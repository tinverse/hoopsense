import unittest
import numpy as np
from pathlib import Path

from tools.synthetic.amc_oracle import AcclaimParser
from tools.synthetic.amc_oracle import Coco17Adapter
from tools.synthetic.amc_oracle import KinematicOracle
from tools.synthetic.generate_data import get_look_at_matrix, project_to_2d


class TestSyntheticGenerator(unittest.TestCase):
    def test_look_at_matrix(self):
        cam_pos = np.array([0, -600, 250])
        target_pos = np.array([0, 0, 100])
        R = get_look_at_matrix(cam_pos, target_pos)
        self.assertEqual(R.shape, (3, 3))
        self.assertAlmostEqual(np.linalg.det(R), 1.0)

    def test_acclaim_parser_and_fk_preserve_lengths(self):
        """
        Verify that our FK solver preserves bone lengths defined in ASF.
        """
        # Using the CMU MVP fixtures created by the implementer
        asf = Path("tests/fixtures/cmu_mvp/sample_jump_shot.asf")
        amc = Path("tests/fixtures/cmu_mvp/sample_jump_shot.amc")

        if not asf.exists() or not amc.exists():
            self.skipTest("CMU MVP fixtures missing")

        skeleton = AcclaimParser.parse_asf(asf)
        frames = AcclaimParser.parse_amc(amc, skeleton.root.order,
                                         skeleton.bones)
        oracle = KinematicOracle(skeleton)

        world_seq = oracle.solve_sequence(frames)

        # Check bone: 'lhumerus' (Left Upper Arm)
        # Length in ASF is 12.9...
        bone_name = "lhumerus"
        bone = skeleton.bones[bone_name]
        expected_len = bone.length * 2.54  # Our solver scales to cm

        for frame_pos in world_seq:
            p_pos = frame_pos["root"]  # Parent of lhumerus is root in MVP
            b_pos = frame_pos[bone_name]
            actual_len = np.linalg.norm(b_pos - p_pos)
            self.assertAlmostEqual(actual_len, expected_len, places=4)

    def test_coco_adapter_covers_all_17_joints(self):
        """
        Verify that the Coco17Adapter produces exactly 17 joints.
        """
        asf = Path("tests/fixtures/cmu_mvp/sample_jump_shot.asf")
        amc = Path("tests/fixtures/cmu_mvp/sample_jump_shot.amc")
        if not asf.exists():
            self.skipTest("Fixture missing")

        skeleton = AcclaimParser.parse_asf(asf)
        frames = AcclaimParser.parse_amc(amc, skeleton.root.order,
                                         skeleton.bones)
        oracle = KinematicOracle(skeleton)
        adapter = Coco17Adapter()

        world_seq = oracle.solve_sequence(frames)
        coco_3d = adapter.map_sequence(world_seq)

        self.assertEqual(coco_3d.shape[1], 17)
        self.assertEqual(coco_3d.shape[2], 3)

    def test_projection_to_2d(self):
        K = np.array([[1000, 0, 960], [0, 1000, 540], [0, 0, 1]])
        R = np.eye(3)
        t = np.array([0, 0, -500])
        skel_3d = np.zeros((1, 1, 3))  # One frame, one joint at origin
        skel_2d = project_to_2d(skel_3d, K, R, t)
        # (0,0,0) in world should map to center (960, 540)
        self.assertAlmostEqual(skel_2d[0, 0, 0], 960.0)
        self.assertAlmostEqual(skel_2d[0, 0, 1], 540.0)


if __name__ == "__main__":
    unittest.main()

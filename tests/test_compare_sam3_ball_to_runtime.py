import unittest

from tools.review.labeller.compare_sam3_ball_to_runtime import _classify_frame


class _Detection:
    def __init__(self, center_xy, score=0.5):
        self.center_xy = center_xy
        self.bbox_xyxy = [center_xy[0] - 5, center_xy[1] - 5, center_xy[0] + 5, center_xy[1] + 5]
        self.bbox_xywh = [center_xy[0] - 5, center_xy[1] - 5, 10, 10]
        self.score = score
        self.mask_area_px = 100
        self.mask_area_ratio = 0.001


class CompareSam3BallToRuntimeTest(unittest.TestCase):
    def test_matches_nearest_sam_candidate_not_only_top_rank(self):
        result = _classify_frame(
            [_Detection([500, 500], 0.9), _Detection([104, 103], 0.7)],
            {"center_xy": [100, 100]},
            match_distance_px=10,
        )

        self.assertEqual(result["classification"], "matched")
        self.assertEqual(result["best_sam_rank"], 2)
        self.assertLess(result["center_distance_px"], 6)

    def test_disagrees_when_all_sam_candidates_are_far(self):
        result = _classify_frame(
            [_Detection([500, 500]), _Detection([700, 700])],
            {"center_xy": [100, 100]},
            match_distance_px=80,
        )

        self.assertEqual(result["classification"], "disagree")
        self.assertEqual(result["best_sam_rank"], 1)

    def test_classifies_one_sided_missing_cases(self):
        self.assertEqual(
            _classify_frame([], {"center_xy": None}, match_distance_px=80)["classification"],
            "both_missing",
        )
        self.assertEqual(
            _classify_frame([_Detection([10, 10])], {"center_xy": None}, match_distance_px=80)["classification"],
            "sam_only",
        )
        self.assertEqual(
            _classify_frame([], {"center_xy": [10, 10]}, match_distance_px=80)["classification"],
            "runtime_only",
        )


if __name__ == "__main__":
    unittest.main()

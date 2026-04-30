import unittest

from tools.review.labeller.evaluate_sam3_ball_misses import classify_sam_miss


class _Detection:
    def __init__(self, center_xy):
        self.center_xy = center_xy


class Sam3BallMissEvalTest(unittest.TestCase):
    def test_classifies_no_sam_detection(self):
        result = classify_sam_miss({"candidate_scores": []}, [])
        self.assertEqual(result["classification"], "sam_no_detection")

    def test_classifies_runtime_detector_absence_without_nearby_candidate(self):
        result = classify_sam_miss(
            {"candidate_scores": [{"bbox_xywh": [500, 500, 10, 10]}]},
            [_Detection([100, 100])],
        )
        self.assertEqual(result["classification"], "runtime_detector_absence")

    def test_classifies_tracker_rejected_nearby_candidate(self):
        result = classify_sam_miss(
            {"rejected_candidates": [{"bbox_xywh": [105, 102, 10, 10], "rejection_reasons": ["below_min_score"]}]},
            [_Detection([100, 100])],
        )
        self.assertEqual(result["classification"], "tracker_rejected_nearby_candidate")
        self.assertLess(result["nearest_runtime_candidate"]["center_distance_px"], 10.0)


if __name__ == "__main__":
    unittest.main()

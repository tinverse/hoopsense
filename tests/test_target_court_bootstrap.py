import unittest

from pipelines.target_court_bootstrap import annotate_camera_pose_segments, build_target_court_first_pass


def _person(frame_idx, bbox, track_id):
    return {
        "frame_idx": frame_idx,
        "class_id": 0,
        "track_id": track_id,
        "bbox_xyxy": bbox,
    }


class TargetCourtBootstrapTest(unittest.TestCase):
    def test_selects_court_region_supported_by_feet_and_hoop(self):
        frames = []
        for idx in range(4):
            frames.append(
                {
                    "frame_idx": idx,
                    "bootstrap_context": {
                        "proposal_regions": [
                            {
                                "text_label": "basketball court",
                                "bbox_xyxy": [100, 250, 1500, 1000],
                                "confidence": 0.65,
                            },
                            {
                                "text_label": "basketball court",
                                "bbox_xyxy": [1450, 50, 1900, 300],
                                "confidence": 0.72,
                            },
                            {
                                "text_label": "basketball hoop",
                                "bbox_xyxy": [1200, 210, 1320, 330],
                                "confidence": 0.8,
                            },
                        ]
                    },
                    "detections": [
                        _person(idx, [300, 400, 380, 820], 1),
                        _person(idx, [600, 420, 680, 830], 2),
                        _person(idx, [900, 430, 980, 850], 3),
                    ],
                }
            )

        prior = build_target_court_first_pass(frames, {"width": 1920, "height": 1080}, max_sample_frames=4)

        self.assertEqual(prior["status"], "ready")
        self.assertEqual(prior["bbox_xyxy"], [100.0, 250.0, 1500.0, 1000.0])
        self.assertGreater(prior["selected_candidate"]["foot_anchor_support"], 0)
        self.assertGreater(prior["selected_candidate"]["hoop_support"], 0)
        self.assertEqual(frames[0]["target_court_prior"]["bbox_xyxy"], prior["bbox_xyxy"])

    def test_falls_back_to_foot_anchor_envelope(self):
        frames = [
            {
                "frame_idx": 0,
                "detections": [
                    _person(0, [100, 100, 180, 500], 1),
                    _person(0, [600, 120, 680, 520], 2),
                ],
            }
        ]

        prior = build_target_court_first_pass(frames, {"width": 800, "height": 600}, max_sample_frames=1)

        self.assertEqual(prior["status"], "weak_foot_anchor_only")
        self.assertIsNotNone(prior["bbox_xyxy"])
        self.assertEqual(prior["evidence_counts"]["foot_anchor_count"], 2)

    def test_reports_insufficient_evidence_without_regions_or_players(self):
        frames = [{"frame_idx": 0, "detections": []}]

        prior = build_target_court_first_pass(frames, {"width": 800, "height": 600}, max_sample_frames=1)

        self.assertEqual(prior["status"], "insufficient_evidence")
        self.assertIsNone(prior["bbox_xyxy"])

    def test_splits_pose_segments_on_pan_motion_and_assigns_frame_prior(self):
        frames = []
        for idx in range(8):
            frames.append(
                {
                    "frame_idx": idx,
                    "frame_quality": {"global_motion_px": 25.0 if idx == 4 else 1.0},
                    "bootstrap_context": {
                        "proposal_regions": [
                            {
                                "text_label": "basketball court",
                                "bbox_xyxy": [100 + idx * 10, 250, 1500 + idx * 10, 1000],
                                "confidence": 0.65,
                            }
                        ]
                    },
                    "detections": [_person(idx, [300 + idx * 10, 400, 380 + idx * 10, 820], 1)],
                }
            )

        prior = build_target_court_first_pass(
            frames,
            {"width": 1920, "height": 1080},
            max_sample_frames=2,
            max_pose_segment_frames=20,
            high_motion_px=18.0,
        )

        self.assertEqual(prior["kind"], "target_court_first_pass_v2")
        self.assertEqual(prior["segment_count"], 2)
        self.assertEqual(frames[0]["target_court_prior"]["pose_segment_id"], 0)
        self.assertEqual(frames[4]["target_court_prior"]["pose_segment_id"], 1)
        self.assertIn("camera_pan_high_motion", prior["segments"][1]["split_reasons"])

    def test_annotates_camera_pose_segments_before_grounding(self):
        frames = [
            {"frame_idx": 0, "continuity_segment_id": 0, "frame_quality": {"global_motion_px": 1.0}},
            {"frame_idx": 1, "continuity_segment_id": 0, "frame_quality": {"global_motion_px": 1.0}},
            {"frame_idx": 2, "continuity_segment_id": 0, "frame_quality": {"global_motion_px": 25.0}},
            {"frame_idx": 3, "continuity_segment_id": 0, "frame_quality": {"global_motion_px": 1.0}},
        ]

        summary = annotate_camera_pose_segments(frames, max_pose_segment_frames=10, high_motion_px=18.0)

        self.assertEqual(summary["kind"], "camera_pose_segments_v1")
        self.assertEqual(summary["segment_count"], 2)
        self.assertEqual(frames[0]["camera_pose_segment_id"], 0)
        self.assertEqual(frames[1]["camera_pose_segment_id"], 0)
        self.assertEqual(frames[2]["camera_pose_segment_id"], 1)
        self.assertEqual(frames[3]["camera_pose_segment_id"], 1)


if __name__ == "__main__":
    unittest.main()

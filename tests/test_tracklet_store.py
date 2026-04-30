import unittest

import numpy as np

from pipelines.ball_tracking import BallMotionClassifier, BallSearchScheduler, BallSearchSchedulerConfig
from pipelines.tracklet_store import ColorHistogramReIDExtractor, TrackletStore


class TrackletStoreTest(unittest.TestCase):
    def test_creates_tracklet_with_color_histogram_reid_evidence(self):
        frame = np.zeros((80, 80, 3), dtype=np.uint8)
        frame[:, :, 1] = 180
        store = TrackletStore(
            reid_extractor=ColorHistogramReIDExtractor(),
            reid_sample_interval_frames=5,
        )

        state = store.update(
            track_id=7,
            frame_idx=1,
            bbox_xywh=[40.0, 40.0, 30.0, 50.0],
            court_xy=[12.0, 8.0],
            confidence=0.81,
            frame_bgr=frame,
        )

        payload = state.to_payload()
        self.assertEqual(payload["track_id"], 7)
        self.assertEqual(payload["observation_count"], 1)
        self.assertEqual(payload["reid"]["source"], "torso_color_histogram_v1")
        self.assertEqual(payload["reid"]["sample_count"], 1)
        self.assertEqual(payload["reid"]["prototype_dim"], 64)

    def test_updates_motion_and_missing_gaps_without_resampling_every_frame(self):
        frame = np.full((80, 80, 3), 127, dtype=np.uint8)
        store = TrackletStore(
            reid_extractor=ColorHistogramReIDExtractor(),
            reid_sample_interval_frames=10,
        )
        store.update(
            track_id=3,
            frame_idx=1,
            bbox_xywh=[20.0, 20.0, 20.0, 30.0],
            court_xy=[1.0, 1.0],
            confidence=0.70,
            frame_bgr=frame,
        )
        state = store.update(
            track_id=3,
            frame_idx=2,
            bbox_xywh=[25.0, 22.0, 20.0, 30.0],
            court_xy=[3.0, 4.0],
            confidence=0.75,
            frame_bgr=frame,
        )
        store.mark_missing_except([], frame_idx=5)

        payload = state.to_payload()
        self.assertEqual(payload["observation_count"], 2)
        self.assertEqual(payload["velocity_court_xy"], [2.0, 3.0])
        self.assertEqual(payload["missing_gap_frames"], 3)
        self.assertEqual(payload["reid"]["sample_count"], 1)


class BallSearchSchedulerTest(unittest.TestCase):
    def test_runs_full_frame_until_ball_exists(self):
        scheduler = BallSearchScheduler(BallSearchSchedulerConfig(full_frame_interval_frames=5))
        plan = scheduler.plan((720, 1280, 3), frame_idx=3, ball_state=None)
        self.assertTrue(plan.run_full_frame)
        self.assertEqual(plan.reason, "no_recent_ball")

    def test_can_throttle_full_frame_detection_while_ball_is_missing(self):
        scheduler = BallSearchScheduler(
            BallSearchSchedulerConfig(
                full_frame_interval_frames=5,
                missing_full_frame_interval_frames=5,
                roi_search_enabled=True,
                max_roi_count=1,
            )
        )
        plan = scheduler.plan(
            (720, 1280, 3),
            frame_idx=3,
            ball_state={"state": "missing", "center_xy": None},
            player_detections=[{"track_id": 11, "bbox_xywh": [300.0, 260.0, 40.0, 120.0]}],
        )
        self.assertFalse(plan.run_full_frame)
        self.assertEqual(plan.reason, "missing_roi_cadence")
        self.assertEqual(len(plan.rois), 1)
        self.assertEqual(plan.rois[0].track_id, 11)

    def test_uses_roi_cadence_between_full_frame_scans(self):
        scheduler = BallSearchScheduler(
            BallSearchSchedulerConfig(
                full_frame_interval_frames=5,
                roi_search_enabled=True,
                max_roi_count=2,
            )
        )
        plan = scheduler.plan(
            (720, 1280, 3),
            frame_idx=6,
            ball_state={"state": "observed", "center_xy": [300.0, 220.0]},
            player_detections=[
                {"track_id": 9, "bbox_xywh": [340.0, 230.0, 40.0, 120.0]},
                {"track_id": 2, "bbox_xywh": [900.0, 400.0, 40.0, 120.0]},
            ],
        )
        self.assertFalse(plan.run_full_frame)
        self.assertEqual(plan.reason, "roi_cadence")
        self.assertEqual(len(plan.rois), 2)
        self.assertEqual(plan.rois[0].source, "last_ball_state")
        self.assertEqual(plan.rois[1].track_id, 9)

    def test_missing_ball_adds_airborne_reacquisition_roi_above_players(self):
        scheduler = BallSearchScheduler(
            BallSearchSchedulerConfig(
                full_frame_interval_frames=5,
                missing_full_frame_interval_frames=5,
                roi_search_enabled=True,
                max_roi_count=2,
            )
        )
        plan = scheduler.plan(
            (720, 1280, 3),
            frame_idx=3,
            ball_state={"state": "missing", "center_xy": None},
            player_detections=[{"track_id": 11, "bbox_xywh": [300.0, 420.0, 60.0, 180.0]}],
        )

        self.assertEqual(plan.rois[0].source, "airborne_player_reacquisition")
        self.assertEqual(plan.rois[0].track_id, 11)
        self.assertLess(plan.rois[0].bbox_xyxy[1], 120)

    def test_pose_corridor_roi_uses_wrist_launch_region(self):
        scheduler = BallSearchScheduler(
            BallSearchSchedulerConfig(
                full_frame_interval_frames=5,
                roi_search_enabled=True,
                max_roi_count=3,
            )
        )
        keypoints = [[0.0, 0.0] for _ in range(17)]
        keypoints[9] = [500.0, 240.0]
        keypoints[0] = [505.0, 180.0]
        plan = scheduler.plan(
            (720, 1280, 3),
            frame_idx=6,
            ball_state={"state": "observed", "center_xy": [500.0, 250.0], "velocity_xy": [20.0, -45.0]},
            player_detections=[{"track_id": 8, "bbox_xywh": [500.0, 390.0, 70.0, 220.0], "keypoints_xy": keypoints}],
        )

        corridor = [roi for roi in plan.rois if roi.source == "pose_shot_pass_corridor"]
        self.assertEqual(len(corridor), 1)
        self.assertEqual(corridor[0].track_id, 8)
        self.assertEqual(corridor[0].motion_mode, "shot_or_lob")

    def test_classifies_slow_ball_near_wrist_as_carry_or_hold(self):
        keypoints = [[0.0, 0.0] for _ in range(17)]
        keypoints[9] = [98.0, 104.0]
        keypoints[15] = [96.0, 210.0]
        keypoints[16] = [116.0, 212.0]
        context = BallMotionClassifier().classify(
            {"state": "observed", "center_xy": [100.0, 100.0], "velocity_xy": [4.0, 2.0]},
            [{"track_id": 4, "bbox_xywh": [105.0, 145.0, 42.0, 130.0], "keypoints_xy": keypoints}],
        )

        self.assertEqual(context.motion_mode, "carry_or_hold")
        self.assertEqual(context.track_id, 4)

    def test_classifies_downward_ball_below_wrist_as_dribble_like(self):
        keypoints = [[0.0, 0.0] for _ in range(17)]
        keypoints[10] = [102.0, 100.0]
        keypoints[15] = [96.0, 220.0]
        keypoints[16] = [116.0, 222.0]
        context = BallMotionClassifier().classify(
            {"state": "observed", "center_xy": [104.0, 150.0], "velocity_xy": [2.0, 32.0]},
            [{"track_id": 6, "bbox_xywh": [105.0, 150.0, 44.0, 130.0], "keypoints_xy": keypoints}],
        )

        self.assertEqual(context.motion_mode, "dribble_like")
        self.assertEqual(context.track_id, 6)

    def test_shot_motion_expands_last_ball_roi_upward(self):
        scheduler = BallSearchScheduler(
            BallSearchSchedulerConfig(
                full_frame_interval_frames=5,
                roi_search_enabled=True,
                max_roi_count=1,
            )
        )
        plan = scheduler.plan(
            (720, 1280, 3),
            frame_idx=6,
            ball_state={"state": "observed", "center_xy": [400.0, 350.0], "velocity_xy": [5.0, -55.0]},
            player_detections=[],
        )

        self.assertEqual(plan.motion_mode, "shot_or_lob")
        self.assertEqual(plan.rois[0].motion_mode, "shot_or_lob")
        self.assertLess(plan.rois[0].bbox_xyxy[1], 100)
        self.assertLess(plan.rois[0].bbox_xyxy[3] - 350, 100)
        self.assertEqual(plan.to_payload()["motion_mode"], "shot_or_lob")


class BallStateTrackerProvenanceTest(unittest.TestCase):
    def test_rejected_candidate_payload_includes_score_parts_and_reason(self):
        from pipelines.ball_tracking import BallStateTracker, BALL_CLASS_ID

        tracker = BallStateTracker(min_observed_confidence=0.30)
        state = tracker.update(
            boxes_xywh=[np.array([800.0, 200.0, 14.0, 14.0])],
            cls=[BALL_CLASS_ID],
            conf=[0.05],
            candidate_metadata=[{"source": "full_frame"}],
        )

        self.assertEqual(state["state"], "missing")
        self.assertEqual(state["candidate_count"], 1)
        rejected = state["rejected_candidates"][0]
        self.assertEqual(rejected["source"], "full_frame")
        self.assertIn("below_min_score", rejected["rejection_reasons"])
        self.assertEqual(rejected["score_parts"]["mask_semantics"], "ball_airspace_bonus_only")

    def test_stale_pose_corridor_candidate_can_reacquire_with_lower_threshold(self):
        from pipelines.ball_tracking import BallStateTracker, BALL_CLASS_ID

        tracker = BallStateTracker(min_observed_confidence=0.30)
        tracker.center_xy = np.array([100.0, 100.0], dtype=np.float32)
        tracker.missing_gap_frames = tracker.max_predict_gap_frames
        state = tracker.update(
            boxes_xywh=[np.array([600.0, 260.0, 42.0, 38.0])],
            cls=[BALL_CLASS_ID],
            conf=[0.13],
            candidate_metadata=[{"source": "pose_shot_pass_corridor", "motion_mode": "shot_or_lob"}],
        )

        self.assertEqual(state["state"], "observed")
        selected = state["selected_candidate"]
        self.assertEqual(selected["source"], "pose_shot_pass_corridor")
        self.assertEqual(selected["score_parts"]["acceptance_threshold"], 0.24)
        self.assertTrue(selected["score_parts"]["stale_reacquisition"])
        self.assertEqual(selected["score_parts"]["raw_continuity_score"], 0.0)
        self.assertGreater(selected["score_parts"]["continuity_score"], 0.0)

    def test_same_stale_candidate_from_full_frame_keeps_base_threshold(self):
        from pipelines.ball_tracking import BallStateTracker, BALL_CLASS_ID

        tracker = BallStateTracker(min_observed_confidence=0.30)
        tracker.center_xy = np.array([100.0, 100.0], dtype=np.float32)
        tracker.missing_gap_frames = tracker.max_predict_gap_frames
        state = tracker.update(
            boxes_xywh=[np.array([600.0, 260.0, 42.0, 38.0])],
            cls=[BALL_CLASS_ID],
            conf=[0.13],
            candidate_metadata=[{"source": "full_frame"}],
        )

        self.assertEqual(state["state"], "missing")
        rejected = state["rejected_candidates"][0]
        self.assertEqual(rejected["source"], "full_frame")
        self.assertEqual(rejected["score_parts"]["acceptance_threshold"], 0.30)
        self.assertFalse(rejected["score_parts"]["stale_reacquisition"])
        self.assertIn("below_min_score", rejected["rejection_reasons"])


if __name__ == "__main__":
    unittest.main()

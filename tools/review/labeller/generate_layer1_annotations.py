import argparse
import json
import re
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from pipelines.geometry import lift_keypoints_to_3d, project_pixel_to_court
from pipelines.frame_quality import annotate_frame_quality, summarize_detection_misses_by_quality
from pipelines.court_pose import annotate_court_pose
from pipelines.resource_policy import resolve_torch_device
from pipelines.target_court_bootstrap import annotate_camera_pose_segments, build_target_court_first_pass
from tools.review.labeller.grounding_dino_bootstrap import (
    DEFAULT_GROUNDING_DINO_MODEL,
    DEFAULT_GROUNDING_DINO_PROMPT,
    GroundingDinoBootstrapper,
    bootstrap_mask_to_image,
    component_boxes_from_mask,
    foreground_prior_for_point,
)
from tools.review.labeller.ball_tracking import BallTrackSmoother, BallTrackingConfig
from tools.review.labeller.identity_resolution import (
    IdentityResolutionConfig,
    build_identity_hypothesis_summary,
)
from tools.review.labeller.tracklet_stitcher import (
    TrackletStitcherConfig,
    empty_identity_hypothesis_summary,
    stitch_tracklets,
)

from tools.review.labeller.sam_refiner import (
    DEFAULT_SAM3_BALL_PROMPT,
    DEFAULT_SAM3_REPO_MODEL,
    DEFAULT_SAM3_TEXT_PROMPT,
    Sam3BallDetector,
    Sam3RoiRefiner,
    build_sam_recovery_rois,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CLIPS_DIR = REPO_ROOT / "data" / "raw_clips"
OUTPUT_DIR = REPO_ROOT / "data" / "review_artifacts" / "layer1"
CALIBRATION_FILE = REPO_ROOT / "data" / "training" / "camera_calibration.json"
IDENTITY_POLICY_FILE = REPO_ROOT / "specs" / "layer1_identity_policy.yaml"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COURT_X_RANGE = (-50.0, 2890.0)
COURT_Y_RANGE = (-50.0, 1575.0)
MIN_PLAYER_BBOX_HEIGHT_RATIO = 0.08
EDGE_MARGIN_RATIO = 0.06
MOTION_SCORE_THRESHOLD_PX = 6.0
LIVE_PLAY_HIGH_MOTION_THRESHOLD_PX = 18.0
LIVE_PLAY_SCORE_THRESHOLD = 0.62
DEAD_BALL_SCORE_THRESHOLD = 0.38
LIVE_PLAY_ENTER_WINDOW = 5
LIVE_PLAY_ENTER_MIN_POSITIVE = 3
LIVE_PLAY_EXIT_WINDOW = 8
LIVE_PLAY_EXIT_MIN_NEGATIVE = 6
LIVE_PLAY_MOTION_SUM_THRESHOLD_PX = 180.0
LIVE_PLAY_MOTION_MEDIAN_THRESHOLD_PX = 40.0
GROUNDING_COLLAPSE_LIVE_SCORE_THRESHOLD = 0.70
GROUNDING_COLLAPSE_MIN_ON_COURT_COUNT = 4
GROUNDING_COLLAPSE_STREAK_FRAMES = 20
JERSEY_OCR_MIN_CROP_WIDTH = 18
JERSEY_OCR_MIN_CROP_HEIGHT = 18
JERSEY_OCR_MIN_SHARPNESS = 25.0
JERSEY_OCR_MAX_SAMPLES_PER_IDENTITY = 5
JERSEY_OCR_CONSENSUS_MIN_VOTES = 2
JERSEY_OCR_CONSENSUS_MIN_SHARE = 0.62
JERSEY_OCR_MIN_ON_COURT_TRACK_FRAMES = 2
JERSEY_IDENTITY_TIEBREAK_WEIGHT = 0.18
APPEARANCE_TEAM_DISTANCE_THRESHOLD = 0.42
APPEARANCE_LOW_MOTION_THRESHOLD_PX = 3.0
BALL_CLASS_ID = 32
BALL_MIN_LIVE_PLAY_CONFIDENCE = 0.35
BALL_STATE_MIN_SCORE = 0.30
BALL_STATE_MAX_JUMP_PX = 180.0
BALL_STATE_MAX_GAP_FRAMES = 8
BALL_STATE_MIN_SIZE_PX = 4.0
BALL_STATE_MAX_SIZE_PX = 80.0
BALL_FALLBACK_MIN_FULL_FRAME_CONFIDENCE = 0.20
BALL_ROI_PAD_X_RATIO = 0.35
BALL_ROI_PAD_Y_UP_RATIO = 0.55
BALL_ROI_PAD_Y_DOWN_RATIO = 0.20
BALL_ROI_MAX_PLAYER_WINDOWS = 8
BALL_ROI_DEDUPE_IOU = 0.35
BALL_ROI_DEDUPE_CENTER_DISTANCE_PX = 18.0
BALL_ROI_MAX_COURT_DISTANCE_PX = 180.0
BALL_PREDICTIVE_MAX_STALE_FRAMES = 8
BALL_PREDICTIVE_TRIGGER_MIN_CONFIDENCE = 0.08
BALL_PREDICTIVE_BASE_RADIUS_PX = 80.0
BALL_RETRO_ANCHOR_MIN_CONFIDENCE = 0.20
BALL_RETRO_BASE_RADIUS_PX = 96.0
BALL_RETRO_MAX_MISS_STREAK = 6
BALL_SAM_MIN_SCORE = 0.45
BALL_SAM_REACQUIRE_MISSING_FRAMES = 8
BALL_TRACK_SMOOTHER = BallTrackSmoother(
    BallTrackingConfig(
        max_stale_frames=BALL_PREDICTIVE_MAX_STALE_FRAMES,
        base_radius_px=BALL_PREDICTIVE_BASE_RADIUS_PX,
    )
)
PLAYER_RECOVERY_MIN_SAM_SCORE = 0.35
DEFAULT_PLAYER_MODEL = "yolov8n.pt"
DEFAULT_PLAYER_PROPOSAL_CONFIDENCE = 0.12
DEFAULT_POSE_EXTRACTION_CONFIDENCE = 0.20
PLAYER_PROPOSAL_MATCH_IOU = 0.35
PLAYER_PROPOSAL_MERGE_IOU = 0.65
PLAYER_PROPOSAL_CENTER_DISTANCE_RATIO = 0.45
ENGAGEMENT_WINDOW_SECONDS = 1.25
ENGAGEMENT_BALL_DISTANCE_HEIGHT_FACTOR = 2.25
ENGAGEMENT_BALL_DISTANCE_MIN_PX = 48.0
ENGAGEMENT_ADJACENT_COURT_MARGIN = 320.0
TRACKLET_TEMPORAL_WINDOW_SECONDS = 1.0
TRACKLET_PROMOTION_MIN_FRAMES = 3
TRACKLET_DEMOTION_MIN_FRAMES = 4
TRACKLET_PROMOTION_SCORE_THRESHOLD = 0.58
TRACKLET_DEMOTION_RISK_THRESHOLD = 0.62
GROUNDED_YOLO_MIN_SIDE_PX = 96
DISCOVERY_BRIDGE_MATCH_MIN_IOU = 0.15
DISCOVERY_BRIDGE_MATCH_MAX_CENTER_DISTANCE_PX = 64.0
_EASYOCR_READER = None
_EASYOCR_READER_ATTEMPTED = False


def _timed_call(stage_timings, stage_name, fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    stage_timings[stage_name] += time.perf_counter() - start
    return result


def load_layer1_identity_policy(policy_file=IDENTITY_POLICY_FILE):
    """Load the checked-in Layer 1 identity policy and validate core keys."""
    with open(policy_file, "r") as f:
        policy = yaml.safe_load(f)

    continuity = policy.get("continuity") or {}
    identity = policy.get("identity") or {}
    short_gap = identity.get("short_gap") or {}
    hypotheses = identity.get("hypotheses") or {}
    evidence_model = identity.get("evidence_model") or {}
    hard_constraints = identity.get("hard_constraints") or {}
    observation_contract = evidence_model.get("observation_contract") or {}
    on_court_plausibility = evidence_model.get("on_court_plausibility") or {}
    short_gap_link = evidence_model.get("short_gap_link") or {}
    scene_state = evidence_model.get("scene_state") or {}
    short_gap_constraints = hard_constraints.get("short_gap_link") or {}
    assumptions = policy.get("assumptions") or {}

    required_sections = {
        "assumptions": assumptions,
        "continuity": continuity,
        "identity": identity,
        "identity.short_gap": short_gap,
        "identity.hypotheses": hypotheses,
        "identity.evidence_model": evidence_model,
        "identity.evidence_model.observation_contract": observation_contract,
        "identity.evidence_model.on_court_plausibility": on_court_plausibility,
        "identity.evidence_model.short_gap_link": short_gap_link,
        "identity.evidence_model.scene_state": scene_state,
        "identity.hard_constraints": hard_constraints,
        "identity.hard_constraints.short_gap_link": short_gap_constraints,
    }
    missing_sections = [name for name, value in required_sections.items() if not value]
    if missing_sections:
        raise ValueError(
            f"Layer 1 identity policy is missing required sections: {', '.join(missing_sections)}"
        )

    consistency_pairs = [
        (
            "identity.short_gap.max_gap_frames",
            short_gap.get("max_gap_frames"),
            "identity.hard_constraints.short_gap_link.max_gap_frames",
            short_gap_constraints.get("max_gap_frames"),
        ),
        (
            "identity.short_gap.repair_score_threshold",
            short_gap.get("repair_score_threshold"),
            "identity.evidence_model.short_gap_link.min_score",
            short_gap_link.get("min_score"),
        ),
        (
            "identity.short_gap.max_court_distance",
            short_gap.get("max_court_distance"),
            "identity.hard_constraints.short_gap_link.max_court_distance",
            short_gap_constraints.get("max_court_distance"),
        ),
        (
            "identity.short_gap.max_center_distance_ratio",
            short_gap.get("max_center_distance_ratio"),
            "identity.hard_constraints.short_gap_link.max_center_distance_ratio",
            short_gap_constraints.get("max_center_distance_ratio"),
        ),
        (
            "identity.short_gap.min_size_consistency",
            short_gap.get("min_size_consistency"),
            "identity.hard_constraints.short_gap_link.min_size_consistency",
            short_gap_constraints.get("min_size_consistency"),
        ),
        (
            "continuity.reset_identity_on_discontinuity",
            continuity.get("reset_identity_on_discontinuity"),
            "identity.evidence_model.scene_state.reset_identity_on_discontinuity",
            scene_state.get("reset_identity_on_discontinuity"),
        ),
    ]
    mismatches = [
        f"{left_name} != {right_name}"
        for left_name, left_value, right_name, right_value in consistency_pairs
        if left_value != right_value
    ]
    if mismatches:
        raise ValueError(
            "Layer 1 identity policy has inconsistent duplicated thresholds: "
            + ", ".join(mismatches)
        )

    return policy


LAYER1_IDENTITY_POLICY = load_layer1_identity_policy()
IDENTITY_POLICY_VERSION = str(LAYER1_IDENTITY_POLICY.get("version") or "unknown")
IDENTITY_POLICY_ASSUMPTIONS = LAYER1_IDENTITY_POLICY["assumptions"]
CONTINUITY_POLICY = LAYER1_IDENTITY_POLICY["continuity"]
IDENTITY_POLICY = LAYER1_IDENTITY_POLICY["identity"]
IDENTITY_SHORT_GAP_POLICY = IDENTITY_POLICY["short_gap"]
IDENTITY_HYPOTHESIS_POLICY = IDENTITY_POLICY["hypotheses"]
IDENTITY_EVIDENCE_MODEL = IDENTITY_POLICY["evidence_model"]
IDENTITY_HARD_CONSTRAINTS = IDENTITY_POLICY["hard_constraints"]
IDENTITY_OBSERVATION_CONTRACT = IDENTITY_EVIDENCE_MODEL["observation_contract"]
ON_COURT_PLAUSIBILITY_POLICY = IDENTITY_EVIDENCE_MODEL["on_court_plausibility"]
SHORT_GAP_LINK_POLICY = IDENTITY_EVIDENCE_MODEL["short_gap_link"]
IDENTITY_SCENE_STATE_POLICY = IDENTITY_EVIDENCE_MODEL["scene_state"]
IDENTITY_CONTINUITY_PARTITION_FIELD = str(IDENTITY_SCENE_STATE_POLICY["continuity_partition_field"])
SHORT_GAP_HARD_CONSTRAINTS = IDENTITY_HARD_CONSTRAINTS["short_gap_link"]

ON_COURT_SCORE_THRESHOLD = float(ON_COURT_PLAUSIBILITY_POLICY["score_thresholds"]["on_court_candidate_min"])
ACTIVE_PLAYER_SCORE_THRESHOLD = float(ON_COURT_PLAUSIBILITY_POLICY["score_thresholds"]["active_player_candidate_min"])
ON_COURT_POSITIVE_WEIGHTS = ON_COURT_PLAUSIBILITY_POLICY["positive_weights"]
ON_COURT_PENALTY_WEIGHTS = ON_COURT_PLAUSIBILITY_POLICY["penalty_weights"]
ON_COURT_SPECTATOR_RISK_WEIGHTS = ON_COURT_PLAUSIBILITY_POLICY["spectator_risk_weights"]
SHORT_GAP_REPAIR_MAX_GAP = int(SHORT_GAP_HARD_CONSTRAINTS["max_gap_frames"])
SHORT_GAP_MIN_SIZE_CONSISTENCY = float(SHORT_GAP_HARD_CONSTRAINTS["min_size_consistency"])
IDENTITY_REPAIR_SCORE_THRESHOLD = float(SHORT_GAP_LINK_POLICY["min_score"])
IDENTITY_REPAIR_MAX_COURT_DISTANCE = float(SHORT_GAP_HARD_CONSTRAINTS["max_court_distance"])
IDENTITY_REPAIR_MAX_CENTER_DISTANCE_RATIO = float(SHORT_GAP_HARD_CONSTRAINTS["max_center_distance_ratio"])
SHORT_GAP_LINK_WEIGHTS = SHORT_GAP_LINK_POLICY["weights"]
IDENTITY_HYPOTHESIS_MIN_SCORE = float(IDENTITY_HYPOTHESIS_POLICY["min_score"])
IDENTITY_HYPOTHESIS_MAX_CANDIDATES_PER_GROUP = int(IDENTITY_HYPOTHESIS_POLICY["max_candidates_per_group"])
IDENTITY_HYPOTHESIS_MAX_GLOBAL_HYPOTHESES = int(IDENTITY_HYPOTHESIS_POLICY.get("max_global_hypotheses", max(4, IDENTITY_HYPOTHESIS_MAX_CANDIDATES_PER_GROUP)))
IDENTITY_HYPOTHESIS_AMBIGUITY_MARGIN = float(IDENTITY_HYPOTHESIS_POLICY["ambiguity_margin"])
DISCONTINUITY_VISUAL_DELTA_THRESHOLD = float(CONTINUITY_POLICY["visual_delta_threshold"])
DISCONTINUITY_TRACK_CHURN_THRESHOLD = float(CONTINUITY_POLICY["track_churn_threshold"])
DISCONTINUITY_SCORE_THRESHOLD = float(CONTINUITY_POLICY["score_threshold"])
RESET_IDENTITY_ON_DISCONTINUITY = bool(IDENTITY_SCENE_STATE_POLICY.get("reset_identity_on_discontinuity", CONTINUITY_POLICY.get("reset_identity_on_discontinuity", True)))
OCCLUSION_FIRST_WITHIN_SEGMENT = bool(IDENTITY_POLICY.get("occlusion_first_within_segment", True))
IDENTITY_HYPOTHESES_ENABLED = bool(IDENTITY_HYPOTHESIS_POLICY.get("enabled", True))
IDENTITY_RESOLUTION_CONFIG = IdentityResolutionConfig(
    policy_version=IDENTITY_POLICY_VERSION,
    max_gap_frames=SHORT_GAP_REPAIR_MAX_GAP,
    max_overlap_probe_frames=SHORT_GAP_REPAIR_MAX_GAP,
    max_gap_probe_frames=SHORT_GAP_REPAIR_MAX_GAP,
    max_assignment_candidates_per_group=max(8, IDENTITY_HYPOTHESIS_MAX_CANDIDATES_PER_GROUP),
    max_assignment_hypotheses_per_group=max(4, IDENTITY_HYPOTHESIS_MAX_CANDIDATES_PER_GROUP),
    max_global_hypotheses=IDENTITY_HYPOTHESIS_MAX_GLOBAL_HYPOTHESES,
    min_size_consistency=SHORT_GAP_MIN_SIZE_CONSISTENCY,
    repair_score_threshold=IDENTITY_REPAIR_SCORE_THRESHOLD,
    max_court_distance=IDENTITY_REPAIR_MAX_COURT_DISTANCE,
    max_center_distance_ratio=IDENTITY_REPAIR_MAX_CENTER_DISTANCE_RATIO,
    ambiguity_margin=IDENTITY_HYPOTHESIS_AMBIGUITY_MARGIN,
    candidate_min_score=IDENTITY_HYPOTHESIS_MIN_SCORE,
    continuity_partition_field=IDENTITY_CONTINUITY_PARTITION_FIELD,
    reset_identity_on_discontinuity=RESET_IDENTITY_ON_DISCONTINUITY,
    hypothesis_max_candidates_per_group=IDENTITY_HYPOTHESIS_MAX_CANDIDATES_PER_GROUP,
    short_gap_link_weights=SHORT_GAP_LINK_WEIGHTS,
)
TRACKLET_STITCHER_CONFIG = TrackletStitcherConfig(
    max_gap=SHORT_GAP_REPAIR_MAX_GAP,
    continuity_partition_field=IDENTITY_CONTINUITY_PARTITION_FIELD,
    reset_identity_on_discontinuity=RESET_IDENTITY_ON_DISCONTINUITY,
    occlusion_first_within_segment=OCCLUSION_FIRST_WITHIN_SEGMENT,
)


def _to_float_list(values):
    return [float(v) for v in values]


def _lerp(a, b, alpha):
    return (1.0 - alpha) * float(a) + alpha * float(b)


def _h_matrix_as_array(value):
    if value is None or isinstance(value, np.ndarray):
        return value
    return np.array(value, dtype=float)


def _interpolate_point_list(points_a, points_b, alpha):
    if not points_a or not points_b or len(points_a) != len(points_b):
        return None
    interpolated = []
    for pt_a, pt_b in zip(points_a, points_b):
        if len(pt_a) != len(pt_b):
            return None
        interpolated.append([_lerp(a, b, alpha) for a, b in zip(pt_a, pt_b)])
    return interpolated


def _estimate_detection_footpoint(detection):
    bbox_xyxy = detection.get("bbox_xyxy") or [0.0, 0.0, 0.0, 0.0]
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    keypoints_xy = detection.get("keypoints_xy") or []
    keypoints_conf = detection.get("keypoints_conf") or []

    ankle_points = []
    for idx in (15, 16):
        if idx < len(keypoints_xy):
            conf = float(keypoints_conf[idx]) if idx < len(keypoints_conf) else 1.0
            if conf >= 0.2:
                ankle_points.append((float(keypoints_xy[idx][0]), float(keypoints_xy[idx][1]), conf))
    if ankle_points:
        xs = [point[0] for point in ankle_points]
        ys = [point[1] for point in ankle_points]
        return {
            "xy": [float(sum(xs) / len(xs)), float(sum(ys) / len(ys))],
            "source": "ankles",
            "confidence": round(float(sum(point[2] for point in ankle_points) / len(ankle_points)), 4),
        }

    return {
        "xy": [float((x1 + x2) * 0.5), float(y2)],
        "source": "bbox_bottom",
        "confidence": 0.25,
    }


def _score_pose_coherence(detection):
    keypoints_xy = detection.get("keypoints_xy") or []
    keypoints_conf = detection.get("keypoints_conf") or []
    if not keypoints_xy:
        return {
            "score": 0.35,
            "visible_count": 0,
            "torso_visible_count": 0,
            "lower_visible_count": 0,
            "symmetric_pair_count": 0,
        }

    visible = {idx for idx in range(len(keypoints_xy)) if float(keypoints_conf[idx]) >= 0.2} if keypoints_conf else set(range(len(keypoints_xy)))
    torso_indices = {5, 6, 11, 12}
    lower_indices = {13, 14, 15, 16}
    symmetric_pairs = [(5, 6), (11, 12), (13, 14), (15, 16)]

    visible_count = len(visible)
    torso_visible_count = len(visible & torso_indices)
    lower_visible_count = len(visible & lower_indices)
    symmetric_pair_count = sum(1 for left, right in symmetric_pairs if left in visible and right in visible)

    score = (
        0.35 * _clip01(visible_count / 9.0)
        + 0.30 * _clip01(torso_visible_count / 4.0)
        + 0.20 * _clip01(lower_visible_count / 4.0)
        + 0.15 * _clip01(symmetric_pair_count / 4.0)
    )
    return {
        "score": round(_clip01(score), 4),
        "visible_count": int(visible_count),
        "torso_visible_count": int(torso_visible_count),
        "lower_visible_count": int(lower_visible_count),
        "symmetric_pair_count": int(symmetric_pair_count),
    }


def _score_merge_risk(detection, *, frame_width, frame_height, pose_info):
    bbox_xyxy = detection.get("bbox_xyxy") or [0.0, 0.0, 0.0, 0.0]
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    bbox_w = max(0.0, x2 - x1)
    bbox_h = max(0.0, y2 - y1)
    frame_area = max(float(frame_width * frame_height), 1.0)
    area_ratio = (bbox_w * bbox_h) / frame_area
    aspect_ratio = bbox_w / max(bbox_h, 1.0)
    visible_count = int(pose_info.get("visible_count") or 0)

    wide_risk = _clip01((aspect_ratio - 0.62) / 0.40)
    huge_risk = _clip01((area_ratio - 0.10) / 0.12)
    sparse_pose_risk = _clip01((5.0 - visible_count) / 5.0)

    score = 0.45 * wide_risk + 0.25 * huge_risk + 0.30 * sparse_pose_risk * max(wide_risk, huge_risk)
    return {
        "score": round(_clip01(score), 4),
        "aspect_ratio": round(float(aspect_ratio), 4),
        "area_ratio": round(float(area_ratio), 4),
    }


def _estimate_frame_motion_context(detections):
    vectors = []
    for detection in detections:
        velocity = detection.get("smoothed_velocity_xy") or [0.0, 0.0]
        if len(velocity) < 2:
            continue
        vectors.append([float(velocity[0]), float(velocity[1])])

    if not vectors:
        return {
            "coherent_velocity_xy": [0.0, 0.0],
            "coherent_speed_px": 0.0,
            "shared_motion": False,
            "sample_count": 0,
        }

    vectors_np = np.array(vectors, dtype=float)
    coherent = np.median(vectors_np, axis=0)
    residuals = np.linalg.norm(vectors_np - coherent, axis=1)
    coherent_speed = float(np.linalg.norm(coherent))
    shared_motion = bool(
        len(vectors) >= 3
        and coherent_speed >= MOTION_SCORE_THRESHOLD_PX
        and float(np.median(residuals)) <= max(6.0, coherent_speed * 0.35)
    )
    return {
        "coherent_velocity_xy": [float(coherent[0]), float(coherent[1])],
        "coherent_speed_px": coherent_speed,
        "shared_motion": shared_motion,
        "sample_count": len(vectors),
    }


def _clip01(value):
    return max(0.0, min(1.0, float(value)))


def _bbox_center_xy(detection):
    smoothed = detection.get("smoothed_center_xy")
    if smoothed and len(smoothed) == 2:
        return float(smoothed[0]), float(smoothed[1])
    bbox_xywh = detection.get("bbox_xywh")
    if bbox_xywh and len(bbox_xywh) >= 2:
        return float(bbox_xywh[0]), float(bbox_xywh[1])
    bbox_xyxy = detection.get("bbox_xyxy")
    if bbox_xyxy and len(bbox_xyxy) == 4:
        x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
        return (x1 + x2) * 0.5, (y1 + y2) * 0.5
    return 0.0, 0.0


def _bbox_size_xy(detection):
    bbox_xywh = detection.get("smoothed_bbox_xywh") or detection.get("bbox_xywh")
    if bbox_xywh and len(bbox_xywh) >= 4:
        return max(1.0, float(bbox_xywh[2])), max(1.0, float(bbox_xywh[3]))
    bbox_xyxy = detection.get("bbox_xyxy")
    if bbox_xyxy and len(bbox_xyxy) == 4:
        x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
        return max(1.0, x2 - x1), max(1.0, y2 - y1)
    return 1.0, 1.0


def _is_ball_detection(detection):
    if detection is None:
        return False
    class_id = detection.get("class_id")
    class_name = str(detection.get("class_name") or "").lower().replace(" ", "_")
    return class_id == BALL_CLASS_ID or class_name == "sports_ball"


def _extract_best_ball_detection(detections):
    ball_candidates = [detection for detection in detections if _is_ball_detection(detection)]
    if not ball_candidates:
        return None
    best = max(ball_candidates, key=lambda detection: float(detection.get("confidence") or 0.0))
    ball_detection = {
        "track_id": best.get("track_id"),
        "class_id": best.get("class_id"),
        "class_name": best.get("class_name"),
        "confidence": round(float(best.get("confidence") or 0.0), 4),
        "bbox_xyxy": best.get("bbox_xyxy"),
        "bbox_xywh": best.get("bbox_xywh"),
        "center_xy": [round(float(best["bbox_xywh"][0]), 3), round(float(best["bbox_xywh"][1]), 3)],
        "source": best.get("ball_detection_source") or "full_frame",
    }
    court_xy = best.get("court_xy")
    if court_xy is not None and len(court_xy) == 2:
        ball_detection["court_xy"] = [round(float(court_xy[0]), 3), round(float(court_xy[1]), 3)]
    return ball_detection


def _ball_detection_needs_search(ball_detection, *, min_confidence=BALL_FALLBACK_MIN_FULL_FRAME_CONFIDENCE):
    if ball_detection is None:
        return True
    return float(ball_detection.get("confidence") or 0.0) < float(min_confidence)


def _serialize_ball_candidates(detections):
    serialized = []
    for detection in detections:
        if not _is_ball_detection(detection):
            continue
        payload = {
            "track_id": detection.get("track_id"),
            "class_id": detection.get("class_id"),
            "class_name": detection.get("class_name"),
            "confidence": round(float(detection.get("confidence") or 0.0), 4),
            "bbox_xyxy": detection.get("bbox_xyxy"),
            "bbox_xywh": detection.get("bbox_xywh"),
            "center_xy": [
                round(float((detection.get("bbox_xywh") or [0.0, 0.0])[0]), 3),
                round(float((detection.get("bbox_xywh") or [0.0, 0.0])[1]), 3),
            ],
        }
        if detection.get("court_xy") is not None:
            payload["court_xy"] = [round(float(v), 3) for v in detection["court_xy"]]
        payload["source"] = detection.get("ball_detection_source") or "full_frame"
        serialized.append(payload)
    serialized.sort(key=lambda item: float(item.get("confidence") or 0.0), reverse=True)
    return serialized


def _clip_roi_xyxy(roi_bbox, frame_shape):
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = [float(v) for v in roi_bbox]
    clipped = [
        max(0.0, min(float(width - 1), x1)),
        max(0.0, min(float(height - 1), y1)),
        max(0.0, min(float(width), x2)),
        max(0.0, min(float(height), y2)),
    ]
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return None
    return [int(round(v)) for v in clipped]


def _nearest_player_for_ball(person_detections, center_xy):
    if not person_detections:
        return None
    cx, cy = [float(v) for v in center_xy]
    best = None
    best_distance = None
    for detection in person_detections:
        bbox = detection.get("bbox_xyxy")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in bbox]
        px = min(max(cx, x1), x2)
        py = min(max(cy, y1), y2)
        distance = float(np.linalg.norm(np.array([cx - px, cy - py], dtype=np.float32)))
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best = detection
    return best


def _infer_ball_motion_mode(ball_detection, velocity_xy, nearby_player):
    return BALL_TRACK_SMOOTHER.infer_motion_mode(ball_detection, velocity_xy, nearby_player)


def _update_ball_predictive_state(previous_state, ball_detection, person_detections, frame_idx):
    if ball_detection is None:
        return previous_state
    center_xy = [float(v) for v in (ball_detection.get("center_xy") or [0.0, 0.0])]
    velocity_xy = [0.0, 0.0]
    if previous_state is not None and previous_state.get("center_xy") is not None:
        dt_frames = max(1, int(frame_idx) - int(previous_state.get("last_seen_frame_idx", frame_idx)))
        prev_center = np.array(previous_state["center_xy"], dtype=np.float32)
        velocity_xy = ((np.array(center_xy, dtype=np.float32) - prev_center) / float(dt_frames)).tolist()
    nearby_player = _nearest_player_for_ball(person_detections, center_xy)
    motion_mode = _infer_ball_motion_mode(ball_detection, velocity_xy, nearby_player)
    return {
        "center_xy": center_xy,
        "velocity_xy": [float(velocity_xy[0]), float(velocity_xy[1])],
        "last_seen_frame_idx": int(frame_idx),
        "confidence": float(ball_detection.get("confidence") or 0.0),
        "motion_mode": motion_mode,
        "nearby_player_track_id": nearby_player.get("track_id") if nearby_player is not None else None,
        "nearby_player_bbox_xyxy": nearby_player.get("bbox_xyxy") if nearby_player is not None else None,
    }


def _predictive_ball_search_roi(frame_shape, predictive_state, *, frame_idx=None):
    return BALL_TRACK_SMOOTHER.predictive_search_roi(frame_shape, predictive_state, frame_idx=frame_idx)


def _run_ball_predictive_search(ball_model, frame, predictive_state, *, frame_idx, device, h_matrix=None):
    is_recent = (
        predictive_state is not None
        and int(frame_idx) - int(predictive_state.get("last_seen_frame_idx", frame_idx)) <= BALL_PREDICTIVE_MAX_STALE_FRAMES
        and float(predictive_state.get("confidence") or 0.0) >= BALL_PREDICTIVE_TRIGGER_MIN_CONFIDENCE
    )
    roi_bbox = _predictive_ball_search_roi(frame.shape, predictive_state, frame_idx=frame_idx) if is_recent else None
    summary = {
        "enabled": predictive_state is not None,
        "triggered": roi_bbox is not None,
        "source": "predictive_roi_v1",
        "roi_bbox_xyxy": roi_bbox,
        "candidate_count": 0,
        "motion_mode": predictive_state.get("motion_mode") if predictive_state else None,
        "last_seen_frame_idx": predictive_state.get("last_seen_frame_idx") if predictive_state else None,
        "nearby_player_track_id": predictive_state.get("nearby_player_track_id") if predictive_state else None,
        "stale_frames": (
            max(0, int(frame_idx) - int(predictive_state.get("last_seen_frame_idx", frame_idx)))
            if predictive_state else None
        ),
    }
    if roi_bbox is None:
        return [], summary
    x1, y1, x2, y2 = roi_bbox
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
        summary["skipped_reason"] = "degenerate_predictive_roi"
        return [], summary
    ball_results = ball_model.predict(
        crop,
        classes=[BALL_CLASS_ID],
        conf=0.02,
        device=device,
        verbose=False,
    )
    detections = []
    if ball_results and ball_results[0].boxes is not None and len(ball_results[0].boxes) > 0:
        result = ball_results[0]
        for det_idx in range(len(result.boxes)):
            detection = build_detection(result, det_idx, ball_model.names, crop, h_matrix=None)
            bbox_xyxy = detection.get("bbox_xyxy") or [0.0, 0.0, 0.0, 0.0]
            bbox_xywh = detection.get("bbox_xywh") or [0.0, 0.0, 0.0, 0.0]
            detection["bbox_xyxy"] = [
                float(bbox_xyxy[0] + x1),
                float(bbox_xyxy[1] + y1),
                float(bbox_xyxy[2] + x1),
                float(bbox_xyxy[3] + y1),
            ]
            detection["bbox_xywh"] = [
                float(bbox_xywh[0] + x1),
                float(bbox_xywh[1] + y1),
                float(bbox_xywh[2]),
                float(bbox_xywh[3]),
            ]
            detection["ball_detection_source"] = "predictive_roi_v1"
            detection["ball_motion_mode"] = predictive_state.get("motion_mode")
            if h_matrix is not None:
                center_x, center_y = detection["bbox_xywh"][0], detection["bbox_xywh"][1]
                court_xy = project_pixel_to_court(center_x, center_y, h_matrix)
                detection["court_xy"] = [float(court_xy[0]), float(court_xy[1])]
            detections.append(detection)
    summary["candidate_count"] = len(detections)
    return detections, summary


def _sam_runtime_status_for_exception(exc):
    text = str(exc).lower()
    if "gated repo" in text or "restricted" in text or "401" in text:
        return "gated_repo"
    if "cuda" in text and ("unavailable" in text or "capable" in text):
        return "cuda_unavailable"
    if "couldn't connect" in text or "connection" in text or "offline" in text or "timed out" in text:
        return "weights_unavailable"
    if "not installed" in text:
        return "unavailable"
    if "out of memory" in text or "cuda out of memory" in text:
        return "oom"
    return "load_failed"


def _build_ball_sam_detection(candidate, *, h_matrix, source, model_name, prompt, trigger_reason):
    detection = {
        "track_id": None,
        "class_id": BALL_CLASS_ID,
        "class_name": "sports_ball",
        "confidence": round(float(candidate.score), 4),
        "bbox_xyxy": [float(v) for v in candidate.bbox_xyxy],
        "bbox_xywh": [float(v) for v in candidate.bbox_xywh],
        "center_xy": [round(float(v), 3) for v in candidate.center_xy],
        "ball_detection_source": source,
        "sam3_ball_detection": {
            "backend": "sam3_repo_v1",
            "model_name": model_name,
            "text_prompt": prompt,
            "trigger_reason": trigger_reason,
            "score": round(float(candidate.score), 4),
            "mask_area_px": int(candidate.mask_area_px),
            "mask_area_ratio": round(float(candidate.mask_area_ratio), 6),
        },
    }
    if h_matrix is not None:
        court_xy = project_pixel_to_court(detection["center_xy"][0], detection["center_xy"][1], h_matrix)
        detection["court_xy"] = [float(court_xy[0]), float(court_xy[1])]
    return detection


def _online_ball_discontinuity(previous_signature, previous_track_ids, current_signature, current_track_ids, detection_count_delta):
    if previous_signature is None:
        return {
            "triggered": False,
            "visual_delta": 0.0,
            "track_churn": 0.0,
            "detection_count_delta": int(detection_count_delta),
            "score": 0.0,
        }
    visual_delta = _frame_visual_delta(previous_signature, current_signature)
    prev_ids = previous_track_ids or set()
    curr_ids = current_track_ids or set()
    union_ids = prev_ids | curr_ids
    track_overlap = (len(prev_ids & curr_ids) / len(union_ids)) if union_ids else 1.0
    track_churn = 1.0 - track_overlap
    detection_count_score = _clip01(float(detection_count_delta) / 6.0)
    visual_score = _clip01(visual_delta / DISCONTINUITY_VISUAL_DELTA_THRESHOLD)
    churn_score = _clip01(track_churn / DISCONTINUITY_TRACK_CHURN_THRESHOLD)
    discontinuity_score = 0.60 * visual_score + 0.30 * churn_score + 0.10 * detection_count_score
    triggered = bool(
        discontinuity_score >= DISCONTINUITY_SCORE_THRESHOLD
        or (visual_delta >= DISCONTINUITY_VISUAL_DELTA_THRESHOLD and track_churn >= 0.55)
    )
    return {
        "triggered": triggered,
        "visual_delta": round(float(visual_delta), 4),
        "track_churn": round(float(track_churn), 4),
        "detection_count_delta": int(detection_count_delta),
        "score": round(float(discontinuity_score), 4),
    }


def _ball_sam_trigger_reason(best_ball_detection, predictive_state, missing_streak, online_discontinuity):
    if not _ball_detection_needs_search(best_ball_detection):
        return None
    if online_discontinuity.get("triggered"):
        return "discontinuity_reset"
    if predictive_state is None:
        return "initial_bootstrap"
    if int(missing_streak) >= BALL_SAM_REACQUIRE_MISSING_FRAMES:
        return "sustained_missing"
    return None


def _run_sam_ball_search(sam_ball_detector, frame, *, h_matrix, trigger_reason):
    summary = {
        "enabled": sam_ball_detector is not None,
        "status": "disabled" if sam_ball_detector is None else "ready",
        "triggered": trigger_reason is not None and sam_ball_detector is not None,
        "trigger_reason": trigger_reason,
        "candidate_count": 0,
        "accepted": False,
        "accepted_score": None,
        "source": None,
        "model_name": getattr(sam_ball_detector, "model_name", None),
        "text_prompt": getattr(sam_ball_detector, "text_prompt", None),
    }
    if sam_ball_detector is None or trigger_reason is None:
        return [], summary, summary["status"]
    try:
        candidates = sam_ball_detector.detect(frame)
    except Exception as exc:
        summary["status"] = _sam_runtime_status_for_exception(exc)
        return [], summary, summary["status"]
    accepted = []
    source = "sam3_initial_bootstrap_v1" if trigger_reason == "initial_bootstrap" else "sam3_reacquire_v1"
    for candidate in candidates:
        if float(candidate.score) < BALL_SAM_MIN_SCORE:
            continue
        accepted.append(
            _build_ball_sam_detection(
                candidate,
                h_matrix=h_matrix,
                source=source,
                model_name=sam_ball_detector.model_name,
                prompt=sam_ball_detector.text_prompt,
                trigger_reason=trigger_reason,
            )
        )
    summary["candidate_count"] = len(accepted)
    if accepted:
        summary["accepted"] = True
        summary["accepted_score"] = round(float(accepted[0].get("confidence") or 0.0), 4)
        summary["source"] = source
    return accepted, summary, summary["status"]


def _detection_sort_key_for_ball_roi(detection):
    reasons = detection.get("active_player_reasons") or {}
    return (
        float(detection.get("active_player_score") or 0.0),
        float(detection.get("on_court_score") or 0.0),
        float(detection.get("motion_speed_px") or 0.0),
        -float(reasons.get("edge_penalty") or 0.0),
        float(detection.get("confidence") or 0.0),
    )


def _ball_roi_detection_is_candidate(detection):
    court_foot_xy = detection.get("court_foot_xy") or detection.get("court_xy")
    if court_foot_xy is None or len(court_foot_xy) != 2:
        return True
    dx = _distance_outside_range(float(court_foot_xy[0]), COURT_X_RANGE)
    dy = _distance_outside_range(float(court_foot_xy[1]), COURT_Y_RANGE)
    distance_outside = float(np.linalg.norm([dx, dy]))
    return distance_outside <= BALL_ROI_MAX_COURT_DISTANCE_PX


def _player_ball_search_rois(frame_shape, person_detections):
    height, width = frame_shape[:2]
    boxes = []
    candidates = [detection for detection in person_detections if _ball_roi_detection_is_candidate(detection)]
    ranked = sorted(candidates, key=_detection_sort_key_for_ball_roi, reverse=True)
    for detection in ranked[:BALL_ROI_MAX_PLAYER_WINDOWS]:
        bbox = detection.get("bbox_xyxy")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in bbox]
        box_w = max(1.0, x2 - x1)
        box_h = max(1.0, y2 - y1)
        rx1 = max(0.0, x1 - box_w * BALL_ROI_PAD_X_RATIO)
        rx2 = min(float(width), x2 + box_w * BALL_ROI_PAD_X_RATIO)
        ry1 = max(0.0, y1 - box_h * BALL_ROI_PAD_Y_UP_RATIO)
        ry2 = min(float(height), y2 + box_h * BALL_ROI_PAD_Y_DOWN_RATIO)
        boxes.append([int(round(rx1)), int(round(ry1)), int(round(rx2)), int(round(ry2))])
    if not boxes:
        return []
    return boxes


def _bbox_center_xyxy(bbox_xyxy):
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def _bbox_iou_xyxy(box_a, box_b):
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = max(0.0, (ax2 - ax1) * (ay2 - ay1)) + max(0.0, (bx2 - bx1) * (by2 - by1)) - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _scene_prior_for_point(scene_prior, x, y, *, frame_width, frame_height):
    if not scene_prior or frame_width <= 0 or frame_height <= 0:
        return 0.0

    mask_score = 0.0
    mask_grid = scene_prior.get("region_mask_grid") or []
    if isinstance(mask_grid, list) and mask_grid:
        rows = len(mask_grid)
        cols = max((len(row) for row in mask_grid if isinstance(row, list)), default=0)
        if rows > 0 and cols > 0:
            x_clamped = min(max(float(x), 0.0), max(float(frame_width) - 1.0, 0.0))
            y_clamped = min(max(float(y), 0.0), max(float(frame_height) - 1.0, 0.0))
            col_idx = min(cols - 1, max(0, int((x_clamped / max(float(frame_width), 1.0)) * cols)))
            row_idx = min(rows - 1, max(0, int((y_clamped / max(float(frame_height), 1.0)) * rows)))
            row = mask_grid[row_idx] if row_idx < len(mask_grid) else []
            if isinstance(row, list) and col_idx < len(row):
                mask_score = _clip01(float(row[col_idx]))

    proposal_score = 0.0
    for proposal in scene_prior.get("proposal_regions") or []:
        bbox_xyxy = proposal.get("bbox_xyxy")
        if not bbox_xyxy or len(bbox_xyxy) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
        if x1 <= float(x) <= x2 and y1 <= float(y) <= y2:
            proposal_score = max(
                proposal_score,
                _clip01(float(proposal.get("confidence") or proposal.get("area_ratio") or 1.0)),
            )

    return _clip01(max(mask_score, proposal_score))


def _distance_outside_range(value, allowed_range):
    low, high = float(allowed_range[0]), float(allowed_range[1])
    value = float(value)
    if value < low:
        return low - value
    if value > high:
        return value - high
    return 0.0


def _compute_detection_target_court_support(
    detection,
    *,
    frame_width,
    frame_height,
    bootstrap_context=None,
    scene_prior=None,
):
    bbox_xyxy = detection.get("bbox_xyxy") or [0.0, 0.0, 0.0, 0.0]
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    bbox_cx = (x1 + x2) * 0.5
    bbox_cy = (y1 + y2) * 0.5
    footpoint = _estimate_detection_footpoint(detection)
    footpoint_xy = footpoint["xy"]

    center_foreground_prior = 0.0
    foot_foreground_prior = 0.0
    if bootstrap_context and bootstrap_context.get("enabled"):
        center_foreground_prior = foreground_prior_for_point(
            bootstrap_context,
            bbox_cx,
            bbox_cy,
        )
        foot_foreground_prior = foreground_prior_for_point(
            bootstrap_context,
            float(footpoint_xy[0]),
            float(footpoint_xy[1]),
        )

    scene_center_prior = _scene_prior_for_point(
        scene_prior,
        bbox_cx,
        bbox_cy,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    scene_foot_prior = _scene_prior_for_point(
        scene_prior,
        float(footpoint_xy[0]),
        float(footpoint_xy[1]),
        frame_width=frame_width,
        frame_height=frame_height,
    )
    effective_center_prior = max(float(center_foreground_prior), float(scene_center_prior))
    effective_foot_prior = max(float(foot_foreground_prior), float(scene_foot_prior))

    court_ground_xy = detection.get("court_foot_xy") or detection.get("court_xy")
    court_in_bounds = None
    court_distance_outside = None
    court_boundary_score = 0.0
    grounding_score = 0.20
    if court_ground_xy is not None and len(court_ground_xy) == 2:
        cx, cy = float(court_ground_xy[0]), float(court_ground_xy[1])
        dx = _distance_outside_range(cx, COURT_X_RANGE)
        dy = _distance_outside_range(cy, COURT_Y_RANGE)
        court_distance_outside = float(np.linalg.norm([dx, dy]))
        court_in_bounds = bool(dx == 0.0 and dy == 0.0)
        court_boundary_score = 1.0 if court_in_bounds else _clip01(
            1.0 - (court_distance_outside / ENGAGEMENT_ADJACENT_COURT_MARGIN)
        )
        grounding_score = court_boundary_score

    target_court_score = _clip01(
        max(
            float(court_boundary_score),
            float(effective_foot_prior),
            float(effective_center_prior) * 0.7,
        )
    )
    return {
        "footpoint_xy": [float(footpoint_xy[0]), float(footpoint_xy[1])],
        "footpoint_source": footpoint["source"],
        "footpoint_confidence": float(footpoint["confidence"]),
        "bootstrap_foreground_prior": float(center_foreground_prior),
        "bootstrap_foot_prior": float(foot_foreground_prior),
        "scene_center_prior": float(scene_center_prior),
        "scene_foot_prior": float(scene_foot_prior),
        "effective_foreground_prior": float(effective_center_prior),
        "effective_foot_prior": float(effective_foot_prior),
        "court_ground_xy": (
            [float(court_ground_xy[0]), float(court_ground_xy[1])]
            if court_ground_xy is not None and len(court_ground_xy) == 2
            else None
        ),
        "court_in_bounds": court_in_bounds,
        "court_distance_outside": court_distance_outside,
        "court_boundary_score": float(court_boundary_score),
        "grounding_score": float(grounding_score),
        "target_court_score": float(target_court_score),
    }


def _dedupe_ball_detections(detections):
    deduped = []
    for detection in sorted(detections, key=lambda item: float(item.get("confidence") or 0.0), reverse=True):
        keep = True
        det_bbox = detection.get("bbox_xyxy") or [0.0, 0.0, 0.0, 0.0]
        det_center = _bbox_center_xyxy(det_bbox)
        for kept in deduped:
            kept_bbox = kept.get("bbox_xyxy") or [0.0, 0.0, 0.0, 0.0]
            kept_center = _bbox_center_xyxy(kept_bbox)
            iou = _bbox_iou_xyxy(det_bbox, kept_bbox)
            center_distance = float(np.linalg.norm(np.array(det_center) - np.array(kept_center)))
            if iou >= BALL_ROI_DEDUPE_IOU or center_distance <= BALL_ROI_DEDUPE_CENTER_DISTANCE_PX:
                keep = False
                break
        if keep:
            deduped.append(detection)
    return deduped


def _run_ball_roi_fallback(ball_model, frame, person_detections, *, device, h_matrix=None):
    roi_bboxes = _player_ball_search_rois(frame.shape, person_detections)
    summary = {
        "enabled": bool(roi_bboxes),
        "triggered": False,
        "source": "player_local_rois_v2",
        "roi_bbox_xyxy": roi_bboxes[0] if roi_bboxes else None,
        "roi_bboxes_xyxy": roi_bboxes,
        "candidate_count": 0,
    }
    if not roi_bboxes:
        return [], summary
    detections = []
    summary["triggered"] = True
    crop_records = []
    for roi_index, roi_bbox in enumerate(roi_bboxes):
        x1, y1, x2, y2 = roi_bbox
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crop_records.append((roi_index, x1, y1, crop))
    if not crop_records:
        return [], summary
    ball_results = ball_model.predict(
        [record[3] for record in crop_records],
        classes=[BALL_CLASS_ID],
        conf=0.02,
        device=device,
        verbose=False,
    )
    for (roi_index, x1, y1, crop), result in zip(crop_records, ball_results):
        if result.boxes is None or len(result.boxes) <= 0:
            continue
        for det_idx in range(len(result.boxes)):
            detection = build_detection(result, det_idx, ball_model.names, crop, h_matrix=None)
            bbox_xyxy = detection.get("bbox_xyxy") or [0.0, 0.0, 0.0, 0.0]
            bbox_xywh = detection.get("bbox_xywh") or [0.0, 0.0, 0.0, 0.0]
            detection["bbox_xyxy"] = [
                float(bbox_xyxy[0] + x1),
                float(bbox_xyxy[1] + y1),
                float(bbox_xyxy[2] + x1),
                float(bbox_xyxy[3] + y1),
            ]
            detection["bbox_xywh"] = [
                float(bbox_xywh[0] + x1),
                float(bbox_xywh[1] + y1),
                float(bbox_xywh[2]),
                float(bbox_xywh[3]),
            ]
            detection["ball_detection_source"] = "player_local_rois_v2"
            detection["ball_detection_roi_index"] = roi_index
            if h_matrix is not None:
                center_x, center_y = detection["bbox_xywh"][0], detection["bbox_xywh"][1]
                court_xy = project_pixel_to_court(center_x, center_y, h_matrix)
                detection["court_xy"] = [float(court_xy[0]), float(court_xy[1])]
            detections.append(detection)
    detections = _dedupe_ball_detections(detections)
    summary["candidate_count"] = len(detections)
    return detections, summary


def _ball_detection_as_candidate(ball_detection):
    if ball_detection is None:
        return None
    candidate = {
        "track_id": ball_detection.get("track_id"),
        "class_id": ball_detection.get("class_id", BALL_CLASS_ID),
        "class_name": ball_detection.get("class_name") or "sports_ball",
        "confidence": float(ball_detection.get("confidence") or 0.0),
        "bbox_xyxy": [float(v) for v in (ball_detection.get("bbox_xyxy") or [0.0, 0.0, 0.0, 0.0])],
        "bbox_xywh": [float(v) for v in (ball_detection.get("bbox_xywh") or [0.0, 0.0, 0.0, 0.0])],
        "ball_detection_source": ball_detection.get("source") or ball_detection.get("ball_detection_source") or "unknown",
    }
    center_xy = ball_detection.get("center_xy")
    if center_xy is not None:
        candidate["center_xy"] = [float(v) for v in center_xy]
    court_xy = ball_detection.get("court_xy")
    if court_xy is not None:
        candidate["court_xy"] = [float(v) for v in court_xy]
    if ball_detection.get("retro_backfilled"):
        candidate["retro_backfilled"] = True
        candidate["retro_anchor_frame_idx"] = ball_detection.get("retro_anchor_frame_idx")
        candidate["retro_link_frame_idx"] = ball_detection.get("retro_link_frame_idx")
    return candidate


def _retro_ball_search_roi(frame_shape, anchor_detection, next_detection=None):
    if anchor_detection is None:
        return None
    center_xy = anchor_detection.get("center_xy") or []
    if len(center_xy) != 2:
        return None
    anchor_center = np.array(center_xy, dtype=np.float32)
    if next_detection is not None and next_detection.get("center_xy") is not None:
        next_center = np.array(next_detection.get("center_xy"), dtype=np.float32)
        velocity_xy = next_center - anchor_center
        predicted_center = anchor_center - velocity_xy
        radius = BALL_RETRO_BASE_RADIUS_PX + 0.45 * float(np.linalg.norm(velocity_xy))
    else:
        predicted_center = anchor_center
        radius = BALL_RETRO_BASE_RADIUS_PX
    roi_bbox = [
        float(predicted_center[0] - radius),
        float(predicted_center[1] - radius),
        float(predicted_center[0] + radius),
        float(predicted_center[1] + radius),
    ]
    return _clip_roi_xyxy(roi_bbox, frame_shape)


def _run_ball_retro_search(ball_model, frame, anchor_detection, next_detection, *, frame_idx, device, h_matrix=None):
    roi_bbox = _retro_ball_search_roi(frame.shape, anchor_detection, next_detection)
    summary = {
        "enabled": anchor_detection is not None,
        "triggered": roi_bbox is not None,
        "source": "retro_search_v1",
        "roi_bbox_xyxy": roi_bbox,
        "candidate_count": 0,
        "anchor_frame_idx": anchor_detection.get("frame_idx") if anchor_detection else None,
        "next_frame_idx": next_detection.get("frame_idx") if next_detection else None,
    }
    if roi_bbox is None:
        return [], summary
    x1, y1, x2, y2 = roi_bbox
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return [], summary
    ball_results = ball_model.predict(
        crop,
        classes=[BALL_CLASS_ID],
        conf=0.02,
        device=device,
        verbose=False,
    )
    detections = []
    if ball_results and ball_results[0].boxes is not None and len(ball_results[0].boxes) > 0:
        result = ball_results[0]
        for det_idx in range(len(result.boxes)):
            detection = build_detection(result, det_idx, ball_model.names, crop, h_matrix=None)
            bbox_xyxy = detection.get("bbox_xyxy") or [0.0, 0.0, 0.0, 0.0]
            bbox_xywh = detection.get("bbox_xywh") or [0.0, 0.0, 0.0, 0.0]
            detection["bbox_xyxy"] = [
                float(bbox_xyxy[0] + x1),
                float(bbox_xyxy[1] + y1),
                float(bbox_xyxy[2] + x1),
                float(bbox_xyxy[3] + y1),
            ]
            detection["bbox_xywh"] = [
                float(bbox_xywh[0] + x1),
                float(bbox_xywh[1] + y1),
                float(bbox_xywh[2]),
                float(bbox_xywh[3]),
            ]
            detection["ball_detection_source"] = "retro_search_v1"
            detection["retro_backfilled"] = True
            detection["retro_anchor_frame_idx"] = anchor_detection.get("frame_idx") if anchor_detection else None
            detection["retro_link_frame_idx"] = next_detection.get("frame_idx") if next_detection else None
            if h_matrix is not None:
                center_x, center_y = detection["bbox_xywh"][0], detection["bbox_xywh"][1]
                court_xy = project_pixel_to_court(center_x, center_y, h_matrix)
                detection["court_xy"] = [float(court_xy[0]), float(court_xy[1])]
            detections.append(detection)
    detections = _dedupe_ball_detections(detections)
    summary["candidate_count"] = len(detections)
    return detections, summary


def retrofit_ball_detections_before_first_observation(frames, *, ball_model, device):
    if not frames:
        return {"enabled": False, "anchor_frame_idx": None, "backfilled_frame_count": 0, "stop_reason": "empty"}
    anchor_idx = None
    for idx, frame in enumerate(frames):
        if not _ball_detection_needs_search(frame.get("ball_detection"), min_confidence=BALL_RETRO_ANCHOR_MIN_CONFIDENCE):
            anchor_idx = idx
            break
    default_summary = {
        "enabled": anchor_idx is not None and anchor_idx > 0,
        "triggered": False,
        "anchor_frame_idx": anchor_idx,
        "roi_bbox_xyxy": None,
        "candidate_count": 0,
        "accepted": False,
        "accepted_source": None,
        "stop_reason": None,
    }
    for frame in frames:
        frame.setdefault("ball_retro_search", dict(default_summary))
    if anchor_idx is None:
        return {"enabled": False, "anchor_frame_idx": None, "backfilled_frame_count": 0, "stop_reason": "no_anchor"}
    if anchor_idx == 0:
        return {"enabled": False, "anchor_frame_idx": 0, "backfilled_frame_count": 0, "stop_reason": "anchor_is_first_frame"}

    anchor_segment_id = frames[anchor_idx].get("continuity_segment_id")
    known_after = _ball_detection_as_candidate(frames[anchor_idx].get("ball_detection"))
    if known_after is None:
        return {"enabled": False, "anchor_frame_idx": anchor_idx, "backfilled_frame_count": 0, "stop_reason": "missing_anchor_detection"}
    known_after["frame_idx"] = int(frames[anchor_idx].get("frame_idx", anchor_idx))
    next_detection = None
    backfilled_frame_count = 0
    miss_streak = 0
    stop_reason = "segment_start"

    for idx in range(anchor_idx - 1, -1, -1):
        frame = frames[idx]
        frame_summary = frame["ball_retro_search"]
        frame_summary.update({
            "enabled": True,
            "anchor_frame_idx": int(frames[anchor_idx].get("frame_idx", anchor_idx)),
        })
        if frame.get("continuity_segment_id") != anchor_segment_id:
            stop_reason = "segment_boundary"
            break

        existing_ball = frame.get("ball_detection")
        if not _ball_detection_needs_search(existing_ball, min_confidence=BALL_RETRO_ANCHOR_MIN_CONFIDENCE):
            current = _ball_detection_as_candidate(existing_ball)
            current["frame_idx"] = int(frame.get("frame_idx", idx))
            next_detection = known_after
            known_after = current
            miss_streak = 0
            frame_summary.update({
                "accepted": True,
                "accepted_source": current.get("ball_detection_source"),
                "stop_reason": None,
            })
            continue

        h_matrix = _h_matrix_as_array(frame.get("_h_matrix"))
        retro_detections, retro_summary = _run_ball_retro_search(
            ball_model,
            frame.get("_frame_bgr"),
            known_after,
            next_detection,
            frame_idx=int(frame.get("frame_idx", idx)),
            device=device,
            h_matrix=h_matrix,
        )
        fallback_detections, fallback_summary = _run_ball_roi_fallback(
            ball_model,
            frame.get("_frame_bgr"),
            frame.get("detections") or [],
            device=device,
            h_matrix=h_matrix,
        )
        combined = []
        for candidate in frame.get("raw_ball_detections") or []:
            built = _ball_detection_as_candidate(candidate)
            if built is not None:
                combined.append(built)
        combined.extend(retro_detections)
        combined.extend(fallback_detections)
        combined = _dedupe_ball_detections(combined)
        frame["raw_ball_detections"] = _serialize_ball_candidates(combined)
        selected = max(combined, key=lambda item: float(item.get("confidence") or 0.0), default=None)
        accepted = False
        if selected is not None:
            best = _extract_best_ball_detection([selected])
            if not _ball_detection_needs_search(best, min_confidence=BALL_RETRO_ANCHOR_MIN_CONFIDENCE):
                best["retro_backfilled"] = True
                best["retro_anchor_frame_idx"] = int(frames[anchor_idx].get("frame_idx", anchor_idx))
                best["retro_link_frame_idx"] = known_after.get("frame_idx")
                frame["ball_detection"] = best
                current = _ball_detection_as_candidate(best)
                current["frame_idx"] = int(frame.get("frame_idx", idx))
                next_detection = known_after
                known_after = current
                backfilled_frame_count += 1
                miss_streak = 0
                accepted = True
        if not accepted:
            miss_streak += 1
            if miss_streak >= BALL_RETRO_MAX_MISS_STREAK:
                stop_reason = "retro_miss_streak"
                frame_summary.update({
                    "triggered": bool(retro_summary.get("triggered") or fallback_summary.get("triggered")),
                    "roi_bbox_xyxy": retro_summary.get("roi_bbox_xyxy"),
                    "candidate_count": int(len(combined)),
                    "accepted": False,
                    "accepted_source": None,
                    "stop_reason": stop_reason,
                })
                break
        frame_summary.update({
            "triggered": bool(retro_summary.get("triggered") or fallback_summary.get("triggered")),
            "roi_bbox_xyxy": retro_summary.get("roi_bbox_xyxy"),
            "candidate_count": int(len(combined)),
            "accepted": bool(accepted),
            "accepted_source": (frame.get("ball_detection") or {}).get("source") if accepted else None,
            "stop_reason": None,
        })

    return {
        "enabled": True,
        "anchor_frame_idx": int(frames[anchor_idx].get("frame_idx", anchor_idx)),
        "backfilled_frame_count": int(backfilled_frame_count),
        "stop_reason": stop_reason,
    }


def _ball_size_score(detection):
    bbox_xywh = detection.get("bbox_xywh") or []
    if len(bbox_xywh) < 4:
        return 0.0
    diameter = max(float(bbox_xywh[2]), float(bbox_xywh[3]))
    if diameter < BALL_STATE_MIN_SIZE_PX or diameter > BALL_STATE_MAX_SIZE_PX:
        return 0.0
    midpoint = 0.5 * (BALL_STATE_MIN_SIZE_PX + BALL_STATE_MAX_SIZE_PX)
    spread = max(1.0, midpoint - BALL_STATE_MIN_SIZE_PX)
    return max(0.0, 1.0 - abs(diameter - midpoint) / spread)


def _ball_continuity_score(detection, previous_center_xy):
    if previous_center_xy is None:
        return 0.5
    center_xy = detection.get("center_xy")
    if not center_xy or len(center_xy) != 2:
        return 0.0
    distance = float(np.linalg.norm(np.array(center_xy, dtype=np.float32) - np.array(previous_center_xy, dtype=np.float32)))
    if distance >= BALL_STATE_MAX_JUMP_PX:
        return 0.0
    return max(0.0, 1.0 - distance / BALL_STATE_MAX_JUMP_PX)


def _target_court_prior_for_ball(center_xy, target_court_prior):
    bbox = (target_court_prior or {}).get("bbox_xyxy")
    if not bbox or len(bbox) != 4 or not center_xy or len(center_xy) != 2:
        return {
            "target_court_prior": 0.0,
            "target_court_relation": "missing_prior",
            "target_court_distance_px": None,
        }
    x, y = [float(v) for v in center_xy]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    margin = 0.08 * max(width, height)
    if x1 <= x <= x2 and y1 <= y <= y2:
        return {
            "target_court_prior": 1.0,
            "target_court_relation": "inside_target_court_region",
            "target_court_distance_px": 0.0,
        }
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    distance = float(np.linalg.norm([dx, dy]))
    if x1 <= x <= x2 and y < y1 and (y1 - y) <= 0.45 * height:
        # Airborne shots/passes often project above the floor-region box.
        prior = max(0.25, 0.65 * (1.0 - min(1.0, (y1 - y) / max(0.45 * height, 1.0))))
        relation = "above_target_court_airspace"
    elif distance <= margin:
        prior = max(0.0, 0.45 * (1.0 - distance / max(margin, 1.0)))
        relation = "near_target_court_region"
    else:
        prior = 0.0
        relation = "outside_target_court_region"
    return {
        "target_court_prior": round(float(prior), 4),
        "target_court_relation": relation,
        "target_court_distance_px": round(float(distance), 3),
    }


def _score_ball_candidate(detection, previous_center_xy, target_court_prior=None):
    confidence = float(detection.get("confidence") or 0.0)
    continuity = _ball_continuity_score(detection, previous_center_xy)
    size_score = _ball_size_score(detection)
    center_xy = detection.get("center_xy") or [0.0, 0.0]
    target_prior = _target_court_prior_for_ball(center_xy, target_court_prior)
    score = 0.55 * confidence + 0.22 * continuity + 0.08 * size_score + 0.15 * float(target_prior["target_court_prior"])
    return {
        "detection": detection,
        "confidence": confidence,
        "continuity_score": continuity,
        "size_score": size_score,
        **target_prior,
        "jump_guard_applied": False,
        "hard_rejection_reason": None,
        "score": score,
    }


def _apply_ball_candidate_hard_guards(candidate, previous_center_xy):
    confidence = float(candidate.get("confidence") or 0.0)
    continuity = float(candidate.get("continuity_score") or 0.0)
    foreground_prior = float(candidate.get("foreground_prior") or 0.0)
    target_relation = candidate.get("target_court_relation")
    if previous_center_xy is None or continuity > 0.0:
        return candidate

    rejection_reason = None
    if confidence < 0.60:
        rejection_reason = "low_confidence_large_jump"
    elif confidence < 0.90:
        rejection_reason = "large_jump_below_high_confidence"
    elif (
        target_relation == "outside_target_court_region"
        and foreground_prior <= 0.05
        and confidence < 0.98
    ):
        rejection_reason = "outside_target_court_large_jump"

    if rejection_reason:
        candidate["jump_guard_applied"] = True
        candidate["hard_rejection_reason"] = rejection_reason
        candidate["score"] = min(float(candidate.get("score") or 0.0), BALL_STATE_MIN_SCORE - 0.001)
    return candidate


def _ball_state_motion_mode_from_velocity(velocity_xy, detection=None):
    return BALL_TRACK_SMOOTHER.state_motion_mode_from_velocity(velocity_xy, detection)


def annotate_ball_state(frames):
    previous_center_xy = None
    previous_velocity_xy = np.array([0.0, 0.0], dtype=np.float32)
    previous_confidence = 0.0
    previous_motion_mode = "unknown_recent"
    missing_gap_frames = 0

    for frame in frames:
        raw_ball_detections = list(frame.get("raw_ball_detections") or [])
        if not raw_ball_detections and frame.get("ball_detection") is not None:
            raw_ball_detections = [frame["ball_detection"]]
        bootstrap_context = frame.get("bootstrap_context")
        target_court_prior = frame.get("target_court_prior")
        candidates = []
        for raw_ball_detection in raw_ball_detections:
            candidate = _score_ball_candidate(raw_ball_detection, previous_center_xy, target_court_prior=target_court_prior)
            foreground_prior = 0.0
            if bootstrap_context and bootstrap_context.get("enabled"):
                center_xy = raw_ball_detection.get("center_xy") or [0.0, 0.0]
                foreground_prior = foreground_prior_for_point(
                    bootstrap_context,
                    center_xy[0],
                    center_xy[1],
                )
                candidate["score"] += 0.10 * foreground_prior
            candidate["foreground_prior"] = foreground_prior
            _apply_ball_candidate_hard_guards(candidate, previous_center_xy)
            candidates.append(candidate)

        candidates.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
        selected = candidates[0] if candidates else None
        ball_state = None
        if selected is not None and selected["score"] >= BALL_STATE_MIN_SCORE:
            detection = selected["detection"]
            center_xy = [round(float(v), 3) for v in (detection.get("center_xy") or [0.0, 0.0])]
            if previous_center_xy is None:
                velocity_xy = [0.0, 0.0]
            else:
                velocity_xy = [
                    round(float(center_xy[0] - previous_center_xy[0]), 3),
                    round(float(center_xy[1] - previous_center_xy[1]), 3),
                ]
            previous_center_xy = np.array(center_xy, dtype=np.float32)
            previous_velocity_xy = np.array(velocity_xy, dtype=np.float32)
            previous_confidence = float(detection.get("confidence") or 0.0)
            previous_motion_mode = _ball_state_motion_mode_from_velocity(velocity_xy, detection)
            missing_gap_frames = 0
            ball_state = {
                "state": "observed",
                "confidence": round(previous_confidence, 4),
                "center_xy": center_xy,
                "bbox_xyxy": detection.get("bbox_xyxy"),
                "bbox_xywh": detection.get("bbox_xywh"),
                "court_xy": detection.get("court_xy"),
                "velocity_xy": velocity_xy,
                "speed_px": round(_vector_norm(velocity_xy), 3),
                "motion_mode": previous_motion_mode,
                "missing_gap_frames": 0,
                "keep_alive_kind": "observed",
                "source": "detector",
                "candidate_count": len(candidates),
                "candidate_scores": [
                    {
                        "score": round(float(candidate["score"]), 4),
                        "confidence": round(float(candidate["confidence"]), 4),
                        "foreground_prior": round(float(candidate.get("foreground_prior") or 0.0), 4),
                        "target_court_prior": round(float(candidate.get("target_court_prior") or 0.0), 4),
                        "target_court_relation": candidate.get("target_court_relation"),
                        "target_court_distance_px": candidate.get("target_court_distance_px"),
                        "jump_guard_applied": bool(candidate.get("jump_guard_applied")),
                        "hard_rejection_reason": candidate.get("hard_rejection_reason"),
                    }
                    for candidate in candidates[:3]
                ],
            }
        elif previous_center_xy is not None and missing_gap_frames < BALL_STATE_MAX_GAP_FRAMES:
            missing_gap_frames += 1
            predicted_center = previous_center_xy + previous_velocity_xy
            previous_center_xy = predicted_center
            predicted_confidence = BALL_TRACK_SMOOTHER.predicted_confidence(previous_confidence, missing_gap_frames)
            keep_alive_kind = BALL_TRACK_SMOOTHER.keep_alive_kind(missing_gap_frames, previous_motion_mode)
            ball_state = {
                "state": "predicted_short_gap",
                "confidence": round(predicted_confidence, 4),
                "center_xy": [round(float(predicted_center[0]), 3), round(float(predicted_center[1]), 3)],
                "bbox_xyxy": None,
                "bbox_xywh": None,
                "court_xy": None,
                "velocity_xy": [round(float(previous_velocity_xy[0]), 3), round(float(previous_velocity_xy[1]), 3)],
                "speed_px": round(_vector_norm(previous_velocity_xy), 3),
                "motion_mode": previous_motion_mode,
                "missing_gap_frames": int(missing_gap_frames),
                "keep_alive_kind": keep_alive_kind,
                "source": "smoothed_prediction",
                "candidate_count": len(candidates),
                "candidate_scores": [
                    {
                        "score": round(float(candidate["score"]), 4),
                        "confidence": round(float(candidate["confidence"]), 4),
                        "foreground_prior": round(float(candidate.get("foreground_prior") or 0.0), 4),
                        "target_court_prior": round(float(candidate.get("target_court_prior") or 0.0), 4),
                        "target_court_relation": candidate.get("target_court_relation"),
                        "target_court_distance_px": candidate.get("target_court_distance_px"),
                        "jump_guard_applied": bool(candidate.get("jump_guard_applied")),
                        "hard_rejection_reason": candidate.get("hard_rejection_reason"),
                    }
                    for candidate in candidates[:3]
                ],
            }
        else:
            missing_gap_frames = max(missing_gap_frames, BALL_STATE_MAX_GAP_FRAMES + 1)
            previous_center_xy = None
            previous_velocity_xy = np.array([0.0, 0.0], dtype=np.float32)
            previous_confidence = 0.0
            previous_motion_mode = "unknown_recent"
            ball_state = {
                "state": "missing",
                "confidence": 0.0,
                "center_xy": None,
                "bbox_xyxy": None,
                "bbox_xywh": None,
                "court_xy": None,
                "velocity_xy": [0.0, 0.0],
                "speed_px": 0.0,
                "motion_mode": previous_motion_mode,
                "missing_gap_frames": int(missing_gap_frames),
                "keep_alive_kind": "expired",
                "source": "smoothed_prediction",
                "candidate_count": len(candidates),
                "candidate_scores": [
                    {
                        "score": round(float(candidate["score"]), 4),
                        "confidence": round(float(candidate["confidence"]), 4),
                        "foreground_prior": round(float(candidate.get("foreground_prior") or 0.0), 4),
                        "target_court_prior": round(float(candidate.get("target_court_prior") or 0.0), 4),
                        "target_court_relation": candidate.get("target_court_relation"),
                        "target_court_distance_px": candidate.get("target_court_distance_px"),
                        "jump_guard_applied": bool(candidate.get("jump_guard_applied")),
                        "hard_rejection_reason": candidate.get("hard_rejection_reason"),
                    }
                    for candidate in candidates[:3]
                ],
            }
        frame["ball_state"] = ball_state
    return {
        "kind": "single_candidate_short_gap_v1",
        "min_score": BALL_STATE_MIN_SCORE,
        "max_jump_px": BALL_STATE_MAX_JUMP_PX,
        "max_gap_frames": BALL_STATE_MAX_GAP_FRAMES,
        "target_court_prior": {
            "enabled": True,
            "weight": 0.15,
            "airspace_policy": "allow_above_target_court_region_with_partial_prior",
        },
    }


def _distance_point_to_bbox(x, y, bbox_xyxy):
    if not bbox_xyxy or len(bbox_xyxy) != 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    nearest_x = min(max(float(x), x1), x2)
    nearest_y = min(max(float(y), y1), y2)
    return float(np.linalg.norm([float(x) - nearest_x, float(y) - nearest_y]))


def _score_detection_ball_affinity(detection, ball_state, *, frame_width, frame_height):
    center_xy = (ball_state or {}).get("center_xy")
    ball_state_name = str((ball_state or {}).get("state") or "missing")
    if center_xy is None or len(center_xy) != 2:
        return {
            "score": 0.0,
            "ball_state": ball_state_name,
            "ball_visible": False,
            "distance_to_bbox_px": None,
            "distance_scale_px": None,
        }
    bbox_xyxy = detection.get("bbox_xyxy") or []
    distance_to_bbox = _distance_point_to_bbox(float(center_xy[0]), float(center_xy[1]), bbox_xyxy)
    _, bbox_h = _bbox_size_xy(detection)
    distance_scale_px = max(
        float(bbox_h) * ENGAGEMENT_BALL_DISTANCE_HEIGHT_FACTOR,
        float(frame_height) * 0.14,
        ENGAGEMENT_BALL_DISTANCE_MIN_PX,
    )
    if distance_to_bbox is None:
        score = 0.0
    elif distance_to_bbox <= 1e-6:
        score = 1.0
    else:
        score = _clip01(1.0 - (float(distance_to_bbox) / max(distance_scale_px, 1e-6)))
    return {
        "score": round(float(score), 4),
        "ball_state": ball_state_name,
        "ball_visible": ball_state_name != "missing",
        "distance_to_bbox_px": round(float(distance_to_bbox), 3) if distance_to_bbox is not None else None,
        "distance_scale_px": round(float(distance_scale_px), 3),
    }


def annotate_player_engagement(frames, video_meta):
    summary = {
        "kind": "target_game_engagement_v1",
        "window_seconds": ENGAGEMENT_WINDOW_SECONDS,
        "frame_count": len(frames),
        "state_counts": {},
        "adjacent_or_other_detection_count": 0,
        "engaged_detection_count": 0,
    }
    if not frames:
        return summary

    fps = float((video_meta or {}).get("fps") or 30.0)
    frame_width = int((video_meta or {}).get("width") or 0)
    frame_height = int((video_meta or {}).get("height") or 0)
    window_frames = max(8, int(round(fps * ENGAGEMENT_WINDOW_SECONDS)))
    summary["window_frames"] = int(window_frames)
    track_counts = _track_frame_counts(frames)
    history = defaultdict(
        lambda: {
            "target": deque(maxlen=window_frames),
            "ball": deque(maxlen=window_frames),
        }
    )

    for frame in frames:
        scene_prior = frame.get("scene_prior") or _build_scene_prior_contract(frame)
        bootstrap_context = frame.get("bootstrap_context")
        ball_state = frame.get("ball_state") or {}
        frame_state_counts = {}
        for detection in frame.get("detections", []):
            if _is_ball_detection(detection):
                continue
            target_support = _compute_detection_target_court_support(
                detection,
                frame_width=frame_width,
                frame_height=frame_height,
                bootstrap_context=bootstrap_context,
                scene_prior=scene_prior,
            )
            ball_affinity = _score_detection_ball_affinity(
                detection,
                ball_state,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            track_id = detection.get("track_id")
            if track_id is not None:
                track_key = int(track_id)
                history[track_key]["target"].append(float(target_support["target_court_score"]))
                history[track_key]["ball"].append(float(ball_affinity["score"]))
                rolling_target = float(sum(history[track_key]["target"]) / len(history[track_key]["target"]))
                rolling_ball_avg = float(sum(history[track_key]["ball"]) / len(history[track_key]["ball"]))
                rolling_ball_peak = float(max(history[track_key]["ball"]))
            else:
                rolling_target = float(target_support["target_court_score"])
                rolling_ball_avg = float(ball_affinity["score"])
                rolling_ball_peak = float(ball_affinity["score"])
            rolling_ball_score = 0.65 * rolling_ball_peak + 0.35 * rolling_ball_avg
            persistence_score = min(track_counts.get(track_id, 1), 5) / 5.0
            motion_score = _clip01(float(detection.get("motion_speed_px") or 0.0) / MOTION_SCORE_THRESHOLD_PX)
            engagement_score = _clip01(
                0.55 * rolling_target
                + 0.20 * rolling_ball_score
                + 0.15 * motion_score
                + 0.10 * persistence_score
            )
            other_game_risk = _clip01(
                0.75 * _clip01((0.42 - rolling_target) / 0.42)
                + 0.15 * _clip01((0.14 - rolling_ball_peak) / 0.14)
                + 0.10 * _clip01((0.25 - float(target_support["effective_foot_prior"])) / 0.25)
            )
            if rolling_target >= 0.60 and engagement_score >= 0.62:
                engagement_state = "target_game_engaged"
            elif rolling_target >= 0.42 and engagement_score >= 0.45:
                engagement_state = "target_court_present"
            elif rolling_target <= 0.25 and rolling_ball_peak <= 0.10:
                engagement_state = "adjacent_or_other"
            else:
                engagement_state = "uncertain"

            detection["target_court_score"] = round(float(rolling_target), 4)
            detection["engagement_score"] = round(float(engagement_score), 4)
            detection["engagement_state"] = engagement_state
            detection["engagement"] = {
                "kind": "target_game_engagement_v1",
                "target_court_score": round(float(rolling_target), 4),
                "target_court_score_immediate": round(float(target_support["target_court_score"]), 4),
                "ball_affinity_score": round(float(rolling_ball_score), 4),
                "ball_affinity_score_immediate": round(float(ball_affinity["score"]), 4),
                "ball_affinity_peak": round(float(rolling_ball_peak), 4),
                "ball_affinity_avg": round(float(rolling_ball_avg), 4),
                "other_game_risk": round(float(other_game_risk), 4),
                "engagement_score": round(float(engagement_score), 4),
                "engagement_state": engagement_state,
                "court_distance_outside": target_support["court_distance_outside"],
                "court_boundary_score": round(float(target_support["court_boundary_score"]), 4),
                "effective_foot_prior": round(float(target_support["effective_foot_prior"]), 4),
                "effective_center_prior": round(float(target_support["effective_foreground_prior"]), 4),
                "ball_state": ball_affinity["ball_state"],
                "ball_visible": bool(ball_affinity["ball_visible"]),
                "ball_distance_to_bbox_px": ball_affinity["distance_to_bbox_px"],
                "ball_distance_scale_px": ball_affinity["distance_scale_px"],
                "window_frames": int(window_frames),
            }
            frame_state_counts[engagement_state] = frame_state_counts.get(engagement_state, 0) + 1
            summary["state_counts"][engagement_state] = summary["state_counts"].get(engagement_state, 0) + 1
            if engagement_state == "adjacent_or_other":
                summary["adjacent_or_other_detection_count"] += 1
            if engagement_state == "target_game_engaged":
                summary["engaged_detection_count"] += 1
        frame["engagement_summary"] = {
            "state_counts": frame_state_counts,
            "ball_state": (ball_state or {}).get("state"),
        }
    return summary


def _tracklet_evidence_key(detection):
    identity_track_id = detection.get("identity_track_id")
    if identity_track_id is not None:
        return ("identity_track_id", int(identity_track_id))
    track_id = detection.get("track_id")
    if track_id is not None:
        return ("track_id", int(track_id))
    return None



def annotate_tracklet_temporal_evidence(frames, video_meta):
    summary = {
        "kind": "tracklet_temporal_evidence_v1",
        "frame_count": len(frames),
        "track_count": 0,
        "window_seconds": TRACKLET_TEMPORAL_WINDOW_SECONDS,
        "window_frames": 0,
        "state_counts": {},
        "promotion_ready_detection_count": 0,
        "demotion_ready_detection_count": 0,
    }
    if not frames:
        return summary

    fps = float((video_meta or {}).get("fps") or 30.0)
    window_frames = max(6, int(round(fps * TRACKLET_TEMPORAL_WINDOW_SECONDS)))
    summary["window_frames"] = int(window_frames)
    track_counts = _track_frame_counts(frames, key_fn=_tracklet_evidence_key)
    summary["track_count"] = len(track_counts)
    history = defaultdict(
        lambda: {
            "engagement": deque(maxlen=window_frames),
            "target": deque(maxlen=window_frames),
            "risk": deque(maxlen=window_frames),
            "motion": deque(maxlen=window_frames),
        }
    )

    for frame in frames:
        frame_state_counts = {}
        for detection in frame.get("detections", []):
            if _is_ball_detection(detection):
                continue
            engagement = detection.get("engagement") or {}
            evidence_key = _tracklet_evidence_key(detection)
            evidence_key_kind = evidence_key[0] if evidence_key is not None else None
            evidence_key_id = evidence_key[1] if evidence_key is not None else None
            track_length = int(track_counts.get(evidence_key, 1))
            engagement_score = float(engagement.get("engagement_score") or detection.get("engagement_score") or 0.0)
            target_score = float(engagement.get("target_court_score") or detection.get("target_court_score") or 0.0)
            other_game_risk = float(engagement.get("other_game_risk") or 0.0)
            motion_score = _clip01(float(detection.get("motion_speed_px") or 0.0) / MOTION_SCORE_THRESHOLD_PX)

            if evidence_key is not None:
                history[evidence_key]["engagement"].append(engagement_score)
                history[evidence_key]["target"].append(target_score)
                history[evidence_key]["risk"].append(other_game_risk)
                history[evidence_key]["motion"].append(motion_score)
                sample_count = len(history[evidence_key]["engagement"])
                avg_engagement = float(sum(history[evidence_key]["engagement"]) / sample_count)
                avg_target = float(sum(history[evidence_key]["target"]) / sample_count)
                avg_risk = float(sum(history[evidence_key]["risk"]) / sample_count)
                avg_motion = float(sum(history[evidence_key]["motion"]) / sample_count)
            else:
                sample_count = 1
                avg_engagement = float(engagement_score)
                avg_target = float(target_score)
                avg_risk = float(other_game_risk)
                avg_motion = float(motion_score)

            persistence_score = min(sample_count, 6) / 6.0
            smoothed_score = _clip01(
                0.40 * avg_engagement
                + 0.30 * avg_target
                + 0.15 * avg_motion
                + 0.15 * persistence_score
            )
            adjacent_risk = _clip01(
                0.60 * avg_risk
                + 0.25 * _clip01((0.38 - avg_target) / 0.38)
                + 0.15 * _clip01((0.20 - avg_motion) / 0.20)
            )
            promotion_ready = bool(
                sample_count >= TRACKLET_PROMOTION_MIN_FRAMES
                and smoothed_score >= TRACKLET_PROMOTION_SCORE_THRESHOLD
                and avg_risk <= 0.45
            )
            demotion_ready = bool(
                sample_count >= TRACKLET_DEMOTION_MIN_FRAMES
                and adjacent_risk >= TRACKLET_DEMOTION_RISK_THRESHOLD
                and avg_target <= 0.35
            )
            if demotion_ready:
                temporal_state = "demote_candidate"
            elif promotion_ready:
                temporal_state = "promote_candidate"
            elif sample_count < TRACKLET_PROMOTION_MIN_FRAMES:
                temporal_state = "pending_tracklet"
            else:
                temporal_state = "hold_tracklet"

            detection["tracklet_temporal"] = {
                "kind": "tracklet_temporal_evidence_v1",
                "track_length_frames": int(track_length),
                "window_frames": int(window_frames),
                "sample_count": int(sample_count),
                "evidence_key_kind": evidence_key_kind,
                "evidence_key_id": int(evidence_key_id) if evidence_key_id is not None else None,
                "avg_engagement_score": round(float(avg_engagement), 4),
                "avg_target_court_score": round(float(avg_target), 4),
                "avg_other_game_risk": round(float(avg_risk), 4),
                "avg_motion_score": round(float(avg_motion), 4),
                "smoothed_score": round(float(smoothed_score), 4),
                "adjacent_risk": round(float(adjacent_risk), 4),
                "promotion_ready": promotion_ready,
                "demotion_ready": demotion_ready,
                "state": temporal_state,
            }
            detection["tracklet_temporal_state"] = temporal_state
            frame_state_counts[temporal_state] = frame_state_counts.get(temporal_state, 0) + 1
            summary["state_counts"][temporal_state] = summary["state_counts"].get(temporal_state, 0) + 1
            if promotion_ready:
                summary["promotion_ready_detection_count"] += 1
            if demotion_ready:
                summary["demotion_ready_detection_count"] += 1
        frame["tracklet_temporal_summary"] = {
            "state_counts": frame_state_counts,
        }
    return summary


def _vector_norm(vec):
    if vec is None:
        return 0.0
    if len(vec) < 2:
        return 0.0
    return float(np.linalg.norm([float(vec[0]), float(vec[1])]))


def _frame_visual_signature(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    thumb = cv2.resize(gray, (32, 18), interpolation=cv2.INTER_AREA)
    return (thumb.astype(np.float32) / 255.0).tolist()


def _frame_visual_delta(signature_a, signature_b):
    if not signature_a or not signature_b:
        return 0.0
    a = np.array(signature_a, dtype=np.float32)
    b = np.array(signature_b, dtype=np.float32)
    if a.shape != b.shape:
        return 0.0
    return float(np.mean(np.abs(a - b)))


def _frame_track_ids(frame):
    return {
        int(detection["track_id"])
        for detection in frame.get("detections", [])
        if detection.get("track_id") is not None
    }


def _initialize_identity_fields(detection):
    canonical_track_id = detection.get("track_id")
    detection["identity_track_id"] = canonical_track_id
    detection["identity_track_source"] = "tracker"
    detection["identity_repair"] = None
    detection["identity_option_track_id"] = detection.get("identity_option_track_id")
    detection["identity_option_count"] = int(detection.get("identity_option_count") or 0)
    detection["identity_is_ambiguous"] = bool(detection.get("identity_is_ambiguous") or False)
    detection["identity_best_canonical_track_id"] = detection.get("identity_best_canonical_track_id", canonical_track_id)
    detection["identity_option_canonical_track_ids"] = list(detection.get("identity_option_canonical_track_ids") or ([] if canonical_track_id is None else [canonical_track_id]))
    detection["identity_option_group_hypothesis_ids"] = list(detection.get("identity_option_group_hypothesis_ids") or [])
    detection["identity_options"] = list(detection.get("identity_options") or [])
    detection["identity_jersey_number"] = detection.get("identity_jersey_number")
    detection["identity_jersey_number_confidence"] = detection.get("identity_jersey_number_confidence")
    detection["identity_jersey_number_source"] = detection.get("identity_jersey_number_source")
    detection["identity_jersey_evidence_count"] = detection.get("identity_jersey_evidence_count")
    detection["appearance_team_distance"] = detection.get("appearance_team_distance")
    detection["appearance_team_bucket"] = detection.get("appearance_team_bucket")


def get_easyocr_reader():
    global _EASYOCR_READER, _EASYOCR_READER_ATTEMPTED
    if _EASYOCR_READER_ATTEMPTED:
        return _EASYOCR_READER
    _EASYOCR_READER_ATTEMPTED = True
    try:
        import torch
        import easyocr
    except Exception:
        _EASYOCR_READER = None
        return None
    try:
        _EASYOCR_READER = easyocr.Reader(["en"], gpu=bool(torch.cuda.is_available()), verbose=False)
    except Exception:
        _EASYOCR_READER = None
    return _EASYOCR_READER


def _torso_crop_bounds(frame_shape, bbox_xyxy, keypoints_xy=None, keypoints_conf=None):
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox_xyxy]
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None

    if keypoints_xy and keypoints_conf:
        torso_indices = [5, 6, 11, 12]
        torso_points = []
        for idx in torso_indices:
            if idx < len(keypoints_xy):
                conf = keypoints_conf[idx] if idx < len(keypoints_conf) else 1.0
                if conf >= 0.2:
                    torso_points.append(keypoints_xy[idx])
        if torso_points:
            xs = [p[0] for p in torso_points]
            ys = [p[1] for p in torso_points]
            pad_x = max(10, int((max(xs) - min(xs)) * 0.35))
            pad_y = max(12, int((max(ys) - min(ys)) * 0.3))
            tx1 = max(0, int(min(xs) - pad_x))
            tx2 = min(width, int(max(xs) + pad_x))
            ty1 = max(0, int(min(ys) - pad_y))
            ty2 = min(height, int(max(ys) + pad_y))
            if tx2 > tx1 and ty2 > ty1:
                return tx1, ty1, tx2, ty2

    top = y1 + int((y2 - y1) * 0.15)
    bottom = y1 + int((y2 - y1) * 0.62)
    left = x1 + int((x2 - x1) * 0.18)
    right = x2 - int((x2 - x1) * 0.18)
    if bottom > top and right > left:
        return left, top, right, bottom
    return None


def _extract_jersey_crop(frame, detection):
    bounds = _torso_crop_bounds(
        frame.shape,
        detection.get("bbox_xyxy", [0, 0, 0, 0]),
        keypoints_xy=detection.get("keypoints_xy"),
        keypoints_conf=detection.get("keypoints_conf"),
    )
    if bounds is None:
        return None, 0.0
    x1, y1, x2, y2 = bounds
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < JERSEY_OCR_MIN_CROP_HEIGHT or crop.shape[1] < JERSEY_OCR_MIN_CROP_WIDTH:
        return None, 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return crop, sharpness


def estimate_torso_color_histogram(crop_bgr):
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    pixels = crop_bgr.reshape(-1, 3).astype(np.float32)
    if pixels.size == 0:
        return None
    quantized = (pixels >= 128.0).astype(np.int32)
    bin_ids = quantized[:, 0] * 4 + quantized[:, 1] * 2 + quantized[:, 2]
    hist = np.bincount(bin_ids, minlength=8).astype(np.float32)
    total = float(hist.sum())
    if total <= 0.0:
        return None
    hist /= total
    return [round(float(v), 6) for v in hist.tolist()]


def histogram_intersection_distance(hist_a, hist_b):
    if not hist_a or not hist_b or len(hist_a) != len(hist_b):
        return None
    intersection = sum(min(float(a), float(b)) for a, b in zip(hist_a, hist_b))
    return round(max(0.0, 1.0 - intersection), 6)


def _normalize_jersey_text(text):
    if not text:
        return None
    raw = str(text).strip().upper()
    if not raw:
        return None
    if not any(ch.isdigit() for ch in raw) and len(raw) > 2:
        return None
    cleaned = (
        raw
        .upper()
        .replace("O", "0")
        .replace("I", "1")
        .replace("L", "1")
        .replace("S", "5")
        .replace("B", "8")
        .replace("Z", "2")
    )
    digits = re.sub(r"[^0-9]", "", cleaned)
    if not digits:
        return None
    if len(digits) > 2:
        digits = digits[-2:]
    try:
        value = int(digits)
    except ValueError:
        return None
    if value <= 0 or value > 99:
        return None
    return str(value)


def _run_jersey_ocr(reader, crop_bgr):
    if reader is None or crop_bgr is None or crop_bgr.size == 0:
        return None
    try:
        results = reader.readtext(crop_bgr, detail=1, allowlist="0123456789OILSBZ", paragraph=False)
    except Exception:
        return None
    best = None
    for item in results:
        if len(item) < 3:
            continue
        _bbox, raw_text, conf = item
        normalized = _normalize_jersey_text(raw_text)
        if normalized is None:
            continue
        confidence = float(conf)
        if best is None or confidence > best["ocr_confidence"]:
            best = {
                "candidate": normalized,
                "raw_text": str(raw_text),
                "ocr_confidence": confidence,
            }
    return best


def _collect_jersey_evidence(reader, detection):
    crop = detection.get("_jersey_crop_bgr")
    sharpness = float(detection.get("_jersey_crop_sharpness") or 0.0)
    if crop is None or sharpness < JERSEY_OCR_MIN_SHARPNESS:
        return None
    ocr = _run_jersey_ocr(reader, crop)
    if ocr is None:
        return None
    evidence = {
        "candidate": ocr["candidate"],
        "raw_text": ocr["raw_text"],
        "ocr_confidence": round(float(ocr["ocr_confidence"]), 4),
        "sharpness": round(sharpness, 3),
        "source": "easyocr_v1",
    }
    detection["jersey_number_evidence"] = evidence
    return evidence


def _resolve_identity_jersey_consensus(evidence_items):
    if len(evidence_items) < JERSEY_OCR_CONSENSUS_MIN_VOTES:
        return None
    votes = {}
    total_weight = 0.0
    total_count = 0
    for evidence in evidence_items:
        candidate = evidence.get("candidate")
        if candidate is None:
            continue
        confidence = float(evidence.get("ocr_confidence") or 0.0)
        sharpness = float(evidence.get("sharpness") or 0.0)
        quality_weight = _clip01(sharpness / 120.0)
        weight = max(0.05, confidence * max(0.2, quality_weight))
        total_weight += weight
        total_count += 1
        row = votes.setdefault(candidate, {"vote_count": 0, "weight": 0.0})
        row["vote_count"] += 1
        row["weight"] += weight
    if total_count < JERSEY_OCR_CONSENSUS_MIN_VOTES or total_weight <= 0.0:
        return None
    winner, winner_row = max(votes.items(), key=lambda item: (item[1]["weight"], item[1]["vote_count"], item[0]))
    confidence = winner_row["weight"] / total_weight
    if winner_row["vote_count"] < JERSEY_OCR_CONSENSUS_MIN_VOTES:
        return None
    if confidence < JERSEY_OCR_CONSENSUS_MIN_SHARE:
        return None
    return {
        "number": winner,
        "confidence": round(confidence, 4),
        "vote_count": int(winner_row["vote_count"]),
        "evidence_count": int(total_count),
        "votes": {
            number: {
                "vote_count": int(row["vote_count"]),
                "weight": round(float(row["weight"]), 4),
            }
            for number, row in sorted(votes.items())
        },
        "source": "easyocr_consensus_v1",
    }


def _sort_and_clip_jersey_samples(samples):
    ranked = sorted(
        samples,
        key=lambda item: (
            float(item.get("sharpness") or 0.0),
            float(item.get("ocr_confidence") or 0.0),
            int(item.get("frame_idx") or -1),
        ),
        reverse=True,
    )
    return ranked[:JERSEY_OCR_MAX_SAMPLES_PER_IDENTITY]


def _serialize_jersey_consensus(consensus):
    return {
        "number": consensus["number"],
        "confidence": consensus["confidence"],
        "vote_count": consensus["vote_count"],
        "evidence_count": consensus["evidence_count"],
        "votes": consensus["votes"],
        "samples": [
            {
                "frame_idx": sample["frame_idx"],
                "track_id": sample["track_id"],
                "identity_track_id": sample.get("identity_track_id"),
                "candidate": sample["candidate"],
                "raw_text": sample["raw_text"],
                "ocr_confidence": sample["ocr_confidence"],
                "sharpness": sample["sharpness"],
            }
            for sample in consensus["samples"]
        ],
    }


def _build_track_option_consensus(track_samples, identity_hypotheses):
    track_identity_options = (identity_hypotheses or {}).get("track_identity_options") or []
    if not track_identity_options:
        return {}

    track_option_consensus = {}
    for record in track_identity_options:
        track_id = int(record["track_id"])
        serialized_options = []
        for option in record.get("options") or []:
            option_samples = []
            for chain_track_id in option.get("chain_track_ids") or []:
                option_samples.extend(track_samples.get(int(chain_track_id), []))
            option_samples = _sort_and_clip_jersey_samples(option_samples)
            consensus = _resolve_identity_jersey_consensus(option_samples)
            serialized_option = {
                "canonical_track_id": int(option["canonical_track_id"]),
                "chain_track_ids": [int(value) for value in option.get("chain_track_ids") or []],
                "support_share": round(float(option.get("support_share") or 0.0), 4),
                "best_total_score": round(float(option.get("best_total_score") or 0.0), 4),
                "best_score_margin_to_best": round(float(option.get("best_score_margin_to_best") or 0.0), 4),
                "hypothesis_ids": list(option.get("hypothesis_ids") or []),
                "group_hypothesis_ids": list(option.get("group_hypothesis_ids") or []),
                "has_consensus": consensus is not None,
            }
            if consensus is not None:
                serialized_option["consensus"] = _serialize_jersey_consensus({**consensus, "samples": option_samples})
            serialized_options.append(serialized_option)
        track_option_consensus[track_id] = {
            "track_id": track_id,
            "is_ambiguous": bool(record.get("is_ambiguous")),
            "option_count": int(record.get("option_count") or len(serialized_options)),
            "best_canonical_track_id": int(record.get("best_canonical_track_id") or track_id),
            "options": serialized_options,
        }
    return track_option_consensus


def annotate_identity_jersey_numbers(frames, *, identity_hypotheses=None):
    reader = None
    identity_samples = {}
    track_samples = {}
    plausible_identity_counts = defaultdict(int)
    plausible_track_counts = defaultdict(int)
    for frame in frames:
        for detection in frame.get("detections", []):
            identity_track_id = detection.get("identity_track_id")
            if identity_track_id is None:
                continue
            if detection.get("active_player_candidate") or detection.get("on_court_candidate"):
                plausible_identity_counts[int(identity_track_id)] += 1
                track_id = detection.get("track_id")
                if track_id is not None:
                    plausible_track_counts[int(track_id)] += 1
    for frame in frames:
        for detection in frame.get("detections", []):
            identity_track_id = detection.get("identity_track_id")
            if identity_track_id is None:
                detection["identity_jersey_evidence_count"] = 0
                continue
            detection["identity_jersey_evidence_count"] = 0
            if not detection.get("on_court_candidate") and not detection.get("active_player_candidate"):
                continue
            track_id = detection.get("track_id")
            identity_count = plausible_identity_counts.get(int(identity_track_id), 0)
            track_count = plausible_track_counts.get(int(track_id), 0) if track_id is not None else 0
            if max(identity_count, track_count) < JERSEY_OCR_MIN_ON_COURT_TRACK_FRAMES:
                continue
            identity_bucket = identity_samples.get(identity_track_id, [])
            track_bucket = track_samples.get(int(track_id), []) if track_id is not None else []
            if len(identity_bucket) >= JERSEY_OCR_MAX_SAMPLES_PER_IDENTITY and (
                track_id is None or len(track_bucket) >= JERSEY_OCR_MAX_SAMPLES_PER_IDENTITY
            ):
                continue
            if reader is None:
                reader = get_easyocr_reader()
            evidence = _collect_jersey_evidence(reader, detection)
            if evidence is None:
                continue
            sample = {
                "frame_idx": frame.get("frame_idx"),
                "track_id": detection.get("track_id"),
                "identity_track_id": identity_track_id,
                **evidence,
            }
            identity_bucket = identity_samples.setdefault(identity_track_id, [])
            identity_bucket.append(sample)
            identity_samples[identity_track_id] = _sort_and_clip_jersey_samples(identity_bucket)
            if track_id is not None:
                track_bucket = track_samples.setdefault(int(track_id), [])
                track_bucket.append(sample)
                track_samples[int(track_id)] = _sort_and_clip_jersey_samples(track_bucket)

    identity_consensus = {}
    for identity_track_id, samples in identity_samples.items():
        consensus = _resolve_identity_jersey_consensus(samples)
        if consensus is None:
            continue
        identity_consensus[identity_track_id] = {
            **consensus,
            "samples": samples,
        }

    track_option_consensus = _build_track_option_consensus(track_samples, identity_hypotheses)

    for frame in frames:
        for detection in frame.get("detections", []):
            identity_track_id = detection.get("identity_track_id")
            consensus = identity_consensus.get(identity_track_id)
            if consensus is not None:
                detection["identity_jersey_number"] = consensus["number"]
                detection["identity_jersey_number_confidence"] = consensus["confidence"]
                detection["identity_jersey_number_source"] = consensus["source"]
                detection["identity_jersey_evidence_count"] = consensus["evidence_count"]
            track_id = detection.get("track_id")
            option_record = track_option_consensus.get(int(track_id)) if track_id is not None else None
            if option_record is not None:
                detection["identity_jersey_options"] = json.loads(json.dumps(option_record["options"]))
                detection["identity_jersey_option_count"] = int(option_record["option_count"])
                detection["identity_jersey_is_ambiguous"] = bool(option_record["is_ambiguous"])
                detection["identity_jersey_best_canonical_track_id"] = int(option_record["best_canonical_track_id"])
            detection.pop("_jersey_crop_bgr", None)
            detection.pop("_jersey_crop_sharpness", None)
    return {
        "reader_available": reader is not None,
        "identity_count_with_consensus": len(identity_consensus),
        "track_option_count_with_consensus": sum(
            1 for record in track_option_consensus.values() if any(option.get("has_consensus") for option in record.get("options") or [])
        ),
        "ambiguous_track_option_count_with_consensus": sum(
            1
            for record in track_option_consensus.values()
            if record.get("is_ambiguous") and any(option.get("has_consensus") for option in record.get("options") or [])
        ),
        "identity_consensus": {
            str(identity_track_id): _serialize_jersey_consensus(consensus)
            for identity_track_id, consensus in sorted(identity_consensus.items())
        },
        "track_option_consensus": {
            str(track_id): json.loads(json.dumps(record))
            for track_id, record in sorted(track_option_consensus.items())
        },
    }


def _jersey_consensus_strength(consensus):
    if not consensus:
        return 0.0
    confidence = float(consensus.get("confidence") or 0.0)
    evidence_count = float(consensus.get("evidence_count") or 0.0)
    vote_count = float(consensus.get("vote_count") or 0.0)
    evidence_factor = min(1.0, evidence_count / float(max(JERSEY_OCR_CONSENSUS_MIN_VOTES + 1, 1)))
    vote_factor = min(1.0, vote_count / float(max(JERSEY_OCR_CONSENSUS_MIN_VOTES, 1)))
    return confidence * (0.6 * evidence_factor + 0.4 * vote_factor)


def resolve_identity_global_hypotheses_with_jersey(identity_hypotheses, jersey_ocr):
    global_hypotheses = list((identity_hypotheses or {}).get("global_hypotheses") or [])
    track_option_consensus = (jersey_ocr or {}).get("track_option_consensus") or {}
    if not global_hypotheses:
        return {
            "kind": "jersey_global_identity_tiebreak_v1",
            "weight": JERSEY_IDENTITY_TIEBREAK_WEIGHT,
            "base_selected_global_hypothesis_id": None,
            "selected_global_hypothesis_id": None,
            "changed_selected_global_hypothesis": False,
            "global_hypothesis_count": 0,
            "global_hypotheses": [],
            "preferred_track_resolution": {},
        }

    reranked = []
    for global_hypothesis in global_hypotheses:
        global_hypothesis_id = global_hypothesis["global_hypothesis_id"]
        jersey_bonus = 0.0
        supported_track_count = 0
        conflicted_track_count = 0
        track_breakdown = []
        for track_id_str, record in sorted(track_option_consensus.items(), key=lambda item: int(item[0])):
            options = list(record.get("options") or [])
            if not options:
                continue
            chosen_option = next((option for option in options if global_hypothesis_id in (option.get("hypothesis_ids") or [])), None)
            if chosen_option is None and len(options) == 1:
                chosen_option = options[0]
            if chosen_option is None:
                continue
            chosen_strength = _jersey_consensus_strength(chosen_option.get("consensus"))
            competitor_strength = max(
                (_jersey_consensus_strength(option.get("consensus")) for option in options if option is not chosen_option),
                default=0.0,
            )
            delta = JERSEY_IDENTITY_TIEBREAK_WEIGHT * (chosen_strength - competitor_strength)
            if chosen_strength > competitor_strength:
                supported_track_count += 1
            elif chosen_strength < competitor_strength:
                conflicted_track_count += 1
            if chosen_strength > 0.0 or competitor_strength > 0.0 or abs(delta) > 1e-9:
                track_breakdown.append({
                    "track_id": int(track_id_str),
                    "chosen_canonical_track_id": int(chosen_option.get("canonical_track_id") or int(track_id_str)),
                    "chosen_consensus_number": (chosen_option.get("consensus") or {}).get("number"),
                    "chosen_consensus_strength": round(float(chosen_strength), 4),
                    "best_competing_consensus_strength": round(float(competitor_strength), 4),
                    "jersey_delta": round(float(delta), 4),
                    "has_competing_consensus": any(option is not chosen_option and option.get("consensus") for option in options),
                })
            jersey_bonus += delta
        reranked.append({
            **global_hypothesis,
            "base_rank": int(global_hypothesis.get("rank") or 0),
            "base_total_score": round(float(global_hypothesis.get("total_score") or 0.0), 4),
            "jersey_bonus": round(float(jersey_bonus), 4),
            "jersey_supported_track_count": int(supported_track_count),
            "jersey_conflicted_track_count": int(conflicted_track_count),
            "track_breakdown": track_breakdown,
            "reranked_total_score": round(float(global_hypothesis.get("total_score") or 0.0) + float(jersey_bonus), 4),
        })

    reranked.sort(
        key=lambda item: (
            -float(item["reranked_total_score"]),
            -int(item.get("jersey_supported_track_count") or 0),
            int(item.get("base_rank") or 0),
            str(item["global_hypothesis_id"]),
        )
    )
    best_total_score = float(reranked[0]["reranked_total_score"])
    normalizer = sum(np.exp(float(item["reranked_total_score"]) - best_total_score) for item in reranked)
    selected_global_hypothesis_id = reranked[0]["global_hypothesis_id"]
    for rank, item in enumerate(reranked, start=1):
        item["reranked_rank"] = int(rank)
        item["reranked_score_margin_to_best"] = round(best_total_score - float(item["reranked_total_score"]), 4)
        item["reranked_score_share"] = round(float(np.exp(float(item["reranked_total_score"]) - best_total_score) / normalizer), 4) if normalizer > 0.0 else 0.0
        item["selected_after_jersey_tiebreak"] = item["global_hypothesis_id"] == selected_global_hypothesis_id

    preferred_track_resolution = {}
    selected_record = reranked[0]
    for track_id_str, record in sorted(track_option_consensus.items(), key=lambda item: int(item[0])):
        options = list(record.get("options") or [])
        if not options:
            continue
        chosen_option = next(
            (option for option in options if selected_global_hypothesis_id in (option.get("hypothesis_ids") or [])),
            None,
        )
        if chosen_option is None and len(options) == 1:
            chosen_option = options[0]
        if chosen_option is None:
            continue
        consensus = chosen_option.get("consensus") or {}
        preferred_track_resolution[str(track_id_str)] = {
            "track_id": int(track_id_str),
            "selected_global_hypothesis_id": selected_global_hypothesis_id,
            "preferred_canonical_track_id": int(chosen_option.get("canonical_track_id") or int(track_id_str)),
            "preferred_chain_track_ids": [int(value) for value in chosen_option.get("chain_track_ids") or []],
            "selected_option_has_consensus": bool(chosen_option.get("has_consensus")),
            "selected_consensus_number": consensus.get("number"),
            "selected_consensus_confidence": consensus.get("confidence"),
            "selected_consensus_evidence_count": consensus.get("evidence_count"),
            "is_ambiguous": bool(record.get("is_ambiguous")),
        }

    return {
        "kind": "jersey_global_identity_tiebreak_v1",
        "weight": JERSEY_IDENTITY_TIEBREAK_WEIGHT,
        "base_selected_global_hypothesis_id": global_hypotheses[0]["global_hypothesis_id"],
        "selected_global_hypothesis_id": selected_global_hypothesis_id,
        "changed_selected_global_hypothesis": global_hypotheses[0]["global_hypothesis_id"] != selected_global_hypothesis_id,
        "global_hypothesis_count": len(reranked),
        "global_hypotheses": reranked,
        "preferred_track_resolution": preferred_track_resolution,
    }


def annotate_detections_with_jersey_global_resolution(frames, resolution):
    preferred_track_resolution = (resolution or {}).get("preferred_track_resolution") or {}
    selected_global_hypothesis_id = (resolution or {}).get("selected_global_hypothesis_id")
    for frame in frames:
        for detection in frame.get("detections", []):
            track_id = detection.get("track_id")
            if track_id is None:
                continue
            preferred = preferred_track_resolution.get(str(int(track_id)))
            if preferred is None:
                continue
            detection["identity_jersey_selected_global_hypothesis_id"] = selected_global_hypothesis_id
            detection["identity_jersey_preferred_canonical_track_id"] = int(preferred["preferred_canonical_track_id"])
            detection["identity_jersey_preferred_chain_track_ids"] = [int(value) for value in preferred.get("preferred_chain_track_ids") or []]
            detection["identity_jersey_selected_option_has_consensus"] = bool(preferred.get("selected_option_has_consensus"))
            if preferred.get("selected_consensus_number") is not None:
                detection["identity_jersey_preferred_number"] = preferred.get("selected_consensus_number")
                detection["identity_jersey_preferred_number_confidence"] = preferred.get("selected_consensus_confidence")


class KalmanTrack2D:
    """Minimal constant-velocity Kalman filter for bbox-center smoothing."""

    def __init__(self, x, y):
        self.state = np.array([x, y, 0.0, 0.0], dtype=float)
        self.cov = np.eye(4, dtype=float) * 25.0
        self.process_noise = np.diag([1.0, 1.0, 4.0, 4.0]).astype(float)
        self.measurement_noise = np.diag([9.0, 9.0]).astype(float)

    def predict(self, dt):
        transition = np.array([
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=float)
        self.state = transition @ self.state
        self.cov = transition @ self.cov @ transition.T + self.process_noise
        return self.state.copy()

    def update(self, measurement_xy):
        observe = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=float)
        innovation = measurement_xy - (observe @ self.state)
        innovation_cov = observe @ self.cov @ observe.T + self.measurement_noise
        kalman_gain = self.cov @ observe.T @ np.linalg.inv(innovation_cov)
        self.state = self.state + kalman_gain @ innovation
        self.cov = (np.eye(4) - kalman_gain @ observe) @ self.cov
        return self.state.copy()


def estimate_uniform_bucket(frame, bbox_xyxy, keypoints_xy=None, keypoints_conf=None):
    x1, y1, x2, y2 = [int(round(v)) for v in bbox_xyxy]
    height, width = frame.shape[:2]
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return {"bucket": "unknown", "luma_mean": None}

    torso_region = None
    if keypoints_xy and keypoints_conf:
        torso_indices = [5, 6, 11, 12]
        torso_points = []
        for idx in torso_indices:
            if idx < len(keypoints_xy):
                conf = keypoints_conf[idx] if idx < len(keypoints_conf) else 1.0
                if conf >= 0.2:
                    torso_points.append(keypoints_xy[idx])
        if torso_points:
            xs = [p[0] for p in torso_points]
            ys = [p[1] for p in torso_points]
            pad_x = max(8, int((max(xs) - min(xs)) * 0.2))
            pad_y = max(8, int((max(ys) - min(ys)) * 0.25))
            tx1 = max(0, int(min(xs) - pad_x))
            tx2 = min(width, int(max(xs) + pad_x))
            ty1 = max(0, int(min(ys) - pad_y))
            ty2 = min(height, int(max(ys) + pad_y))
            if tx2 > tx1 and ty2 > ty1:
                torso_region = frame[ty1:ty2, tx1:tx2]

    if torso_region is None or torso_region.size == 0:
        top = y1 + int((y2 - y1) * 0.18)
        bottom = y1 + int((y2 - y1) * 0.58)
        left = x1 + int((x2 - x1) * 0.15)
        right = x2 - int((x2 - x1) * 0.15)
        if bottom > top and right > left:
            torso_region = frame[top:bottom, left:right]

    if torso_region is None or torso_region.size == 0:
        return {"bucket": "unknown", "luma_mean": None}

    gray = cv2.cvtColor(torso_region, cv2.COLOR_BGR2GRAY)
    luma_mean = float(gray.mean())
    if luma_mean >= 150:
        bucket = "light"
    elif luma_mean <= 95:
        bucket = "dark"
    else:
        bucket = "unknown"
    return {"bucket": bucket, "luma_mean": luma_mean}


def score_active_player(
    detection,
    *,
    frame_width,
    frame_height,
    track_frame_count=1,
    bootstrap_context=None,
    scene_prior=None,
    frame_motion_context=None,
):
    confidence = float(detection.get("confidence") or 0.0)
    bbox_xyxy = detection.get("bbox_xyxy") or [0.0, 0.0, 0.0, 0.0]
    x1, y1, x2, y2 = bbox_xyxy
    bbox_h = max(0.0, y2 - y1)
    bbox_cx = (x1 + x2) * 0.5
    bbox_cy = (y1 + y2) * 0.5

    reasons = {}

    reasons["confidence"] = round(confidence, 4)
    confidence_score = _clip01(confidence)

    height_ratio = bbox_h / max(float(frame_height), 1.0)
    height_score = _clip01((height_ratio - MIN_PLAYER_BBOX_HEIGHT_RATIO) / 0.18)
    reasons["height_ratio"] = round(height_ratio, 4)

    edge_margin_x = frame_width * EDGE_MARGIN_RATIO
    edge_margin_y = frame_height * EDGE_MARGIN_RATIO
    edge_penalty = 0.0
    if x1 < edge_margin_x or x2 > frame_width - edge_margin_x:
        edge_penalty += 0.15
    if y1 < edge_margin_y or y2 > frame_height - edge_margin_y:
        edge_penalty += 0.05
    reasons["edge_penalty"] = round(edge_penalty, 4)

    target_court_support = _compute_detection_target_court_support(
        detection,
        frame_width=frame_width,
        frame_height=frame_height,
        bootstrap_context=bootstrap_context,
        scene_prior=scene_prior,
    )
    footpoint_xy = target_court_support["footpoint_xy"]
    court_ground_xy = target_court_support["court_ground_xy"]
    court_in_bounds = target_court_support["court_in_bounds"]
    grounding_score = float(target_court_support["grounding_score"])
    reasons["footpoint_xy"] = [round(float(footpoint_xy[0]), 2), round(float(footpoint_xy[1]), 2)]
    reasons["footpoint_source"] = target_court_support["footpoint_source"]
    reasons["footpoint_confidence"] = round(float(target_court_support["footpoint_confidence"]), 4)
    reasons["court_in_bounds"] = court_in_bounds
    reasons["court_ground_xy"] = (
        [round(float(court_ground_xy[0]), 2), round(float(court_ground_xy[1]), 2)]
        if court_ground_xy is not None and len(court_ground_xy) == 2
        else None
    )
    reasons["court_distance_outside"] = round(float(target_court_support["court_distance_outside"]), 3) if target_court_support["court_distance_outside"] is not None else None
    reasons["court_boundary_score"] = round(float(target_court_support["court_boundary_score"]), 4)

    persistence_score = min(track_frame_count, 5) / 5.0
    reasons["track_frame_count"] = int(track_frame_count)

    motion_speed = float(detection.get("motion_speed_px") or 0.0)
    coherent_velocity = (
        (frame_motion_context or {}).get("coherent_velocity_xy")
        or [0.0, 0.0]
    )
    coherent_speed = float((frame_motion_context or {}).get("coherent_speed_px") or 0.0)
    shared_motion = bool((frame_motion_context or {}).get("shared_motion"))
    smoothed_velocity = detection.get("smoothed_velocity_xy")
    if smoothed_velocity is not None and len(smoothed_velocity) >= 2:
        relative_velocity = [
            float(smoothed_velocity[0]) - float(coherent_velocity[0]),
            float(smoothed_velocity[1]) - float(coherent_velocity[1]),
        ]
        relative_motion_speed = float(np.linalg.norm(relative_velocity))
    else:
        relative_motion_speed = motion_speed
    motion_score = _clip01(relative_motion_speed / MOTION_SCORE_THRESHOLD_PX)
    reasons["motion_speed_px"] = round(motion_speed, 3)
    reasons["relative_motion_speed_px"] = round(relative_motion_speed, 3)
    reasons["frame_coherent_motion_px"] = round(coherent_speed, 3)
    reasons["frame_shared_motion"] = shared_motion

    center_foreground_prior = float(target_court_support["bootstrap_foreground_prior"])
    foot_foreground_prior = float(target_court_support["bootstrap_foot_prior"])
    scene_center_prior = float(target_court_support["scene_center_prior"])
    scene_foot_prior = float(target_court_support["scene_foot_prior"])
    effective_center_prior = float(target_court_support["effective_foreground_prior"])
    effective_foot_prior = float(target_court_support["effective_foot_prior"])
    reasons["bootstrap_foreground_prior"] = round(float(center_foreground_prior), 4)
    reasons["bootstrap_foot_prior"] = round(float(foot_foreground_prior), 4)
    reasons["scene_prior_status"] = (scene_prior or {}).get("prior_status")
    reasons["scene_center_prior"] = round(float(scene_center_prior), 4)
    reasons["scene_foot_prior"] = round(float(scene_foot_prior), 4)
    reasons["effective_foreground_prior"] = round(float(effective_center_prior), 4)
    reasons["effective_foot_prior"] = round(float(effective_foot_prior), 4)
    reasons["target_court_score"] = round(float(target_court_support["target_court_score"]), 4)

    pose_info = _score_pose_coherence(detection)
    reasons["pose_visible_count"] = int(pose_info["visible_count"])
    reasons["pose_torso_visible_count"] = int(pose_info["torso_visible_count"])
    reasons["pose_lower_visible_count"] = int(pose_info["lower_visible_count"])
    reasons["pose_symmetric_pair_count"] = int(pose_info["symmetric_pair_count"])
    reasons["pose_coherence"] = round(float(pose_info["score"]), 4)

    merge_info = _score_merge_risk(
        detection,
        frame_width=frame_width,
        frame_height=frame_height,
        pose_info=pose_info,
    )
    reasons["merge_risk"] = round(float(merge_info["score"]), 4)
    reasons["bbox_aspect_ratio"] = round(float(merge_info["aspect_ratio"]), 4)
    reasons["bbox_area_ratio"] = round(float(merge_info["area_ratio"]), 4)

    sam3_refinement = detection.get("sam3_refinement") or {}
    sam3_score = float(sam3_refinement.get("sam_score") or 0.0)
    sam3_mask_area_ratio = sam3_refinement.get("mask_area_ratio")
    reasons["sam3_score"] = round(sam3_score, 4)
    reasons["sam3_mask_area_ratio"] = round(float(sam3_mask_area_ratio), 4) if sam3_mask_area_ratio is not None else None
    reasons["sam3_source_kind"] = sam3_refinement.get("source_kind")

    appearance_distance = detection.get("appearance_team_distance")
    reasons["appearance_team_distance"] = round(float(appearance_distance), 4) if appearance_distance is not None else None
    appearance_penalty = 0.0
    if (
        appearance_distance is not None
        and motion_speed <= APPEARANCE_LOW_MOTION_THRESHOLD_PX
        and edge_penalty >= 0.15
        and float(appearance_distance) > APPEARANCE_TEAM_DISTANCE_THRESHOLD
    ):
        appearance_penalty = 0.14 * _clip01(
            (float(appearance_distance) - APPEARANCE_TEAM_DISTANCE_THRESHOLD)
            / max(1.0 - APPEARANCE_TEAM_DISTANCE_THRESHOLD, 1e-6)
        )
    reasons["appearance_penalty"] = round(appearance_penalty, 4)

    engagement = detection.get("engagement") or {}
    engagement_score = float(engagement.get("engagement_score") or 0.0)
    target_game_score = float(engagement.get("target_court_score") or target_court_support["target_court_score"])
    other_game_risk = float(engagement.get("other_game_risk") or 0.0)
    engagement_state = str(engagement.get("engagement_state") or detection.get("engagement_state") or "unknown")
    reasons["engagement_score"] = round(float(engagement_score), 4)
    reasons["engagement_state"] = engagement_state
    reasons["other_game_risk"] = round(float(other_game_risk), 4)

    tracklet_temporal = detection.get("tracklet_temporal") or {}
    tracklet_temporal_score = float(tracklet_temporal.get("smoothed_score") or 0.0)
    tracklet_temporal_risk = float(tracklet_temporal.get("adjacent_risk") or 0.0)
    tracklet_temporal_state = str(tracklet_temporal.get("state") or detection.get("tracklet_temporal_state") or "none")
    tracklet_promotion_ready = bool(tracklet_temporal.get("promotion_ready"))
    tracklet_demotion_ready = bool(tracklet_temporal.get("demotion_ready"))
    reasons["tracklet_temporal_score"] = round(float(tracklet_temporal_score), 4)
    reasons["tracklet_temporal_risk"] = round(float(tracklet_temporal_risk), 4)
    reasons["tracklet_temporal_state"] = tracklet_temporal_state
    reasons["tracklet_promotion_ready"] = tracklet_promotion_ready
    reasons["tracklet_demotion_ready"] = tracklet_demotion_ready

    geometry_prior = max(
        float(grounding_score),
        float(effective_foot_prior),
        float(effective_center_prior) * 0.7,
        float(target_game_score),
    )
    reasons["geometry_prior"] = round(float(geometry_prior), 4)
    ungrounded_shared_motion_penalty = 0.0
    if float(geometry_prior) <= 0.45 and shared_motion and relative_motion_speed <= MOTION_SCORE_THRESHOLD_PX:
        ungrounded_shared_motion_penalty = 0.14 * _clip01(
            (coherent_speed - relative_motion_speed) / max(MOTION_SCORE_THRESHOLD_PX * 2.0, 1e-6)
        )
    reasons["ungrounded_shared_motion_penalty"] = round(float(ungrounded_shared_motion_penalty), 4)
    engagement_bonus = 0.06 * float(engagement_score) + 0.04 * float(target_game_score)
    engagement_penalty = float(other_game_risk) * (0.14 if float(target_game_score) < 0.45 else 0.05)
    tracklet_temporal_bonus = (0.08 if tracklet_promotion_ready else 0.03) * float(tracklet_temporal_score)
    tracklet_temporal_penalty = (0.10 if tracklet_demotion_ready else 0.0) * float(tracklet_temporal_risk)
    reasons["engagement_bonus"] = round(float(engagement_bonus), 4)
    reasons["engagement_penalty"] = round(float(engagement_penalty), 4)
    reasons["tracklet_temporal_bonus"] = round(float(tracklet_temporal_bonus), 4)
    reasons["tracklet_temporal_penalty"] = round(float(tracklet_temporal_penalty), 4)
    on_court_score = (
        float(ON_COURT_POSITIVE_WEIGHTS["confidence"]) * confidence_score
        + float(ON_COURT_POSITIVE_WEIGHTS["bbox_height"]) * height_score
        + float(ON_COURT_POSITIVE_WEIGHTS["court_grounding"]) * float(grounding_score)
        + float(ON_COURT_POSITIVE_WEIGHTS["foot_foreground_prior"]) * float(effective_foot_prior)
        + float(ON_COURT_POSITIVE_WEIGHTS["center_foreground_prior"]) * float(effective_center_prior)
        + float(ON_COURT_POSITIVE_WEIGHTS["pose_coherence"]) * float(pose_info["score"])
        + float(ON_COURT_POSITIVE_WEIGHTS["track_persistence"]) * float(persistence_score)
        + float(ON_COURT_POSITIVE_WEIGHTS["motion"]) * float(motion_score)
        + float(ON_COURT_POSITIVE_WEIGHTS["sam3_support"]) * _clip01(sam3_score / 0.8)
        + float(engagement_bonus)
        + float(tracklet_temporal_bonus)
        - float(ON_COURT_PENALTY_WEIGHTS["edge"]) * float(edge_penalty)
        - float(ON_COURT_PENALTY_WEIGHTS["merge_risk"]) * float(merge_info["score"])
        - float(ON_COURT_PENALTY_WEIGHTS["appearance_mismatch"]) * float(appearance_penalty)
        - float(ON_COURT_PENALTY_WEIGHTS["ungrounded_shared_motion"]) * float(ungrounded_shared_motion_penalty)
        - float(engagement_penalty)
        - float(tracklet_temporal_penalty)
    )
    spectator_risk = _clip01(
        float(ON_COURT_SPECTATOR_RISK_WEIGHTS["edge"]) * _clip01(edge_penalty / 0.20)
        + float(ON_COURT_SPECTATOR_RISK_WEIGHTS["weak_geometry"]) * (1.0 - float(geometry_prior))
        + float(ON_COURT_SPECTATOR_RISK_WEIGHTS["merge_risk"]) * float(merge_info["score"])
        + float(ON_COURT_SPECTATOR_RISK_WEIGHTS["appearance_mismatch"]) * _clip01(appearance_penalty / 0.14)
        + float(ON_COURT_SPECTATOR_RISK_WEIGHTS["weak_pose"]) * (1.0 - float(pose_info["score"]))
        + 0.20 * float(other_game_risk)
    )
    on_court_score = round(_clip01(on_court_score), 4)
    reasons["on_court_score"] = on_court_score
    reasons["spectator_risk"] = round(float(spectator_risk), 4)

    active_score = _clip01(
        on_court_score
        + 0.09 * float(motion_score)
        + 0.04 * float(persistence_score)
        + 0.04 * float(engagement_score)
        + 0.05 * float(tracklet_temporal_score)
        - 0.03 * float(merge_info["score"])
        - 0.04 * float(other_game_risk)
        - 0.05 * float(tracklet_temporal_risk)
    )

    active_score = round(float(active_score), 4)
    on_court_candidate = on_court_score >= ON_COURT_SCORE_THRESHOLD
    return {
        "on_court_score": on_court_score,
        "on_court_candidate": on_court_candidate,
        "score": active_score,
        "candidate": on_court_candidate and active_score >= ACTIVE_PLAYER_SCORE_THRESHOLD,
        "reasons": reasons,
    }


def load_calibration(clip_id, calibration_file):
    if calibration_file is None or not calibration_file.exists():
        return None
    with open(calibration_file, "r") as f:
        calibrations = json.load(f)
    return calibrations.get(clip_id)


def get_frame_homography(calibration, frame_idx):
    if not calibration:
        return None
    if "h_sequence" in calibration and calibration["h_sequence"]:
        raw_h = calibration["h_sequence"].get(str(frame_idx))
        if raw_h is not None:
            return np.array(raw_h, dtype=float)
    if "h_matrix" in calibration:
        return np.array(calibration["h_matrix"], dtype=float)
    return None


def build_detection(result, det_idx, model_names, frame, h_matrix=None):
    boxes = result.boxes
    cls_id = int(boxes.cls[det_idx].item())
    confidence = float(boxes.conf[det_idx].item())
    track_id = None
    if boxes.id is not None:
        track_id = int(boxes.id[det_idx].item())

    detection = {
        "track_id": track_id,
        "class_id": cls_id,
        "class_name": model_names.get(cls_id, str(cls_id)),
        "confidence": confidence,
        "bbox_xyxy": _to_float_list(boxes.xyxy[det_idx].tolist()),
        "bbox_xywh": _to_float_list(boxes.xywh[det_idx].tolist()),
    }

    if cls_id == 0 and result.keypoints is not None and det_idx < len(result.keypoints):
        keypoints_xy = result.keypoints.xy[det_idx].tolist()
        detection["keypoints_xy"] = [[float(x), float(y)] for x, y in keypoints_xy]
        if result.keypoints.conf is not None:
            detection["keypoints_conf"] = _to_float_list(
                result.keypoints.conf[det_idx].tolist()
            )
        if h_matrix is not None:
            keypoints_np = np.array(detection["keypoints_xy"], dtype=float)
            lifted = lift_keypoints_to_3d(keypoints_np, h_matrix)
            detection["lifted_keypoints_xyz"] = [
                [float(x), float(y), float(z)] for x, y, z in lifted.tolist()
            ]

    if h_matrix is not None:
        center_x, center_y, _, _ = detection["bbox_xywh"]
        court_xy = project_pixel_to_court(center_x, center_y, h_matrix)
        detection["court_xy"] = [float(court_xy[0]), float(court_xy[1])]
        footpoint = _estimate_detection_footpoint(detection)
        foot_court_xy = project_pixel_to_court(float(footpoint["xy"][0]), float(footpoint["xy"][1]), h_matrix)
        detection["court_foot_xy"] = [float(foot_court_xy[0]), float(foot_court_xy[1])]

    if cls_id == 0:
        uniform_stats = estimate_uniform_bucket(
            frame,
            detection["bbox_xyxy"],
            keypoints_xy=detection.get("keypoints_xy"),
            keypoints_conf=detection.get("keypoints_conf"),
        )
        detection["uniform_bucket"] = uniform_stats["bucket"]
        detection["uniform_luma_mean"] = uniform_stats["luma_mean"]
        jersey_crop, jersey_sharpness = _extract_jersey_crop(frame, detection)
        if jersey_crop is not None:
            detection["_jersey_crop_bgr"] = jersey_crop
            detection["_jersey_crop_sharpness"] = jersey_sharpness
            appearance_hist = estimate_torso_color_histogram(jersey_crop)
            if appearance_hist is not None:
                detection["appearance_histogram_rgbq"] = appearance_hist

    return detection




def _copy_pose_fields_from_detection(target, pose_detection, frame, h_matrix=None):
    for key in ("keypoints_xy", "keypoints_conf", "lifted_keypoints_xyz"):
        if key in pose_detection:
            target[key] = pose_detection[key]
    target["pose_detection_confidence"] = float(pose_detection.get("confidence") or 0.0)
    target["pose_detection_source"] = "pose_enrichment"
    uniform_stats = estimate_uniform_bucket(
        frame,
        target["bbox_xyxy"],
        keypoints_xy=target.get("keypoints_xy"),
        keypoints_conf=target.get("keypoints_conf"),
    )
    target["uniform_bucket"] = uniform_stats["bucket"]
    target["uniform_luma_mean"] = uniform_stats["luma_mean"]
    jersey_crop, jersey_sharpness = _extract_jersey_crop(frame, target)
    if jersey_crop is not None:
        target["_jersey_crop_bgr"] = jersey_crop
        target["_jersey_crop_sharpness"] = jersey_sharpness
        appearance_hist = estimate_torso_color_histogram(jersey_crop)
        if appearance_hist is not None:
            target["appearance_histogram_rgbq"] = appearance_hist
    if h_matrix is not None and target.get("keypoints_xy"):
        keypoints_np = np.array(target["keypoints_xy"], dtype=float)
        lifted = lift_keypoints_to_3d(keypoints_np, h_matrix)
        target["lifted_keypoints_xyz"] = [
            [float(x), float(y), float(z)] for x, y, z in lifted.tolist()
        ]


def _bbox_center_distance_ratio(box_a, box_b):
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    center_a = np.array([(ax1 + ax2) * 0.5, (ay1 + ay2) * 0.5], dtype=np.float32)
    center_b = np.array([(bx1 + bx2) * 0.5, (by1 + by2) * 0.5], dtype=np.float32)
    distance = float(np.linalg.norm(center_a - center_b))
    norm = max(ay2 - ay1, by2 - by1, 1.0)
    return distance / norm


def _ensure_player_detection_source(detection, source, stage=None):
    sources = list(detection.get("player_detection_sources") or [])
    if source and source not in sources:
        sources.append(source)
    if sources:
        detection["player_detection_sources"] = sources
    if source and not detection.get("player_detection_source"):
        detection["player_detection_source"] = source
    stages = list(detection.get("player_proposal_stages") or [])
    if stage and stage not in stages:
        stages.append(stage)
    if stages:
        detection["player_proposal_stages"] = stages


def _serialize_player_candidates(detections):
    payload = []
    for detection in detections:
        if _is_ball_detection(detection):
            continue
        payload.append(
            {
                "track_id": int(detection["track_id"]) if detection.get("track_id") is not None else None,
                "confidence": round(float(detection.get("confidence") or 0.0), 4),
                "bbox_xyxy": [round(float(v), 3) for v in (detection.get("bbox_xyxy") or [])],
                "player_detection_source": detection.get("player_detection_source") or "unknown",
                "player_detection_sources": list(detection.get("player_detection_sources") or []),
                "player_proposal_stages": list(detection.get("player_proposal_stages") or []),
                "pose_supported": bool(detection.get("keypoints_xy")),
                "recovered_candidate": bool(detection.get("recovered_candidate")),
                "recovered_candidate_source": detection.get("recovered_candidate_source"),
            }
        )
    return payload


def _extend_raw_player_candidates(frame, detections):
    existing = list(frame.get("raw_player_detections") or [])
    existing.extend(_serialize_player_candidates(detections))
    frame["raw_player_detections"] = existing


def _match_pose_to_person_detections(person_detections, pose_result, pose_model_names, frame, h_matrix=None):
    if not pose_result or pose_result[0].boxes is None or len(pose_result[0].boxes) == 0:
        return person_detections
    pose_candidates = []
    result = pose_result[0]
    for det_idx in range(len(result.boxes)):
        pose_detection = build_detection(result, det_idx, pose_model_names, frame, h_matrix=h_matrix)
        if int(pose_detection.get("class_id", -1)) != 0:
            continue
        pose_candidates.append(pose_detection)
    used = set()
    for detection in person_detections:
        if _is_ball_detection(detection):
            continue
        detection_bbox = detection.get("bbox_xyxy") or []
        best_idx = None
        best_score = None
        best_iou = 0.0
        best_distance_ratio = None
        for idx, pose_detection in enumerate(pose_candidates):
            if idx in used:
                continue
            pose_bbox = pose_detection.get("bbox_xyxy") or []
            iou = _bbox_iou_xyxy(detection_bbox, pose_bbox)
            distance_ratio = _bbox_center_distance_ratio(detection_bbox, pose_bbox)
            if iou < PLAYER_PROPOSAL_MATCH_IOU and distance_ratio > PLAYER_PROPOSAL_CENTER_DISTANCE_RATIO:
                continue
            score = float(iou) - 0.2 * float(distance_ratio)
            if best_score is None or score > best_score:
                best_idx = idx
                best_score = score
                best_iou = float(iou)
                best_distance_ratio = float(distance_ratio)
        if best_idx is None:
            continue
        pose_detection = pose_candidates[best_idx]
        used.add(best_idx)
        _copy_pose_fields_from_detection(detection, pose_detection, frame, h_matrix=h_matrix)
        detection["pose_match"] = {
            "iou": round(best_iou, 4),
            "center_distance_ratio": round(float(best_distance_ratio or 0.0), 4),
            "confidence": round(float(pose_detection.get("confidence") or 0.0), 4),
        }
    return person_detections


def _build_person_detections(result, model_names, frame, h_matrix=None, *, source, proposal_stage, include_track_ids=True):
    detections = []
    if not result or result[0].boxes is None or len(result[0].boxes) == 0:
        return detections
    result0 = result[0]
    for det_idx in range(len(result0.boxes)):
        detection = build_detection(result0, det_idx, model_names, frame, h_matrix=h_matrix)
        if int(detection.get("class_id", -1)) != 0:
            continue
        if not include_track_ids:
            detection["track_id"] = None
        detection["player_detection_source"] = source
        detection["player_detection_sources"] = [source]
        detection["player_proposal_stages"] = [proposal_stage]
        detections.append(detection)
    return detections


def _merge_player_detections(existing_detections, proposal_detections):
    merged = list(existing_detections)
    novel_count = 0
    supported_count = 0
    for proposal in proposal_detections:
        proposal_bbox = proposal.get("bbox_xyxy") or []
        best_match = None
        best_iou = 0.0
        best_distance_ratio = None
        for detection in merged:
            if _is_ball_detection(detection):
                continue
            detection_bbox = detection.get("bbox_xyxy") or []
            iou = _bbox_iou_xyxy(detection_bbox, proposal_bbox)
            distance_ratio = _bbox_center_distance_ratio(detection_bbox, proposal_bbox)
            if iou >= PLAYER_PROPOSAL_MERGE_IOU or (
                iou >= PLAYER_PROPOSAL_MATCH_IOU and distance_ratio <= PLAYER_PROPOSAL_CENTER_DISTANCE_RATIO
            ):
                if best_match is None or iou > best_iou:
                    best_match = detection
                    best_iou = float(iou)
                    best_distance_ratio = float(distance_ratio)
        if best_match is None:
            merged.append(proposal)
            novel_count += 1
            continue
        source = proposal.get("player_detection_source") or "unknown"
        for src in proposal.get("player_detection_sources") or [source]:
            _ensure_player_detection_source(best_match, src)
        for stage in proposal.get("player_proposal_stages") or []:
            _ensure_player_detection_source(best_match, source=None, stage=stage)
        support_payload = {
            "source": source,
            "confidence": round(float(proposal.get("confidence") or 0.0), 4),
            "iou": round(best_iou, 4),
            "center_distance_ratio": round(float(best_distance_ratio or 0.0), 4),
            "bbox_xyxy": [round(float(v), 3) for v in proposal_bbox],
        }
        support_list = list(best_match.get("player_detection_support") or [])
        support_list.append(support_payload)
        best_match["player_detection_support"] = support_list
        if source == "grounded_detector":
            best_match["grounded_detector_support"] = support_payload
        supported_count += 1
    return merged, {"novel_detection_count": novel_count, "supported_detection_count": supported_count}


def _build_sam_recovered_detection(
    proposal,
    frame,
    h_matrix=None,
    *,
    backend="sam3_repo_v1",
    model_name=None,
    text_prompt=None,
):
    source_kind = str(proposal.get("source_kind") or "unknown")
    detection = {
        "track_id": None,
        "class_id": 0,
        "class_name": "person",
        "confidence": float(max(0.2, proposal.get("sam_score") or 0.0)),
        "bbox_xyxy": [float(v) for v in proposal["bbox_xyxy"]],
        "bbox_xywh": [float(v) for v in proposal["bbox_xywh"]],
        "recovered_candidate": True,
        "recovered_candidate_source": f"sam3_{source_kind}",
        "sam3_refinement": {
            "backend": backend,
            "model_name": model_name,
            "text_prompt": text_prompt,
            "sam_score": float(proposal.get("sam_score") or 0.0),
            "mask_area_px": int(proposal.get("mask_area_px") or 0),
            "mask_area_ratio": proposal.get("mask_area_ratio"),
            "source_kind": proposal.get("source_kind"),
            "source_roi_bbox_xyxy": proposal.get("source_roi_bbox_xyxy"),
            "source_iou": proposal.get("source_iou"),
            "source_trigger_reason": proposal.get("source_trigger_reason"),
        },
    }
    if h_matrix is not None:
        center_x, center_y, _, _ = detection["bbox_xywh"]
        court_xy = project_pixel_to_court(center_x, center_y, h_matrix)
        detection["court_xy"] = [float(court_xy[0]), float(court_xy[1])]
        footpoint = _estimate_detection_footpoint(detection)
        foot_court_xy = project_pixel_to_court(float(footpoint["xy"][0]), float(footpoint["xy"][1]), h_matrix)
        detection["court_foot_xy"] = [float(foot_court_xy[0]), float(foot_court_xy[1])]
    uniform_stats = estimate_uniform_bucket(frame, detection["bbox_xyxy"])
    detection["uniform_bucket"] = uniform_stats["bucket"]
    detection["uniform_luma_mean"] = uniform_stats["luma_mean"]
    return detection


def _track_frame_counts(frames, key_fn=None):
    counts = {}
    key_fn = key_fn or (lambda detection: detection.get("track_id"))
    for frame in frames:
        seen = set()
        for detection in frame.get("detections", []):
            track_key = key_fn(detection)
            if track_key is None or track_key in seen:
                continue
            counts[track_key] = counts.get(track_key, 0) + 1
            seen.add(track_key)
    return counts


def annotate_active_players(frames, video_meta):
    track_counts = _track_frame_counts(frames, key_fn=_tracklet_evidence_key)
    frame_width = int(video_meta["width"])
    frame_height = int(video_meta["height"])
    for frame in frames:
        bootstrap_context = frame.get("bootstrap_context")
        scene_prior = frame.get("scene_prior") or _build_scene_prior_contract(frame)
        frame_motion_context = _estimate_frame_motion_context(frame.get("detections", []))
        for detection in frame.get("detections", []):
            score_info = score_active_player(
                detection,
                frame_width=frame_width,
                frame_height=frame_height,
                track_frame_count=track_counts.get(_tracklet_evidence_key(detection), 1),
                bootstrap_context=bootstrap_context,
                scene_prior=scene_prior,
                frame_motion_context=frame_motion_context,
            )
            detection["active_player_score"] = score_info["score"]
            detection["on_court_score"] = score_info["on_court_score"]
            detection["on_court_candidate"] = score_info["on_court_candidate"]
            detection["active_player_candidate"] = score_info["candidate"]
            detection["active_player_reasons"] = score_info["reasons"]
    return frames


def annotate_sam_player_recovery(
    frames,
    *,
    player_recovery_backend="none",
    player_recovery_model=DEFAULT_SAM3_REPO_MODEL,
    player_recovery_prompt=DEFAULT_SAM3_TEXT_PROMPT,
    device="cpu",
):
    """Run ROI-scoped SAM recovery and refinement and mutate frame detections in place.

    This stage enriches existing YOLO detections and may append recovered player
    candidates, but it is still an artifact-generation pass rather than the
    authoritative runtime tracker handoff. Normalized `discovery_proposals` are
    materialized later from the accepted SAM outputs.
    """
    summary = {
        "backend": player_recovery_backend,
        "model_name": player_recovery_model if player_recovery_backend != "none" else None,
        "text_prompt": player_recovery_prompt if player_recovery_backend != "none" else None,
        "status": "disabled" if player_recovery_backend == "none" else "pending",
        "frame_count_with_rois": 0,
        "roi_count": 0,
        "refined_detection_count": 0,
        "recovered_detection_count": 0,
    }
    if not frames or player_recovery_backend == "none":
        return summary

    if player_recovery_backend != "sam3":
        summary["status"] = "unsupported_backend"
        return summary

    refiner = Sam3RoiRefiner(
        model_name=player_recovery_model,
        text_prompt=player_recovery_prompt,
        device=device,
    )
    last_status = "no_rois"
    for frame in frames:
        rois = build_sam_recovery_rois(frame)
        if not rois:
            continue
        summary["frame_count_with_rois"] += 1
        summary["roi_count"] += len(rois)
        result = refiner.refine(frame.get("_frame_bgr"), rois)
        last_status = result.status
        if not result.enabled or not result.proposals:
            continue
        detections = frame.get("detections", [])
        for proposal in result.proposals:
            if float(proposal.get("sam_score") or 0.0) < PLAYER_RECOVERY_MIN_SAM_SCORE:
                continue
            if proposal.get("source_kind") == "ambiguous_yolo_detection" and proposal.get("source_track_id") is not None:
                target = next((d for d in detections if d.get("track_id") == proposal.get("source_track_id")), None)
                if target is None:
                    continue
                target["sam3_refinement"] = {
                    "backend": result.backend,
                    "model_name": player_recovery_model,
                    "text_prompt": player_recovery_prompt,
                    "sam_score": proposal.get("sam_score"),
                    "mask_area_px": proposal.get("mask_area_px"),
                    "mask_area_ratio": proposal.get("mask_area_ratio"),
                    "refined_bbox_xyxy": [float(v) for v in proposal["bbox_xyxy"]],
                    "source_kind": proposal.get("source_kind"),
                    "source_merge_risk": proposal.get("source_merge_risk"),
                    "source_trigger_reason": proposal.get("source_trigger_reason"),
                }
                summary["refined_detection_count"] += 1
                continue
            if proposal.get("source_kind") in {"unexplained_grounding_dino_region", "grounding_anchor_region"}:
                detections.append(
                    _build_sam_recovered_detection(
                        proposal,
                        frame.get("_frame_bgr"),
                        frame.get("_h_matrix"),
                        backend=result.backend,
                        model_name=player_recovery_model,
                        text_prompt=player_recovery_prompt,
                    )
                )
                summary["recovered_detection_count"] += 1
        frame["detections"] = detections

    summary["status"] = last_status
    return summary


def annotate_team_appearance_consistency(frames):
    prototypes = {}
    grouped = {}
    for frame in frames:
        for detection in frame.get("detections", []):
            hist = detection.get("appearance_histogram_rgbq")
            if not hist:
                continue
            if not detection.get("active_player_candidate"):
                continue
            if float(detection.get("motion_speed_px") or 0.0) < APPEARANCE_LOW_MOTION_THRESHOLD_PX:
                continue
            reasons = detection.get("active_player_reasons") or {}
            if reasons.get("court_in_bounds") is False:
                continue
            if float(reasons.get("edge_penalty") or 0.0) >= 0.15:
                continue
            bucket = detection.get("uniform_bucket") if detection.get("uniform_bucket") in {"dark", "light"} else "unknown"
            grouped.setdefault(bucket, []).append([float(v) for v in hist])

    for bucket, samples in grouped.items():
        if not samples:
            continue
        prototypes[bucket] = np.mean(np.array(samples, dtype=np.float32), axis=0).tolist()

    for frame in frames:
        for detection in frame.get("detections", []):
            hist = detection.get("appearance_histogram_rgbq")
            if not hist:
                detection["appearance_team_distance"] = None
                detection["appearance_team_bucket"] = None
                continue
            preferred_bucket = detection.get("uniform_bucket") if detection.get("uniform_bucket") in {"dark", "light"} else None
            candidate_buckets = []
            if preferred_bucket and preferred_bucket in prototypes:
                candidate_buckets.append(preferred_bucket)
            candidate_buckets.extend(bucket for bucket in prototypes if bucket not in candidate_buckets)
            best_bucket = None
            best_distance = None
            for bucket in candidate_buckets:
                distance = histogram_intersection_distance(hist, prototypes[bucket])
                if distance is None:
                    continue
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_bucket = bucket
            detection["appearance_team_distance"] = best_distance
            detection["appearance_team_bucket"] = best_bucket
    return {
        "prototype_count": len(prototypes),
        "prototype_buckets": sorted(prototypes.keys()),
    }


def score_live_play_frame(frame):
    detections = frame.get("detections", [])
    ball_state = frame.get("ball_state")
    ball_detection = frame.get("ball_detection")
    active = [d for d in detections if d.get("active_player_candidate")]
    active_reasons = [d.get("active_player_reasons") or {} for d in active]
    on_court_active = [
        d for d, reasons in zip(active, active_reasons)
        if reasons.get("court_in_bounds") is not False
    ]
    motion_speeds = [float(d.get("motion_speed_px") or 0.0) for d in active]
    velocity_vectors = [
        [float(v[0]), float(v[1])]
        for v in (d.get("smoothed_velocity_xy") or [0.0, 0.0] for d in active)
    ]
    motion_energy_sum = float(sum(motion_speeds))
    motion_energy_median = float(np.median(motion_speeds)) if motion_speeds else 0.0
    high_motion_count = sum(1 for speed in motion_speeds if speed >= LIVE_PLAY_HIGH_MOTION_THRESHOLD_PX)
    edge_active_count = sum(
        1 for reasons in active_reasons
        if float(reasons.get("edge_penalty") or 0.0) >= 0.15
    )
    appearance_mismatch_count = sum(
        1 for detection in active
        if detection.get("appearance_team_distance") is not None
        and float(detection.get("appearance_team_distance")) > APPEARANCE_TEAM_DISTANCE_THRESHOLD
    )
    track_churn_count = sum(
        1 for detection in detections
        if detection.get("identity_repair") or detection.get("synthesized")
    )

    median_vx = float(np.median([vec[0] for vec in velocity_vectors])) if velocity_vectors else 0.0
    median_abs_vy = float(np.median([abs(vec[1]) for vec in velocity_vectors])) if velocity_vectors else 0.0
    positive_x = sum(1 for vec in velocity_vectors if vec[0] > 0.0)
    negative_x = sum(1 for vec in velocity_vectors if vec[0] < 0.0)
    x_sign_dominance = (
        max(positive_x, negative_x) / max(len(velocity_vectors), 1)
        if velocity_vectors else 0.0
    )
    camera_pan_pressure = (
        _clip01((x_sign_dominance - 0.8) / 0.2)
        * _clip01(abs(median_vx) / 120.0)
        * _clip01(1.0 - (median_abs_vy / 40.0))
    )

    on_court_presence_score = _clip01((len(on_court_active) - 1) / 3.0)
    motion_sum_score = _clip01(motion_energy_sum / LIVE_PLAY_MOTION_SUM_THRESHOLD_PX)
    motion_median_score = _clip01(motion_energy_median / LIVE_PLAY_MOTION_MEDIAN_THRESHOLD_PX)
    high_motion_score = _clip01(high_motion_count / 2.0)
    continuity_score = 1.0 - _clip01(track_churn_count / max(len(detections), 1))
    effective_ball = ball_state or ball_detection
    if ball_state:
        ball_confidence = float(ball_state.get("confidence") or 0.0)
        ball_signal_score = 1.0 if ball_state.get("state") != "missing" and ball_confidence >= BALL_MIN_LIVE_PLAY_CONFIDENCE else 0.0
    else:
        ball_confidence = float(ball_detection.get("confidence") or 0.0) if ball_detection else 0.0
        ball_signal_score = 1.0 if ball_confidence >= BALL_MIN_LIVE_PLAY_CONFIDENCE else 0.0
    motion_attenuation = max(0.05, 1.0 - 0.95 * camera_pan_pressure)
    motion_sum_score *= motion_attenuation
    motion_median_score *= motion_attenuation
    high_motion_score *= motion_attenuation

    edge_ratio = edge_active_count / max(len(active), 1) if active else 1.0
    appearance_ratio = appearance_mismatch_count / max(len(active), 1) if active else 0.0
    sparse_dead_score = 1.0 - _clip01((len(on_court_active) - 1) / 2.0)
    low_motion_dead_score = 1.0 - _clip01(motion_energy_sum / 80.0)
    edge_dead_score = _clip01(edge_ratio)
    appearance_dead_score = _clip01(appearance_ratio)
    churn_dead_score = _clip01(track_churn_count / max(len(detections), 1))

    live_play_bias = (
        0.30 * on_court_presence_score
        + 0.25 * motion_sum_score
        + 0.20 * motion_median_score
        + 0.15 * high_motion_score
        + 0.08 * continuity_score
        + 0.02 * ball_signal_score
    )
    dead_ball_bias = (
        0.25 * sparse_dead_score
        + 0.15 * low_motion_dead_score
        + 0.15 * edge_dead_score
        + 0.10 * appearance_dead_score
        + 0.10 * churn_dead_score
        + 0.35 * camera_pan_pressure
    )
    score = _clip01(0.5 + 0.5 * (live_play_bias - dead_ball_bias))
    if score >= LIVE_PLAY_SCORE_THRESHOLD:
        label = "live_play"
    elif score <= DEAD_BALL_SCORE_THRESHOLD:
        label = "dead_ball"
    else:
        label = "uncertain"
    reasons = {
        "active_candidate_count": int(len(active)),
        "on_court_active_candidate_count": int(len(on_court_active)),
        "motion_energy_sum": round(motion_energy_sum, 3),
        "motion_energy_median": round(motion_energy_median, 3),
        "high_motion_active_candidate_count": int(high_motion_count),
        "edge_active_candidate_count": int(edge_active_count),
        "appearance_mismatch_candidate_count": int(appearance_mismatch_count),
        "track_churn_count": int(track_churn_count),
        "camera_pan_pressure": round(camera_pan_pressure, 4),
        "dead_ball_bias": round(dead_ball_bias, 4),
        "live_play_bias": round(live_play_bias, 4),
        "ball_signal_present": bool(ball_signal_score),
        "ball_confidence": round(ball_confidence, 4) if effective_ball else None,
        "ball_state": ball_state.get("state") if ball_state else ("observed" if ball_detection else "missing"),
    }
    return {
        "score": round(score, 4),
        "label": label,
        "reasons": reasons,
    }


def _trailing_window(values, end_idx, window):
    start_idx = max(0, end_idx - window + 1)
    return values[start_idx : end_idx + 1]


def _window_average(values):
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _window_count(values, predicate):
    return sum(1 for value in values if predicate(float(value)))


def _dominant_live_play_signal(frames):
    if not frames:
        return "no_frames"
    ball_present_count = sum(
        1 for frame in frames
        if (frame.get("live_play_reasons") or {}).get("ball_signal_present")
    )
    labels = [frame.get("live_play_label") for frame in frames]
    if labels and all(label == "live_play" for label in labels):
        if ball_present_count >= max(2, len(frames) // 3):
            return "ball_and_on_court_motion"
        motion_total = sum((frame.get("live_play_reasons") or {}).get("motion_energy_sum", 0.0) for frame in frames)
        on_court_total = sum((frame.get("live_play_reasons") or {}).get("on_court_active_candidate_count", 0) for frame in frames)
        return "on_court_motion" if motion_total >= on_court_total * 20.0 else "stable_on_court_presence"
    if labels and all(label == "dead_ball" for label in labels):
        sparse_total = sum((frame.get("live_play_reasons") or {}).get("on_court_active_candidate_count", 0) for frame in frames)
        motion_total = sum((frame.get("live_play_reasons") or {}).get("motion_energy_sum", 0.0) for frame in frames)
        return "low_on_court_motion" if motion_total <= sparse_total * 12.0 else "edge_heavy_idle_scene"
    return "mixed_evidence"


def annotate_continuity_segments(frames):
    if not frames:
        return {"segments": []}

    segments = []
    current_segment_id = 0
    segment_start_idx = 0

    for idx, frame in enumerate(frames):
        if idx == 0:
            frame["continuity_segment_id"] = current_segment_id
            frame["discontinuity_score"] = 0.0
            frame["discontinuity_label"] = "continuous"
            frame["discontinuity_reasons"] = {
                "visual_delta": 0.0,
                "track_churn": 0.0,
                "detection_count_delta": 0,
            }
            for detection in frame.get("detections", []):
                detection["continuity_segment_id"] = current_segment_id
            continue

        previous = frames[idx - 1]
        visual_delta = _frame_visual_delta(
            previous.get("_frame_visual_signature"),
            frame.get("_frame_visual_signature"),
        )
        prev_ids = _frame_track_ids(previous)
        curr_ids = _frame_track_ids(frame)
        union_ids = prev_ids | curr_ids
        track_overlap = (len(prev_ids & curr_ids) / len(union_ids)) if union_ids else 1.0
        track_churn = 1.0 - track_overlap
        detection_count_delta = abs(
            len(frame.get("detections", [])) - len(previous.get("detections", []))
        )
        detection_count_score = _clip01(detection_count_delta / 6.0)
        visual_score = _clip01(visual_delta / DISCONTINUITY_VISUAL_DELTA_THRESHOLD)
        churn_score = _clip01(track_churn / DISCONTINUITY_TRACK_CHURN_THRESHOLD)
        discontinuity_score = (
            0.60 * visual_score
            + 0.30 * churn_score
            + 0.10 * detection_count_score
        )
        is_discontinuous = (
            discontinuity_score >= DISCONTINUITY_SCORE_THRESHOLD
            or (
                visual_delta >= DISCONTINUITY_VISUAL_DELTA_THRESHOLD
                and track_churn >= 0.55
            )
        )

        if is_discontinuous:
            previous_segment_frames = frames[segment_start_idx:idx]
            segments.append({
                "segment_id": current_segment_id,
                "start_frame": int(previous_segment_frames[0]["frame_idx"]),
                "end_frame": int(previous_segment_frames[-1]["frame_idx"]),
                "start_t_ms": int(previous_segment_frames[0]["t_ms"]),
                "end_t_ms": int(previous_segment_frames[-1]["t_ms"]),
                "frame_count": len(previous_segment_frames),
            })
            current_segment_id += 1
            segment_start_idx = idx

        frame["continuity_segment_id"] = current_segment_id
        frame["discontinuity_score"] = round(float(discontinuity_score), 4)
        frame["discontinuity_label"] = "discontinuity" if is_discontinuous else "continuous"
        frame["discontinuity_reasons"] = {
            "visual_delta": round(float(visual_delta), 4),
            "track_churn": round(float(track_churn), 4),
            "detection_count_delta": int(detection_count_delta),
        }
        for detection in frame.get("detections", []):
            detection["continuity_segment_id"] = current_segment_id

    final_segment_frames = frames[segment_start_idx:]
    if final_segment_frames:
        segments.append({
            "segment_id": current_segment_id,
            "start_frame": int(final_segment_frames[0]["frame_idx"]),
            "end_frame": int(final_segment_frames[-1]["frame_idx"]),
            "start_t_ms": int(final_segment_frames[0]["t_ms"]),
            "end_t_ms": int(final_segment_frames[-1]["t_ms"]),
            "frame_count": len(final_segment_frames),
        })

    for frame in frames:
        frame.pop("_frame_visual_signature", None)

    return {"segments": segments}


def annotate_bootstrap_contexts(
    frames,
    *,
    bootstrap_foreground_backend="none",
    bootstrap_foreground_model=DEFAULT_GROUNDING_DINO_MODEL,
    bootstrap_foreground_prompt=DEFAULT_GROUNDING_DINO_PROMPT,
    device="cpu",
):
    """Attach per-segment bootstrap and grounding contexts to frames.

    The current runtime still consumes these low-level contexts directly for
    grounded reruns and collapse handling. The higher-level `scene_prior`
    contract is emitted later as a normalized artifact view over the same state.
    """
    if not frames:
        return {"backend": bootstrap_foreground_backend, "contexts": []}

    contexts = []
    context_by_bootstrap_key = {}
    bootstrapper = None
    for frame in frames:
        segment_id = frame.get("continuity_segment_id")
        camera_pose_segment_id = frame.get("camera_pose_segment_id", segment_id)
        bootstrap_key = (segment_id, camera_pose_segment_id)
        if bootstrap_key in context_by_bootstrap_key:
            payload = context_by_bootstrap_key[bootstrap_key]
            frame["bootstrap_context"] = payload
            frame["grounding_context"] = json.loads(json.dumps(payload.get("grounding_context") or {}))
            continue

        if bootstrap_foreground_backend == "grounding_dino":
            source_frame = frame.get("_frame_bgr")
            if source_frame is not None:
                if bootstrapper is None:
                    bootstrapper = GroundingDinoBootstrapper(
                        model_name=bootstrap_foreground_model,
                        text_prompt=bootstrap_foreground_prompt,
                        device=device,
                    )
                payload = bootstrapper.run_on_frame(
                    source_frame,
                    frame_idx=frame.get("frame_idx", 0),
                ).to_payload()
            else:
                payload = {
                    "enabled": False,
                    "status": "frame_unavailable",
                    "backend": bootstrap_foreground_backend,
                    "model_name": bootstrap_foreground_model,
                    "frame_idx": frame.get("frame_idx", 0),
                }
        else:
            payload = {
                "enabled": False,
                "status": "disabled",
                "backend": bootstrap_foreground_backend,
                "model_name": None,
                "frame_idx": frame.get("frame_idx", 0),
            }

        if int(segment_id or 0) == 0 and int(camera_pose_segment_id or 0) == 0:
            trigger_reason = "initial"
        elif segment_id is not None and segment_id != 0 and camera_pose_segment_id == segment_id:
            trigger_reason = "discontinuity"
        elif camera_pose_segment_id is not None:
            trigger_reason = "camera_pose_segment"
        else:
            trigger_reason = "discontinuity"
        grounding_context = {
            "enabled": False,
            "pipeline_order": ["grounding_dino", "sam", "yolo"],
            "trigger_reason": trigger_reason,
            "camera_pose_segment_id": camera_pose_segment_id,
            "source_backend": payload.get("backend"),
            "source_status": payload.get("status"),
            "proposal_regions": [],
            "sam_roi_policy": "unexplained_grounding_dino_regions_and_ambiguous_yolo",
            "yolo_search_policy": "full_frame",
            "yolo_search_region_mask_grid": None,
            "yolo_search_region_mask_shape": None,
            "yolo_search_region_bbox_xyxy": None,
            "should_rerun_yolo": False,
        }
        image_width = int(payload.get("image_width") or 0)
        image_height = int(payload.get("image_height") or 0)
        mask_grid = payload.get("mask_grid")
        mask_shape = payload.get("mask_shape")
        if payload.get("enabled") and payload.get("status") == "ready" and mask_grid and image_width > 0 and image_height > 0:
            full_mask = bootstrap_mask_to_image(payload)
            component_regions = []
            should_rerun_yolo = False
            if full_mask is not None:
                for component in component_boxes_from_mask(full_mask):
                    bbox = component["bbox_xyxy"]
                    bbox_width = float(bbox[2] - bbox[0])
                    bbox_height = float(bbox[3] - bbox[1])
                    should_rerun_yolo = should_rerun_yolo or (
                        bbox_width >= GROUNDED_YOLO_MIN_SIDE_PX and bbox_height >= GROUNDED_YOLO_MIN_SIDE_PX
                    )
                    component_regions.append({
                        "kind": "grounding_dino_play_region_component",
                        "bbox_xyxy": [float(v) for v in bbox],
                        "confidence": round(float(component.get("area_ratio") or 0.0), 4),
                        "area_ratio": component.get("area_ratio"),
                    })
            grounding_context.update({
                "enabled": True,
                "proposal_regions": component_regions,
                "yolo_search_policy": "mask_outside_play_region",
                "yolo_search_region_mask_grid": mask_grid,
                "yolo_search_region_mask_shape": mask_shape,
                "yolo_search_region_bbox_xyxy": payload.get("foreground_bbox_xyxy"),
                "should_rerun_yolo": should_rerun_yolo,
            })
        payload["continuity_segment_id"] = segment_id
        payload["camera_pose_segment_id"] = camera_pose_segment_id
        payload["grounding_context"] = grounding_context
        context_by_bootstrap_key[bootstrap_key] = payload
        frame["bootstrap_context"] = payload
        frame["grounding_context"] = json.loads(json.dumps(grounding_context))
        contexts.append(payload)

    return {
        "backend": bootstrap_foreground_backend,
        "contexts": contexts,
    }




def _scene_prior_status(bootstrap_context, grounding_context):
    if grounding_context.get("enabled") and (
        grounding_context.get("proposal_regions")
        or grounding_context.get("yolo_search_region_mask_grid")
    ):
        return "ready"
    return str(
        grounding_context.get("source_status")
        or bootstrap_context.get("status")
        or "inactive"
    )


def _build_scene_prior_contract(frame):
    bootstrap_context = frame.get("bootstrap_context") or {}
    grounding_context = frame.get("grounding_context") or {}
    region_mask_grid = grounding_context.get("yolo_search_region_mask_grid")
    if region_mask_grid is None:
        region_mask_grid = bootstrap_context.get("mask_grid")
    region_mask_shape = grounding_context.get("yolo_search_region_mask_shape")
    if region_mask_shape is None:
        region_mask_shape = bootstrap_context.get("mask_shape")
    return {
        "frame_idx": int(frame.get("frame_idx", 0)),
        "continuity_segment_id": frame.get("continuity_segment_id"),
        "prior_status": _scene_prior_status(bootstrap_context, grounding_context),
        "region_mask_shape": json.loads(json.dumps(region_mask_shape)),
        "region_mask_grid": json.loads(json.dumps(region_mask_grid)),
        "proposal_regions": json.loads(json.dumps(grounding_context.get("proposal_regions") or [])),
        "source_model": bootstrap_context.get("model_name"),
        "source_backend": bootstrap_context.get("backend"),
        "trigger_reason": grounding_context.get("trigger_reason"),
        "yolo_search_policy": grounding_context.get("yolo_search_policy"),
        "yolo_search_region_bbox_xyxy": json.loads(json.dumps(grounding_context.get("yolo_search_region_bbox_xyxy"))),
        "should_rerun_yolo": bool(grounding_context.get("should_rerun_yolo")),
        "yolo_rerun_applied": bool(grounding_context.get("yolo_rerun_applied")),
        "collapse_triggered": bool(grounding_context.get("collapse_triggered")),
        "collapse_trigger_frame_idx": grounding_context.get("collapse_trigger_frame_idx"),
    }


def _build_discovery_proposals(frame):
    proposals = []
    for detection in frame.get("detections", []):
        sam3_refinement = detection.get("sam3_refinement") or {}
        sam_score = sam3_refinement.get("sam_score")
        if sam_score is None:
            continue
        bbox_xyxy = sam3_refinement.get("refined_bbox_xyxy") or detection.get("bbox_xyxy")
        if not bbox_xyxy:
            continue
        source_kind = sam3_refinement.get("source_kind")
        source_region = {"kind": source_kind or "sam3_roi", "bbox_xyxy": None}
        if sam3_refinement.get("source_roi_bbox_xyxy"):
            source_region["bbox_xyxy"] = [float(v) for v in sam3_refinement["source_roi_bbox_xyxy"]]
        proposal = {
            "frame_idx": int(frame.get("frame_idx", 0)),
            "entity_type": "player",
            "bbox_xyxy": [float(v) for v in bbox_xyxy],
            "mask_or_polygon": {
                "kind": "bbox_proxy",
                "bbox_xyxy": [float(v) for v in bbox_xyxy],
            },
            "score": round(float(sam_score), 4),
            "source_model": sam3_refinement.get("model_name") or sam3_refinement.get("backend"),
            "source_prompt": sam3_refinement.get("text_prompt"),
            "source_region": source_region,
            "source_backend": sam3_refinement.get("backend"),
            "proposal_role": "recovery" if detection.get("recovered_candidate") else "refinement",
            "mask_area_px": sam3_refinement.get("mask_area_px"),
            "mask_area_ratio": sam3_refinement.get("mask_area_ratio"),
            "anchor_region_kind": source_kind,
            "source_trigger_reason": sam3_refinement.get("source_trigger_reason"),
        }
        if detection.get("track_id") is not None:
            proposal["source_track_id"] = int(detection["track_id"])
        proposals.append(proposal)
    return proposals


def annotate_scene_discovery_contracts(frames):
    """Normalize staged discovery evidence into explicit artifact contracts.

    This function is intentionally post-hoc today: it converts bootstrap,
    grounding, and accepted SAM outputs into the explicit `scene_prior` and
    `discovery_proposals` contracts used for auditability and downstream
    integration work. Earlier runtime stages are not yet driven by these
    normalized contracts.
    """
    summary = {
        "kind": "scene_discovery_tracking_v1",
        "pipeline_order": ["grounding_dino", "sam", "yolo"],
        "scene_prior_frame_count": 0,
        "scene_prior_ready_frame_count": 0,
        "scene_prior_trigger_reasons": [],
        "discovery_proposal_frame_count": 0,
        "discovery_proposal_count": 0,
        "discovery_refinement_count": 0,
        "discovery_recovery_count": 0,
        "discovery_entity_types": [],
    }
    trigger_reasons = set()
    entity_types = set()
    for frame in frames:
        scene_prior = _build_scene_prior_contract(frame)
        discovery_proposals = _build_discovery_proposals(frame)
        frame["scene_prior"] = scene_prior
        frame["discovery_proposals"] = discovery_proposals
        summary["scene_prior_frame_count"] += 1
        if scene_prior.get("prior_status") == "ready":
            summary["scene_prior_ready_frame_count"] += 1
        if scene_prior.get("trigger_reason"):
            trigger_reasons.add(str(scene_prior["trigger_reason"]))
        if discovery_proposals:
            summary["discovery_proposal_frame_count"] += 1
            summary["discovery_proposal_count"] += len(discovery_proposals)
        for proposal in discovery_proposals:
            entity_types.add(str(proposal.get("entity_type") or "unknown"))
            if proposal.get("proposal_role") == "recovery":
                summary["discovery_recovery_count"] += 1
            else:
                summary["discovery_refinement_count"] += 1
    summary["scene_prior_trigger_reasons"] = sorted(trigger_reasons)
    summary["discovery_entity_types"] = sorted(entity_types)
    return summary


def _increment_counter(counter, key, amount=1):
    counter[str(key)] = int(counter.get(str(key), 0)) + int(amount)


def _build_staged_perception_summary(
    frames,
    *,
    bootstrap_summary,
    grounding_summary,
    collapse_summary,
    scene_discovery_summary,
    player_recovery_summary,
    engagement_summary,
    temporal_summary=None,
    identity_hypotheses,
):
    pipeline_order = (
        scene_discovery_summary.get("pipeline_order")
        or grounding_summary.get("pipeline_order")
        or ["grounding_dino", "sam", "yolo"]
    )
    bootstrap_contexts = bootstrap_summary.get("contexts") or []
    bootstrap_ready_contexts = [
        context for context in bootstrap_contexts
        if context.get("enabled") and context.get("status") == "ready"
    ]
    bootstrap_reasons = sorted({
        str(context.get("reason") or (context.get("grounding_context") or {}).get("trigger_reason"))
        for context in bootstrap_contexts
        if context.get("reason") or (context.get("grounding_context") or {}).get("trigger_reason")
    })

    scene_prior_supported_detection_count = 0
    scene_prior_strengthened_detection_count = 0
    recovered_candidate_count = 0
    recovered_candidate_active_count = 0
    repaired_recovery_count = 0
    synthesized_identity_bridge_count = 0
    ambiguous_identity_detection_count = 0
    ambiguous_identity_frame_count = 0
    identity_option_detection_count = 0
    identity_option_total = 0
    reranked_identity_delta_detection_count = 0
    grounded_rerun_applied_frame_count = 0
    grounded_rerun_detection_count = 0
    collapse_triggered_frame_count = 0
    raw_player_proposal_count = 0
    player_detection_source_counts = {}
    engagement_state_counts = {}
    tracklet_temporal_state_counts = {}
    ball_state_source_counts = {}
    ball_detection_source_counts = {}
    ball_state_counts = {}
    ball_predictive_trigger_count = 0
    ball_fallback_trigger_count = 0
    ball_retro_accepted_count = 0
    ball_sam_trigger_count = 0
    ball_sam_accepted_count = 0
    ball_sam_status_counts = {}

    for frame in frames:
        grounding_context = frame.get("grounding_context") or {}
        frame_has_ambiguous_identity = False
        raw_player_proposal_count += len(frame.get("raw_player_detections") or [])
        if grounding_context.get("yolo_rerun_applied"):
            grounded_rerun_applied_frame_count += 1
            grounded_rerun_detection_count += int(grounding_context.get("yolo_rerun_detection_count") or 0)
        if grounding_context.get("collapse_triggered"):
            collapse_triggered_frame_count += 1

        if (frame.get("ball_predictive_search") or {}).get("triggered"):
            ball_predictive_trigger_count += 1
        if (frame.get("ball_fallback") or {}).get("triggered"):
            ball_fallback_trigger_count += 1
        if (frame.get("ball_retro_search") or {}).get("accepted"):
            ball_retro_accepted_count += 1
        ball_sam = frame.get("ball_sam") or {}
        if ball_sam.get("triggered"):
            ball_sam_trigger_count += 1
        if ball_sam.get("accepted"):
            ball_sam_accepted_count += 1
        _increment_counter(ball_sam_status_counts, ball_sam.get("status") or "disabled")

        ball_state = frame.get("ball_state") or {}
        _increment_counter(ball_state_counts, ball_state.get("state") or "missing")
        _increment_counter(ball_state_source_counts, ball_state.get("source") or "unknown")
        ball_detection = frame.get("ball_detection") or {}
        if ball_state.get("state") == "observed":
            _increment_counter(
                ball_detection_source_counts,
                ball_detection.get("ball_detection_source") or ball_detection.get("source") or "unknown",
            )

        for detection in frame.get("detections", []):
            _increment_counter(player_detection_source_counts, detection.get("player_detection_source") or "unknown")
            _increment_counter(engagement_state_counts, detection.get("engagement_state") or "unknown")
            _increment_counter(tracklet_temporal_state_counts, detection.get("tracklet_temporal_state") or "unknown")
            reasons = detection.get("active_player_reasons") or {}
            if float(reasons.get("scene_center_prior") or 0.0) > 0.0 or float(reasons.get("scene_foot_prior") or 0.0) > 0.0:
                scene_prior_supported_detection_count += 1
            if (
                float(reasons.get("effective_foreground_prior") or 0.0) > float(reasons.get("bootstrap_foreground_prior") or 0.0)
                or float(reasons.get("effective_foot_prior") or 0.0) > float(reasons.get("bootstrap_foot_prior") or 0.0)
            ):
                scene_prior_strengthened_detection_count += 1
            if detection.get("recovered_candidate"):
                recovered_candidate_count += 1
                if detection.get("active_player_candidate"):
                    recovered_candidate_active_count += 1
            identity_option_count = int(detection.get("identity_option_count") or 0)
            if identity_option_count > 0:
                identity_option_detection_count += 1
                identity_option_total += identity_option_count
            if detection.get("identity_is_ambiguous"):
                ambiguous_identity_detection_count += 1
                frame_has_ambiguous_identity = True
            if (
                detection.get("identity_track_id") is not None
                and detection.get("identity_best_canonical_track_id") is not None
                and int(detection.get("identity_track_id")) != int(detection.get("identity_best_canonical_track_id"))
            ):
                reranked_identity_delta_detection_count += 1
            if detection.get("identity_track_source") == "repaired_recovery":
                repaired_recovery_count += 1
            if detection.get("synthesized") and (detection.get("identity_repair") or {}).get("kind") == "short_gap_identity_bridge":
                synthesized_identity_bridge_count += 1
        if frame_has_ambiguous_identity:
            ambiguous_identity_frame_count += 1

    help_signal_count = sum(
        int(value > 0)
        for value in [
            scene_prior_strengthened_detection_count,
            grounded_rerun_applied_frame_count,
            repaired_recovery_count,
            ball_predictive_trigger_count + ball_fallback_trigger_count + ball_retro_accepted_count + ball_sam_accepted_count,
        ]
    )

    return {
        "kind": "staged_perception_summary_v1",
        "pipeline_order": list(pipeline_order),
        "clip_helped": bool(help_signal_count > 0),
        "help_signal_count": int(help_signal_count),
        "bootstrap": {
            "context_count": len(bootstrap_contexts),
            "ready_context_count": len(bootstrap_ready_contexts),
            "trigger_reasons": bootstrap_reasons,
            "backend": bootstrap_summary.get("backend"),
        },
        "scene_prior": {
            "ready_frame_count": int(scene_discovery_summary.get("scene_prior_ready_frame_count") or 0),
            "trigger_reasons": list(scene_discovery_summary.get("scene_prior_trigger_reasons") or []),
            "supported_detection_count": int(scene_prior_supported_detection_count),
            "strengthened_detection_count": int(scene_prior_strengthened_detection_count),
        },
        "player": {
            "raw_proposal_count": int(raw_player_proposal_count),
            "observed_detection_source_counts": player_detection_source_counts,
            "engagement_state_counts": engagement_state_counts,
            "tracklet_temporal_state_counts": tracklet_temporal_state_counts,
            "adjacent_or_other_detection_count": int((engagement_summary or {}).get("adjacent_or_other_detection_count") or 0),
            "engaged_detection_count": int((engagement_summary or {}).get("engaged_detection_count") or 0),
            "tracklet_promotion_ready_detection_count": int((temporal_summary or {}).get("promotion_ready_detection_count") or 0),
            "tracklet_demotion_ready_detection_count": int((temporal_summary or {}).get("demotion_ready_detection_count") or 0),
        },
        "grounded_yolo": {
            "requested_frame_count": int(grounding_summary.get("frame_count_with_grounding") or 0),
            "rerun_frame_count": int(grounding_summary.get("frame_count_rerun") or 0),
            "rerun_applied_frame_count": int(grounded_rerun_applied_frame_count),
            "rerun_detection_count": int(grounded_rerun_detection_count),
            "novel_detection_count": int(grounding_summary.get("novel_detection_count") or 0),
            "supported_detection_count": int(grounding_summary.get("supported_detection_count") or 0),
            "collapse_triggered_segment_count": len(collapse_summary.get("triggered_segment_ids") or []),
            "collapse_triggered_frame_count": int(collapse_triggered_frame_count),
        },
        "sam": {
            "status": player_recovery_summary.get("status"),
            "frame_count_with_rois": int(player_recovery_summary.get("frame_count_with_rois") or 0),
            "refined_detection_count": int(player_recovery_summary.get("refined_detection_count") or 0),
            "recovered_detection_count": int(player_recovery_summary.get("recovered_detection_count") or 0),
            "recovered_candidate_count": int(recovered_candidate_count),
            "recovered_candidate_active_count": int(recovered_candidate_active_count),
        },
        "identity": {
            "hypothesis_group_count": int(identity_hypotheses.get("group_count") or 0),
            "selected_link_count": int(identity_hypotheses.get("selected_link_count") or 0),
            "track_identity_option_count": int(identity_hypotheses.get("track_identity_option_count") or 0),
            "ambiguous_track_identity_option_count": int(identity_hypotheses.get("ambiguous_track_identity_option_count") or 0),
            "identity_option_detection_count": int(identity_option_detection_count),
            "identity_option_total": int(identity_option_total),
            "ambiguous_detection_count": int(ambiguous_identity_detection_count),
            "ambiguous_frame_count": int(ambiguous_identity_frame_count),
            "best_canonical_delta_detection_count": int(reranked_identity_delta_detection_count),
            "repaired_recovery_count": int(repaired_recovery_count),
            "synthesized_bridge_count": int(synthesized_identity_bridge_count),
        },
        "ball": {
            "state_counts": ball_state_counts,
            "state_source_counts": ball_state_source_counts,
            "observed_detection_source_counts": ball_detection_source_counts,
            "predictive_trigger_count": int(ball_predictive_trigger_count),
            "fallback_trigger_count": int(ball_fallback_trigger_count),
            "retro_backfilled_frame_count": int(ball_retro_accepted_count),
            "sam_trigger_count": int(ball_sam_trigger_count),
            "sam_accepted_count": int(ball_sam_accepted_count),
            "sam_status_counts": ball_sam_status_counts,
        },
    }


def _apply_grounding_mask(frame_bgr, grounding_context):
    if frame_bgr is None:
        return None, False
    if not grounding_context or grounding_context.get("yolo_search_policy") != "mask_outside_play_region":
        return frame_bgr, False
    mask_grid = grounding_context.get("yolo_search_region_mask_grid")
    if not mask_grid:
        return frame_bgr, False
    mask = np.array(mask_grid, dtype=np.uint8)
    if mask.ndim != 2:
        return frame_bgr, False
    height, width = frame_bgr.shape[:2]
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    if mask.shape[:2] != (height, width):
        return frame_bgr, False
    masked = np.zeros_like(frame_bgr)
    masked[mask > 0] = frame_bgr[mask > 0]
    return masked, True


def rerun_grounded_person_detections(
    frames,
    *,
    player_model_name,
    device,
    conf_threshold,
):
    summary = {
        "enabled": False,
        "status": "disabled",
        "pipeline_order": ["grounding_dino", "sam", "yolo"],
        "frame_count_with_grounding": 0,
        "frame_count_rerun": 0,
        "detection_count": 0,
        "novel_detection_count": 0,
        "supported_detection_count": 0,
        "search_policy": "full_frame",
    }
    if not frames:
        return summary

    rerun_frames = [
        frame for frame in frames
        if (frame.get("grounding_context") or {}).get("should_rerun_yolo")
    ]
    summary["frame_count_with_grounding"] = len(rerun_frames)
    if not rerun_frames:
        summary["status"] = "no_grounded_segments"
        return summary

    player_model = YOLO(player_model_name)
    summary["enabled"] = True
    summary["search_policy"] = "mask_outside_play_region"
    for frame in frames:
        grounding_context = frame.get("grounding_context") or {}
        run_frame, was_masked = _apply_grounding_mask(frame.get("_frame_bgr"), grounding_context)
        if not grounding_context.get("should_rerun_yolo"):
            continue
        player_results = player_model.predict(
            run_frame,
            classes=[0],
            conf=conf_threshold,
            device=device,
            verbose=False,
        )
        h_matrix = _h_matrix_as_array(frame.get("_h_matrix"))
        detections = _build_person_detections(
            player_results,
            player_model.names,
            frame.get("_frame_bgr"),
            h_matrix=h_matrix,
            source="grounded_detector",
            proposal_stage="grounded_rerun",
            include_track_ids=False,
        )
        for detection in detections:
            detection["continuity_segment_id"] = frame.get("continuity_segment_id")
        merged, merge_summary = _merge_player_detections(frame.get("detections", []), detections)
        frame["detections"] = merged
        _extend_raw_player_candidates(frame, detections)
        grounding_context["yolo_rerun_applied"] = True
        grounding_context["yolo_rerun_detection_count"] = len(detections)
        grounding_context["yolo_rerun_novel_detection_count"] = int(merge_summary["novel_detection_count"])
        grounding_context["yolo_rerun_supported_detection_count"] = int(merge_summary["supported_detection_count"])
        grounding_context["yolo_rerun_masked"] = bool(was_masked)
        summary["frame_count_rerun"] += 1
        summary["detection_count"] += len(detections)
        summary["novel_detection_count"] += int(merge_summary["novel_detection_count"])
        summary["supported_detection_count"] += int(merge_summary["supported_detection_count"])
    summary["status"] = "ready"
    return summary


def mark_tracking_collapse_reground(frames):
    summary = {
        "enabled": True,
        "trigger": "detection_collapse",
        "live_score_threshold": GROUNDING_COLLAPSE_LIVE_SCORE_THRESHOLD,
        "min_on_court_count": GROUNDING_COLLAPSE_MIN_ON_COURT_COUNT,
        "streak_frames": GROUNDING_COLLAPSE_STREAK_FRAMES,
        "triggered_segment_ids": [],
        "triggered_frame_indices": [],
    }
    if not frames:
        summary["enabled"] = False
        return summary

    streak = 0
    triggered_segments = set()
    triggered_frames = []
    for frame in frames:
        live_score = float(frame.get("live_play_score") or 0.0)
        on_court_count = int((frame.get("live_play_reasons") or {}).get("on_court_active_candidate_count", 0))
        if (
            live_score >= GROUNDING_COLLAPSE_LIVE_SCORE_THRESHOLD
            and on_court_count < GROUNDING_COLLAPSE_MIN_ON_COURT_COUNT
            and frame.get("discontinuity_label") != "discontinuity"
        ):
            streak += 1
        else:
            streak = 0
        frame["grounding_collapse_streak"] = int(streak)
        if streak < GROUNDING_COLLAPSE_STREAK_FRAMES:
            continue
        segment_id = frame.get("continuity_segment_id")
        if segment_id in triggered_segments:
            continue
        triggered_segments.add(segment_id)
        triggered_frames.append(int(frame.get("frame_idx", 0)))
        for candidate_frame in frames:
            if candidate_frame.get("continuity_segment_id") != segment_id:
                continue
            grounding_context = candidate_frame.get("grounding_context") or {}
            if not grounding_context:
                continue
            grounding_context["collapse_triggered"] = True
            grounding_context["collapse_trigger_frame_idx"] = int(frame.get("frame_idx", 0))
            grounding_context["collapse_live_play_score"] = round(live_score, 4)
            grounding_context["collapse_on_court_count"] = int(on_court_count)
            grounding_context["should_rerun_yolo"] = bool(
                grounding_context.get("enabled") and grounding_context.get("proposal_regions")
            )
            if grounding_context.get("trigger_reason") not in {"initial", "discontinuity"}:
                grounding_context["trigger_reason"] = "tracking_collapse"

    summary["triggered_segment_ids"] = sorted(int(v) for v in triggered_segments)
    summary["triggered_frame_indices"] = triggered_frames
    return summary


def annotate_live_play(frames, video_meta=None):
    if not frames:
        return {"segments": []}

    raw_scores = []
    raw_labels = []
    for frame in frames:
        live_info = score_live_play_frame(frame)
        frame["live_play_score"] = live_info["score"]
        frame["live_play_label"] = live_info["label"]
        frame["live_play_reasons"] = live_info["reasons"]
        raw_scores.append(live_info["score"])
        raw_labels.append(live_info["label"])

    state = "dead_ball"
    for idx, frame in enumerate(frames):
        enter_window = _trailing_window(raw_scores, idx, LIVE_PLAY_ENTER_WINDOW)
        exit_window = _trailing_window(raw_scores, idx, LIVE_PLAY_EXIT_WINDOW)
        dead_window = _trailing_window(raw_scores, idx, LIVE_PLAY_ENTER_WINDOW)
        sustain_live = (
            _window_average(enter_window) >= LIVE_PLAY_SCORE_THRESHOLD
            or _window_count(enter_window, lambda score: score >= 0.55) >= 2
        )
        enter_live = (
            len(enter_window) >= LIVE_PLAY_ENTER_MIN_POSITIVE
            and _window_average(enter_window) >= LIVE_PLAY_SCORE_THRESHOLD
            and _window_count(enter_window, lambda score: score >= 0.55) >= LIVE_PLAY_ENTER_MIN_POSITIVE
        )
        exit_live = (
            len(exit_window) >= LIVE_PLAY_EXIT_MIN_NEGATIVE
            and _window_average(exit_window) <= DEAD_BALL_SCORE_THRESHOLD
            and _window_count(exit_window, lambda score: score <= 0.45) >= LIVE_PLAY_EXIT_MIN_NEGATIVE
        )
        enter_dead = (
            len(dead_window) >= 3
            and _window_average(dead_window) <= DEAD_BALL_SCORE_THRESHOLD
            and _window_count(dead_window, lambda score: score <= 0.45) >= 3
        )

        if state == "live_play":
            if exit_live:
                state = "dead_ball"
            elif not sustain_live:
                state = "uncertain"
        else:
            if enter_live:
                state = "live_play"
            elif enter_dead:
                state = "dead_ball"
            else:
                state = "uncertain"

        frame["live_play_state_label"] = state

    segments = []
    segment_start = 0
    for idx in range(1, len(frames) + 1):
        if idx < len(frames) and frames[idx]["live_play_label"] == frames[segment_start]["live_play_label"]:
            continue
        segment_frames = frames[segment_start:idx]
        scores = [float(frame.get("live_play_score") or 0.0) for frame in segment_frames]
        segments.append({
            "start_frame": int(segment_frames[0]["frame_idx"]),
            "end_frame": int(segment_frames[-1]["frame_idx"]),
            "start_t_ms": int(segment_frames[0]["t_ms"]),
            "end_t_ms": int(segment_frames[-1]["t_ms"]),
            "label": segment_frames[0]["live_play_label"],
            "mean_score": round(_window_average(scores), 4),
            "peak_score": round(max(scores) if scores else 0.0, 4),
            "frame_count": len(segment_frames),
            "reasons_summary": {
                "dominant_signal": _dominant_live_play_signal(segment_frames),
            },
        })
        segment_start = idx

    for frame in frames:
        segment = next(
            (
                seg for seg in segments
                if seg["start_frame"] <= frame["frame_idx"] <= seg["end_frame"]
            ),
            None,
        )
        if segment is not None:
            frame["live_play_segment_label"] = segment["label"]
            frame["live_play_segment_start_frame"] = segment["start_frame"]
            frame["live_play_segment_end_frame"] = segment["end_frame"]
            frame["live_play_segment_duration_ms"] = int(segment["end_t_ms"] - segment["start_t_ms"])

    return {"segments": segments}


def smooth_track_motion(frames, video_meta):
    fps = float(video_meta.get("fps") or 30.0)
    dt = 1.0 / max(fps, 1.0)
    track_filters = {}
    last_sizes = {}

    for frame in frames:
        for detection in frame.get("detections", []):
            track_id = detection.get("track_id")
            bbox_xywh = detection.get("bbox_xywh")
            if track_id is None or not bbox_xywh or len(bbox_xywh) < 4:
                detection["motion_speed_px"] = 0.0
                continue

            center_x, center_y, width, height = [float(v) for v in bbox_xywh]
            kalman = track_filters.get(track_id)
            if kalman is None:
                kalman = KalmanTrack2D(center_x, center_y)
                track_filters[track_id] = kalman
                kalman_state = kalman.state.copy()
            else:
                kalman.predict(dt)
                kalman_state = kalman.update(np.array([center_x, center_y], dtype=float))

            vel_x = float(kalman_state[2])
            vel_y = float(kalman_state[3])
            motion_speed = float(np.linalg.norm([vel_x, vel_y]))
            last_width, last_height = last_sizes.get(track_id, (width, height))
            last_sizes[track_id] = (width, height)

            detection["smoothed_center_xy"] = [float(kalman_state[0]), float(kalman_state[1])]
            detection["smoothed_velocity_xy"] = [vel_x, vel_y]
            detection["motion_speed_px"] = motion_speed
            detection["smoothed_bbox_xywh"] = [
                float(kalman_state[0]),
                float(kalman_state[1]),
                float((last_width + width) * 0.5),
                float((last_height + height) * 0.5),
            ]
    return frames


def _interpolate_detection(start_det, end_det, frame_idx, t_ms, alpha):
    canonical_track_id = start_det.get("identity_track_id", start_det.get("track_id"))
    synthesized = {
        "track_id": start_det.get("track_id"),
        "identity_track_id": canonical_track_id,
        "identity_track_source": start_det.get("identity_track_source", "tracker"),
        "identity_repair": start_det.get("identity_repair"),
        "identity_jersey_number": start_det.get("identity_jersey_number"),
        "identity_jersey_number_confidence": start_det.get("identity_jersey_number_confidence"),
        "identity_jersey_number_source": start_det.get("identity_jersey_number_source"),
        "class_id": start_det.get("class_id"),
        "class_name": start_det.get("class_name"),
        "continuity_segment_id": start_det.get("continuity_segment_id"),
        "confidence": round(_lerp(start_det.get("confidence", 0.0), end_det.get("confidence", 0.0), alpha), 4),
        "bbox_xyxy": [_lerp(a, b, alpha) for a, b in zip(start_det.get("bbox_xyxy", []), end_det.get("bbox_xyxy", []))],
        "bbox_xywh": [_lerp(a, b, alpha) for a, b in zip(start_det.get("bbox_xywh", []), end_det.get("bbox_xywh", []))],
        "uniform_bucket": start_det.get("uniform_bucket") if start_det.get("uniform_bucket") == end_det.get("uniform_bucket") else "unknown",
        "uniform_luma_mean": None,
        "repair_source": {
            "kind": "short_gap_interpolation",
            "start_frame_idx": start_det.get("_frame_idx"),
            "end_frame_idx": end_det.get("_frame_idx"),
        },
        "synthesized": True,
    }
    if start_det.get("court_xy") and end_det.get("court_xy"):
        synthesized["court_xy"] = [
            _lerp(start_det["court_xy"][0], end_det["court_xy"][0], alpha),
            _lerp(start_det["court_xy"][1], end_det["court_xy"][1], alpha),
        ]
    keypoints_xy = _interpolate_point_list(
        start_det.get("keypoints_xy"),
        end_det.get("keypoints_xy"),
        alpha,
    )
    if keypoints_xy is not None:
        synthesized["keypoints_xy"] = keypoints_xy
    keypoints_conf = _interpolate_point_list(
        [[c] for c in start_det.get("keypoints_conf", [])],
        [[c] for c in end_det.get("keypoints_conf", [])],
        alpha,
    )
    if keypoints_conf is not None:
        synthesized["keypoints_conf"] = [row[0] for row in keypoints_conf]
    lifted_xyz = _interpolate_point_list(
        start_det.get("lifted_keypoints_xyz"),
        end_det.get("lifted_keypoints_xyz"),
        alpha,
    )
    if lifted_xyz is not None:
        synthesized["lifted_keypoints_xyz"] = lifted_xyz
    synthesized["_frame_idx"] = frame_idx
    synthesized["_t_ms"] = t_ms
    return synthesized


def _match_discovery_recovery_detection(frame, predicted_bbox_xyxy):
    discovery_proposals = frame.get("discovery_proposals")
    if discovery_proposals is None:
        discovery_proposals = _build_discovery_proposals(frame)

    predicted_center = np.array(_bbox_center_xyxy(predicted_bbox_xyxy), dtype=np.float32)
    best_match = None
    for detection in frame.get("detections", []):
        if detection.get("track_id") is not None or not detection.get("recovered_candidate"):
            continue
        detection_bbox = detection.get("bbox_xyxy")
        if not detection_bbox or len(detection_bbox) != 4:
            continue

        proposal_score = float((detection.get("sam3_refinement") or {}).get("sam_score") or 0.0)
        matched_proposal = None
        for proposal in discovery_proposals or []:
            if proposal.get("proposal_role") != "recovery" or proposal.get("entity_type") != "player":
                continue
            proposal_bbox = proposal.get("bbox_xyxy")
            if not proposal_bbox or len(proposal_bbox) != 4:
                continue
            if _bbox_iou_xyxy(detection_bbox, proposal_bbox) >= 0.95:
                matched_proposal = proposal
                proposal_score = max(proposal_score, float(proposal.get("score") or 0.0))
                break

        if proposal_score < PLAYER_RECOVERY_MIN_SAM_SCORE:
            continue

        iou = _bbox_iou_xyxy(detection_bbox, predicted_bbox_xyxy)
        detection_center = np.array(_bbox_center_xyxy(detection_bbox), dtype=np.float32)
        center_distance = float(np.linalg.norm(detection_center - predicted_center))
        if iou < DISCOVERY_BRIDGE_MATCH_MIN_IOU and center_distance > DISCOVERY_BRIDGE_MATCH_MAX_CENTER_DISTANCE_PX:
            continue

        match_score = (
            0.50 * _clip01(float(proposal_score))
            + 0.30 * _clip01(iou)
            + 0.20 * _clip01(1.0 - (center_distance / max(DISCOVERY_BRIDGE_MATCH_MAX_CENTER_DISTANCE_PX, 1e-6)))
        )
        candidate = {
            "detection": detection,
            "proposal": matched_proposal,
            "proposal_score": round(float(proposal_score), 4),
            "iou": round(float(iou), 4),
            "center_distance_px": round(float(center_distance), 3),
            "match_score": round(float(match_score), 4),
        }
        if best_match is None or candidate["match_score"] > best_match["match_score"]:
            best_match = candidate
    return best_match


def repair_short_track_gaps(
    frames,
    video_meta=None,
    *,
    max_gap=SHORT_GAP_REPAIR_MAX_GAP,
    return_hypothesis_summary=False,
):
    track_frames = {}
    fps = float((video_meta or {}).get("fps") or 30.0)
    for frame in frames:
        for detection in frame.get("detections", []):
            track_id = detection.get("track_id")
            _initialize_identity_fields(detection)
            if track_id is None:
                continue
            detection["_frame_idx"] = frame["frame_idx"]
            detection["_t_ms"] = frame["t_ms"]
            track_frames.setdefault(track_id, []).append((frame["frame_idx"], detection))

    if not IDENTITY_HYPOTHESES_ENABLED:
        hypothesis_summary = empty_identity_hypothesis_summary()
    else:
        hypothesis_summary = build_identity_hypothesis_summary(
            track_frames,
            fps,
            config=IDENTITY_RESOLUTION_CONFIG,
            max_gap=max_gap,
        )

    stitcher_config = TrackletStitcherConfig(
        max_gap=max_gap,
        continuity_partition_field=TRACKLET_STITCHER_CONFIG.continuity_partition_field,
        reset_identity_on_discontinuity=TRACKLET_STITCHER_CONFIG.reset_identity_on_discontinuity,
        occlusion_first_within_segment=TRACKLET_STITCHER_CONFIG.occlusion_first_within_segment,
    )
    stitch_tracklets(
        frames,
        track_frames,
        hypothesis_summary,
        config=stitcher_config,
        interpolate_detection=_interpolate_detection,
        match_discovery_recovery_detection=_match_discovery_recovery_detection,
    )
    identity_links = hypothesis_summary["selected_links"]
    if return_hypothesis_summary:
        return frames, {
            "kind": "bounded_identity_mht_v1",
            "min_score": IDENTITY_HYPOTHESIS_MIN_SCORE,
            "ambiguity_margin": IDENTITY_HYPOTHESIS_AMBIGUITY_MARGIN,
            "group_count": len(hypothesis_summary["groups"]),
            "selected_link_count": len(identity_links),
            "decision_ledger_count": len(hypothesis_summary.get("decision_ledger") or []),
            "global_hypothesis_count": len(hypothesis_summary.get("global_hypotheses") or []),
            "track_identity_option_count": len(hypothesis_summary.get("track_identity_options") or []),
            "ambiguous_track_identity_option_count": sum(1 for item in (hypothesis_summary.get("track_identity_options") or []) if item.get("is_ambiguous")),
            "groups": hypothesis_summary["groups"],
            "decision_ledger": hypothesis_summary.get("decision_ledger") or [],
            "global_hypotheses": hypothesis_summary.get("global_hypotheses") or [],
            "track_identity_options": hypothesis_summary.get("track_identity_options") or [],
        }
    return frames


def annotate_clip(
    video_path,
    output_path,
    model_name,
    ball_model_name,
    device,
    conf_threshold,
    calibration_file,
    *,
    player_model_name=DEFAULT_PLAYER_MODEL,
    player_tracker_backend="default",
    player_conf_threshold=DEFAULT_PLAYER_PROPOSAL_CONFIDENCE,
    pose_conf_threshold=DEFAULT_POSE_EXTRACTION_CONFIDENCE,
    bootstrap_foreground_backend="none",
    bootstrap_foreground_model=DEFAULT_GROUNDING_DINO_MODEL,
    bootstrap_foreground_prompt=DEFAULT_GROUNDING_DINO_PROMPT,
    player_recovery_backend="none",
    player_recovery_model=DEFAULT_SAM3_REPO_MODEL,
    player_recovery_prompt=DEFAULT_SAM3_TEXT_PROMPT,
    ball_bootstrap_backend="none",
    ball_bootstrap_model=DEFAULT_SAM3_REPO_MODEL,
    ball_bootstrap_prompt=DEFAULT_SAM3_BALL_PROMPT,
):
    """Generate one full Layer 1 perception artifact for a single clip.

    The pipeline is still detection-centric at runtime: YOLO detection, ball
    search, continuity, grounding, recovery, repair, and review metadata are run
    first, and the normalized staged-perception contracts are materialized just
    before serialization.
    """
    resolved_torch_device, resource_policy = resolve_torch_device(device)
    device = str(resolved_torch_device)
    stage_timings = defaultdict(float)
    overall_start = time.perf_counter()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    player_model = YOLO(player_model_name)
    pose_model = YOLO(model_name)
    ball_model = YOLO(ball_model_name)
    sam_ball_detector = None
    sam_ball_status = "disabled"
    if ball_bootstrap_backend == "sam3":
        try:
            sam_ball_detector = Sam3BallDetector(
                model_name=ball_bootstrap_model,
                text_prompt=ball_bootstrap_prompt,
                device=device,
            )
            sam_ball_detector.load()
            sam_ball_status = "ready"
        except Exception as exc:
            sam_ball_status = _sam_runtime_status_for_exception(exc)
            sam_ball_detector = None
    frames = []
    frame_idx = 0
    clip_id = video_path.stem
    calibration = load_calibration(clip_id, calibration_file)
    previous_ball_predictive_state = None
    previous_ball_missing_streak = 0
    previous_frame_visual_signature = None
    previous_person_track_ids = set()
    previous_person_detection_count = 0
    while True:
        frame_loop_start = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break

        t_ms = int((frame_idx / fps) * 1000)
        player_track_kwargs = {
            "persist": True,
            "classes": [0],
            "conf": player_conf_threshold,
            "device": device,
            "verbose": False,
        }
        if player_tracker_backend != "default":
            player_track_kwargs["tracker"] = player_tracker_backend
        player_results = _timed_call(
            stage_timings,
            "frame.player_track",
            player_model.track,
            frame,
            **player_track_kwargs,
        )
        pose_results = _timed_call(
            stage_timings,
            "frame.pose_predict",
            pose_model.predict,
            frame,
            classes=[0],
            conf=pose_conf_threshold,
            device=device,
            verbose=False,
        )
        ball_results = _timed_call(
            stage_timings,
            "frame.ball_predict",
            ball_model.predict,
            frame,
            classes=[BALL_CLASS_ID],
            conf=0.05,
            device=device,
            verbose=False,
        )

        detections = []
        h_matrix = get_frame_homography(calibration, frame_idx)
        person_detections = _build_person_detections(
            player_results,
            player_model.names,
            frame,
            h_matrix=h_matrix,
            source="full_frame_detector",
            proposal_stage="full_frame",
            include_track_ids=True,
        )
        person_detections = _match_pose_to_person_detections(
            person_detections,
            pose_results,
            pose_model.names,
            frame,
            h_matrix=h_matrix,
        )
        detections.extend(person_detections)
        current_frame_visual_signature = _frame_visual_signature(frame)
        current_person_track_ids = {
            int(detection["track_id"])
            for detection in person_detections
            if detection.get("track_id") is not None
        }
        online_discontinuity = _online_ball_discontinuity(
            previous_frame_visual_signature,
            previous_person_track_ids,
            current_frame_visual_signature,
            current_person_track_ids,
            abs(len(person_detections) - previous_person_detection_count),
        )
        if online_discontinuity["triggered"]:
            previous_ball_predictive_state = None
        ball_predictive_summary = {
            "enabled": previous_ball_predictive_state is not None,
            "triggered": False,
            "source": "predictive_roi_v1",
            "roi_bbox_xyxy": None,
            "candidate_count": 0,
            "motion_mode": previous_ball_predictive_state.get("motion_mode") if previous_ball_predictive_state else None,
            "last_seen_frame_idx": previous_ball_predictive_state.get("last_seen_frame_idx") if previous_ball_predictive_state else None,
            "nearby_player_track_id": previous_ball_predictive_state.get("nearby_player_track_id") if previous_ball_predictive_state else None,
        }
        ball_fallback_summary = {
            "enabled": False,
            "triggered": False,
            "source": "player_local_rois_v2",
            "roi_bbox_xyxy": None,
            "candidate_count": 0,
        }
        ball_sam_summary = {
            "enabled": ball_bootstrap_backend == "sam3",
            "status": sam_ball_status,
            "triggered": False,
            "trigger_reason": None,
            "candidate_count": 0,
            "accepted": False,
            "accepted_score": None,
            "source": None,
            "model_name": ball_bootstrap_model if ball_bootstrap_backend == "sam3" else None,
            "text_prompt": ball_bootstrap_prompt if ball_bootstrap_backend == "sam3" else None,
            "missing_streak_before": int(previous_ball_missing_streak),
        }
        if ball_results and ball_results[0].boxes is not None and len(ball_results[0].boxes) > 0:
            result = ball_results[0]
            for det_idx in range(len(result.boxes)):
                built = build_detection(result, det_idx, ball_model.names, frame, h_matrix=h_matrix)
                built["ball_detection_source"] = "full_frame"
                detections.append(built)
        raw_ball_candidates = _serialize_ball_candidates(detections)
        best_ball_detection = _extract_best_ball_detection(detections)
        if _ball_detection_needs_search(best_ball_detection):
            predictive_ball_detections, ball_predictive_summary = _timed_call(
                stage_timings,
                "frame.ball_predictive_search",
                _run_ball_predictive_search,
                ball_model,
                frame,
                previous_ball_predictive_state,
                frame_idx=frame_idx,
                device=device,
                h_matrix=h_matrix,
            )
            detections.extend(predictive_ball_detections)
            raw_ball_candidates = _serialize_ball_candidates(detections)
            best_ball_detection = _extract_best_ball_detection(detections)
        if _ball_detection_needs_search(best_ball_detection):
            fallback_ball_detections, ball_fallback_summary = _timed_call(
                stage_timings,
                "frame.ball_roi_fallback",
                _run_ball_roi_fallback,
                ball_model,
                frame,
                person_detections,
                device=device,
                h_matrix=h_matrix,
            )
            detections.extend(fallback_ball_detections)
            raw_ball_candidates = _serialize_ball_candidates(detections)
            best_ball_detection = _extract_best_ball_detection(detections)
        ball_sam_trigger_reason = _ball_sam_trigger_reason(
            best_ball_detection,
            previous_ball_predictive_state,
            previous_ball_missing_streak,
            online_discontinuity,
        )
        sam_ball_detections, ball_sam_summary, sam_ball_status = _timed_call(
            stage_timings,
            "frame.ball_sam_search",
            _run_sam_ball_search,
            sam_ball_detector,
            frame,
            h_matrix=h_matrix,
            trigger_reason=ball_sam_trigger_reason,
        )
        ball_sam_summary["enabled"] = ball_bootstrap_backend == "sam3"
        if sam_ball_detector is None and ball_bootstrap_backend == "sam3":
            ball_sam_summary["status"] = sam_ball_status
        ball_sam_summary["missing_streak_before"] = int(previous_ball_missing_streak)
        ball_sam_summary["online_discontinuity"] = online_discontinuity
        if sam_ball_status != "ready":
            sam_ball_detector = None
        if sam_ball_detections:
            detections.extend(sam_ball_detections)
            raw_ball_candidates = _serialize_ball_candidates(detections)
            best_ball_detection = _extract_best_ball_detection(detections)
        ball_detection = best_ball_detection
        person_detections = [detection for detection in detections if not _is_ball_detection(detection)]
        previous_ball_predictive_state = _update_ball_predictive_state(
            previous_ball_predictive_state,
            ball_detection,
            person_detections,
            frame_idx,
        )
        if _ball_detection_needs_search(ball_detection):
            previous_ball_missing_streak += 1
        else:
            previous_ball_missing_streak = 0

        frames.append({
            "frame_idx": frame_idx,
            "t_ms": t_ms,
            "calibrated": h_matrix is not None,
            "detections": person_detections,
            "ball_detection": ball_detection,
            "raw_player_detections": _serialize_player_candidates(person_detections),
            "raw_ball_detections": raw_ball_candidates,
            "ball_predictive_search": ball_predictive_summary,
            "ball_fallback": ball_fallback_summary,
            "ball_sam": ball_sam_summary,
            "_frame_visual_signature": current_frame_visual_signature,
            "_frame_bgr": frame.copy(),
            "_h_matrix": None if h_matrix is None else h_matrix.tolist(),
        })
        previous_frame_visual_signature = current_frame_visual_signature
        previous_person_track_ids = current_person_track_ids
        previous_person_detection_count = len(person_detections)
        frame_idx += 1
        stage_timings["frame.total"] += time.perf_counter() - frame_loop_start

    cap.release()

    relative_path = str(video_path.relative_to(CLIPS_DIR))
    video_meta = {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
    }
    continuity = _timed_call(stage_timings, "post.continuity_segments", annotate_continuity_segments, frames)
    frame_quality_summary = _timed_call(stage_timings, "post.frame_quality", annotate_frame_quality, frames)
    quality_miss_summary = summarize_detection_misses_by_quality(frames)
    camera_pose_segments = _timed_call(stage_timings, "post.camera_pose_segments", annotate_camera_pose_segments, frames)
    bootstrap_summary = _timed_call(
        stage_timings,
        "post.bootstrap_contexts",
        annotate_bootstrap_contexts,
        frames,
        bootstrap_foreground_backend=bootstrap_foreground_backend,
        bootstrap_foreground_model=bootstrap_foreground_model,
        bootstrap_foreground_prompt=bootstrap_foreground_prompt,
        device=device,
    )
    scene_discovery_summary = _timed_call(stage_timings, "post.scene_discovery_contracts", annotate_scene_discovery_contracts, frames)
    target_court_first_pass = _timed_call(
        stage_timings,
        "post.target_court_first_pass",
        build_target_court_first_pass,
        frames,
        video_meta,
    )
    frames = _timed_call(stage_timings, "post.smooth_track_motion.initial", smooth_track_motion, frames, video_meta)
    frames = _timed_call(stage_timings, "post.annotate_active_players.initial", annotate_active_players, frames, video_meta)
    preliminary_live_play = _timed_call(stage_timings, "post.live_play.initial", annotate_live_play, frames, video_meta)
    collapse_summary = _timed_call(stage_timings, "post.collapse_reground", mark_tracking_collapse_reground, frames)
    grounding_summary = _timed_call(
        stage_timings,
        "post.grounded_person_rerun",
        rerun_grounded_person_detections,
        frames,
        player_model_name=player_model_name,
        device=device,
        conf_threshold=player_conf_threshold,
    )
    for frame in frames:
        if frame.get("_h_matrix") is not None:
            frame["_h_matrix"] = _h_matrix_as_array(frame["_h_matrix"])
    retro_ball_summary = _timed_call(
        stage_timings,
        "post.retro_ball_before_first_observation",
        retrofit_ball_detections_before_first_observation,
        frames,
        ball_model=ball_model,
        device=device,
    )
    ball_state_summary = _timed_call(stage_timings, "post.ball_state", annotate_ball_state, frames)
    player_recovery_summary = _timed_call(
        stage_timings,
        "post.sam_player_recovery",
        annotate_sam_player_recovery,
        frames,
        player_recovery_backend=player_recovery_backend,
        player_recovery_model=player_recovery_model,
        player_recovery_prompt=player_recovery_prompt,
        device=device,
    )
    engagement_summary = _timed_call(stage_timings, "post.player_engagement.initial", annotate_player_engagement, frames, video_meta)
    temporal_summary = _timed_call(stage_timings, "post.tracklet_temporal.initial", annotate_tracklet_temporal_evidence, frames, video_meta)
    appearance_summary = _timed_call(stage_timings, "post.team_appearance", annotate_team_appearance_consistency, frames)
    frames = _timed_call(stage_timings, "post.annotate_active_players.second", annotate_active_players, frames, video_meta)
    frames, identity_hypotheses = _timed_call(
        stage_timings,
        "post.repair_short_track_gaps",
        repair_short_track_gaps,
        frames,
        video_meta,
        return_hypothesis_summary=True,
    )
    frames = _timed_call(stage_timings, "post.smooth_track_motion.second", smooth_track_motion, frames, video_meta)
    engagement_summary = _timed_call(stage_timings, "post.player_engagement.second", annotate_player_engagement, frames, video_meta)
    temporal_summary = _timed_call(stage_timings, "post.tracklet_temporal.second", annotate_tracklet_temporal_evidence, frames, video_meta)
    frames = _timed_call(stage_timings, "post.annotate_active_players.third", annotate_active_players, frames, video_meta)
    jersey_ocr = _timed_call(stage_timings, "post.identity_jersey_ocr", annotate_identity_jersey_numbers, frames, identity_hypotheses=identity_hypotheses)
    identity_global_resolution = _timed_call(
        stage_timings,
        "post.identity_global_resolution",
        resolve_identity_global_hypotheses_with_jersey,
        identity_hypotheses,
        jersey_ocr,
    )
    _timed_call(stage_timings, "post.apply_identity_global_resolution", annotate_detections_with_jersey_global_resolution, frames, identity_global_resolution)
    live_play = _timed_call(stage_timings, "post.live_play.final", annotate_live_play, frames, video_meta)
    court_pose_summary = _timed_call(stage_timings, "post.court_pose", annotate_court_pose, frames, video_meta)
    staged_perception_summary = _timed_call(
        stage_timings,
        "post.staged_perception_summary",
        _build_staged_perception_summary,
        frames,
        bootstrap_summary=bootstrap_summary,
        grounding_summary=grounding_summary,
        collapse_summary=collapse_summary,
        scene_discovery_summary=scene_discovery_summary,
        player_recovery_summary=player_recovery_summary,
        engagement_summary=engagement_summary,
        temporal_summary=temporal_summary,
        identity_hypotheses=identity_hypotheses,
    )
    segment_grounding = []
    for segment in continuity["segments"]:
        source_frame = next(
            (
                frame for frame in frames
                if int(frame.get("continuity_segment_id", -1)) == int(segment["segment_id"])
                and int(frame.get("frame_idx", -1)) == int((frame.get("bootstrap_context") or {}).get("frame_idx", -2))
            ),
            None,
        )
        grounding_context = (source_frame or {}).get("grounding_context") or {}
        segment["grounding"] = {
            "status": "anchored" if grounding_context.get("enabled") else "inactive",
            "pipeline_order": grounding_context.get("pipeline_order") or ["grounding_dino", "sam", "yolo"],
            "trigger_reason": grounding_context.get("trigger_reason"),
            "source_frame_idx": int((source_frame or {}).get("frame_idx", segment["start_frame"])),
            "yolo_search_policy": grounding_context.get("yolo_search_policy"),
            "yolo_search_region_bbox_xyxy": grounding_context.get("yolo_search_region_bbox_xyxy"),
            "proposal_region_count": len(grounding_context.get("proposal_regions") or []),
            "collapse_triggered": bool(grounding_context.get("collapse_triggered")),
            "collapse_trigger_frame_idx": grounding_context.get("collapse_trigger_frame_idx"),
        }
        segment_grounding.append(segment["grounding"])

    artifact = {
        "schema_version": "1.4.0",
        "clip_id": clip_id,
        "video_path": relative_path,
        "video": video_meta,
        "model": {
            "player_name": player_model_name,
            "player_tracker_backend": player_tracker_backend,
            "name": model_name,
            "ball_name": ball_model_name,
            "ball_bootstrap_backend": ball_bootstrap_backend,
            "ball_bootstrap_model": ball_bootstrap_model if ball_bootstrap_backend != "none" else None,
            "ball_bootstrap_prompt": ball_bootstrap_prompt if ball_bootstrap_backend != "none" else None,
            "task": "detector_first_pose_enrichment_plus_ball_detect",
            "device": device,
            "requested_device": resource_policy.requested_device,
            "classes": ["person", "sports_ball"],
        },
        "postprocess": {
            "resource_policy": resource_policy.to_payload(),
            "timings_sec": {
                key: round(float(value), 4)
                for key, value in sorted(stage_timings.items())
            },
            "overall_runtime_sec": round(time.perf_counter() - overall_start, 4),
            "frame_quality": frame_quality_summary,
            "quality_miss_summary": quality_miss_summary,
            "court_pose": court_pose_summary,
            "camera_pose_segments": camera_pose_segments,
            "identity_policy": {
                "source": str(IDENTITY_POLICY_FILE.relative_to(REPO_ROOT)),
                "version": IDENTITY_POLICY_VERSION,
                "assumptions": {
                    "player_set_persistence": IDENTITY_POLICY_ASSUMPTIONS.get("player_set_persistence"),
                    "disappearance_default": IDENTITY_POLICY_ASSUMPTIONS.get("disappearance_default"),
                    "continuity_default": IDENTITY_POLICY_ASSUMPTIONS.get("continuity_default"),
                    "reset_condition": IDENTITY_POLICY_ASSUMPTIONS.get("reset_condition"),
                },
                "continuity": {
                    "kind": CONTINUITY_POLICY.get("kind"),
                    "reset_identity_on_discontinuity": RESET_IDENTITY_ON_DISCONTINUITY,
                },
                "identity": {
                    "conservation_prior": IDENTITY_POLICY.get("conservation_prior"),
                    "occlusion_first_within_segment": OCCLUSION_FIRST_WITHIN_SEGMENT,
                    "hypotheses_enabled": IDENTITY_HYPOTHESES_ENABLED,
                },
                "evidence_model": IDENTITY_EVIDENCE_MODEL,
                "hard_constraints": IDENTITY_HARD_CONSTRAINTS,
            },
            "active_player_score_threshold": ACTIVE_PLAYER_SCORE_THRESHOLD,
            "on_court_score_threshold": ON_COURT_SCORE_THRESHOLD,
            "player_plausibility": {
                "kind": "multisignal_on_court_plausibility_v1",
                "signals": [
                    "confidence",
                    "bbox_height",
                    "edge_penalty",
                    "court_grounding",
                    "pose_coherence",
                    "track_persistence",
                    "motion",
                    "bootstrap_play_region",
                    "appearance_penalty",
                    "merge_risk",
                ],
            },
            "short_gap_repair_max_gap": SHORT_GAP_REPAIR_MAX_GAP,
            "identity_repair_score_threshold": IDENTITY_REPAIR_SCORE_THRESHOLD,
            "motion_score_threshold_px": MOTION_SCORE_THRESHOLD_PX,
            "track_motion_smoother": "kalman_constant_velocity_v1",
            "identity_repair": "trajectory_bridge_v1",
            "identity_hypotheses": {
                "kind": identity_hypotheses["kind"],
                "min_score": identity_hypotheses["min_score"],
                "ambiguity_margin": identity_hypotheses["ambiguity_margin"],
                "group_count": identity_hypotheses["group_count"],
                "selected_link_count": identity_hypotheses["selected_link_count"],
                "decision_ledger_count": identity_hypotheses.get("decision_ledger_count", 0),
                "global_hypothesis_count": identity_hypotheses.get("global_hypothesis_count", 0),
                "track_identity_option_count": identity_hypotheses.get("track_identity_option_count", 0),
                "ambiguous_track_identity_option_count": identity_hypotheses.get("ambiguous_track_identity_option_count", 0),
            },
            "appearance_cue": {
                "kind": "torso_rgb_quantized_histogram_v1",
                "team_distance_threshold": APPEARANCE_TEAM_DISTANCE_THRESHOLD,
                "low_motion_threshold_px": APPEARANCE_LOW_MOTION_THRESHOLD_PX,
                "prototype_count": appearance_summary["prototype_count"],
                "prototype_buckets": appearance_summary["prototype_buckets"],
            },
            "grounding_workflow": grounding_summary,
            "grounding_collapse": collapse_summary,
            "scene_discovery_tracking": scene_discovery_summary,
            "target_court_first_pass": target_court_first_pass,
            "preliminary_live_play_segments": len(preliminary_live_play["segments"]),
            "player_recovery": player_recovery_summary,
            "ball_state": {
                "kind": ball_state_summary["kind"],
                "min_score": ball_state_summary["min_score"],
                "max_jump_px": ball_state_summary["max_jump_px"],
                "max_gap_frames": ball_state_summary["max_gap_frames"],
                "retro_anchor_min_confidence": BALL_RETRO_ANCHOR_MIN_CONFIDENCE,
                "retro_base_radius_px": BALL_RETRO_BASE_RADIUS_PX,
                "retro_backfilled_frame_count": int(retro_ball_summary.get("backfilled_frame_count") or 0),
            },
            "live_play_gate": {
                "kind": "heuristic_frame_and_segment_v1",
                "live_threshold": LIVE_PLAY_SCORE_THRESHOLD,
                "dead_threshold": DEAD_BALL_SCORE_THRESHOLD,
                "high_motion_threshold_px": LIVE_PLAY_HIGH_MOTION_THRESHOLD_PX,
                "enter_window": LIVE_PLAY_ENTER_WINDOW,
                "enter_min_positive": LIVE_PLAY_ENTER_MIN_POSITIVE,
                "exit_window": LIVE_PLAY_EXIT_WINDOW,
                "exit_min_negative": LIVE_PLAY_EXIT_MIN_NEGATIVE,
                "ball_signal_present": True,
                "ball_min_confidence": BALL_MIN_LIVE_PLAY_CONFIDENCE,
                "segment_count": len(live_play["segments"]),
            },
            "continuity": {
                "kind": "visual_and_track_churn_v1",
                "visual_delta_threshold": DISCONTINUITY_VISUAL_DELTA_THRESHOLD,
                "track_churn_threshold": DISCONTINUITY_TRACK_CHURN_THRESHOLD,
                "score_threshold": DISCONTINUITY_SCORE_THRESHOLD,
                "segment_count": len(continuity["segments"]),
            },
            "jersey_ocr": {
                "backend": "easyocr_v1",
                "reader_available": jersey_ocr["reader_available"],
                "max_samples_per_identity": JERSEY_OCR_MAX_SAMPLES_PER_IDENTITY,
                "min_sharpness": JERSEY_OCR_MIN_SHARPNESS,
                "consensus_min_votes": JERSEY_OCR_CONSENSUS_MIN_VOTES,
                "consensus_min_share": JERSEY_OCR_CONSENSUS_MIN_SHARE,
                "min_on_court_track_frames": JERSEY_OCR_MIN_ON_COURT_TRACK_FRAMES,
                "identity_count_with_consensus": jersey_ocr["identity_count_with_consensus"],
                "track_option_count_with_consensus": jersey_ocr.get("track_option_count_with_consensus", 0),
                "ambiguous_track_option_count_with_consensus": jersey_ocr.get("ambiguous_track_option_count_with_consensus", 0),
                "global_hypothesis_count": identity_global_resolution.get("global_hypothesis_count", 0),
                "selected_global_hypothesis_id": identity_global_resolution.get("selected_global_hypothesis_id"),
                "changed_selected_global_hypothesis": bool(identity_global_resolution.get("changed_selected_global_hypothesis")),
            },
        },
        "staged_perception": staged_perception_summary,
        "calibration": {
            "enabled": calibration is not None,
            "source": str(calibration_file.relative_to(REPO_ROOT)) if calibration_file and calibration_file.exists() else None,
            "type": calibration.get("type") if calibration else None,
        },
        "identity_jersey_consensus": jersey_ocr["identity_consensus"],
        "identity_jersey_track_options": jersey_ocr.get("track_option_consensus") or {},
        "identity_hypotheses": identity_hypotheses["groups"],
        "identity_global_hypotheses": identity_hypotheses.get("global_hypotheses") or [],
        "identity_global_hypothesis_resolution": identity_global_resolution,
        "identity_track_options": identity_hypotheses.get("track_identity_options") or [],
        "identity_link_decisions": identity_hypotheses.get("decision_ledger") or [],
        "continuity_segments": continuity["segments"],
        "live_play_segments": live_play["segments"],
        "bootstrap_foreground": {
            "backend": bootstrap_summary["backend"],
            "contexts": bootstrap_summary["contexts"],
        },
        "grounding_segments": segment_grounding,
        "frames": frames,
    }

    for frame in artifact["frames"]:
        frame.pop("_frame_visual_signature", None)
        frame.pop("_frame_bgr", None)
        frame.pop("_h_matrix", None)

    with open(output_path, "w") as f:
        json.dump(artifact, f, indent=2)
    print("[TIMING] overall_runtime_sec", round(time.perf_counter() - overall_start, 4))
    for key, value in sorted(stage_timings.items(), key=lambda item: item[1], reverse=True):
        print("[TIMING]", key, round(float(value), 4))


def main():
    parser = argparse.ArgumentParser(description="Generate Layer 1 annotations for the labeller.")
    parser.add_argument("clip_path", help="Absolute or repo-relative path to the clip")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--player-model", default=DEFAULT_PLAYER_MODEL, help="Ultralytics detection model used for person proposals")
    parser.add_argument(
        "--player-tracker-backend",
        default="default",
        help="Ultralytics tracker backend for player tracking: default, bytetrack.yaml, or botsort.yaml",
    )
    parser.add_argument("--model", default="yolov8n-pose.pt", help="Ultralytics pose model name or path")
    parser.add_argument("--ball-model", default="yolov8n.pt", help="Ultralytics detection model used for sports-ball proposals")
    parser.add_argument(
        "--bootstrap-foreground-backend",
        choices=["none", "grounding_dino"],
        default="none",
        help="Optional bootstrap foreground/background pre-pass backend",
    )
    parser.add_argument(
        "--bootstrap-foreground-model",
        default=DEFAULT_GROUNDING_DINO_MODEL,
        help="Model id used when the Grounding DINO bootstrap foreground backend is enabled",
    )
    parser.add_argument(
        "--bootstrap-foreground-prompt",
        default=DEFAULT_GROUNDING_DINO_PROMPT,
        help="Prompt text used when the Grounding DINO bootstrap foreground backend is enabled",
    )
    parser.add_argument(
        "--player-recovery-backend",
        choices=["none", "sam3"],
        default="none",
        help="Optional SAM 3 recovered-player backend for unexplained DINO blobs and ambiguous YOLO detections",
    )
    parser.add_argument(
        "--ball-bootstrap-backend",
        choices=["none", "sam3"],
        default="none",
        help="Optional SAM 3 runtime bootstrap backend for initial and sustained-missing ball reacquisition",
    )
    parser.add_argument(
        "--player-recovery-model",
        default=DEFAULT_SAM3_REPO_MODEL,
        help="Model id used when the SAM 3 player recovery backend is enabled",
    )
    parser.add_argument(
        "--player-recovery-prompt",
        default=DEFAULT_SAM3_TEXT_PROMPT,
        help="Text prompt passed to SAM 3 when the recovered-player backend is enabled",
    )
    parser.add_argument(
        "--ball-bootstrap-model",
        default=DEFAULT_SAM3_REPO_MODEL,
        help="Model id used when the SAM 3 ball bootstrap backend is enabled",
    )
    parser.add_argument(
        "--ball-bootstrap-prompt",
        default=DEFAULT_SAM3_BALL_PROMPT,
        help="Text prompt passed to SAM 3 when the ball bootstrap backend is enabled",
    )
    parser.add_argument("--device", default="cuda:0", help="Ultralytics device selector")
    parser.add_argument("--player-conf", type=float, default=DEFAULT_PLAYER_PROPOSAL_CONFIDENCE, help="Person detector proposal confidence threshold")
    parser.add_argument("--pose-conf", type=float, default=DEFAULT_POSE_EXTRACTION_CONFIDENCE, help="Pose extraction confidence threshold")
    parser.add_argument("--conf", type=float, default=0.25, help="Backward-compatible alias for pose extraction confidence threshold")
    parser.add_argument(
        "--calibration-file",
        default=str(CALIBRATION_FILE),
        help="Calibration JSON used to add court-space and lifted coordinates when available",
    )
    args = parser.parse_args()

    clip_path = Path(args.clip_path)
    if not clip_path.is_absolute():
        clip_path = (REPO_ROOT / clip_path).resolve()
    output_path = Path(args.output) if args.output else OUTPUT_DIR / f"{clip_path.stem}.perception.json"
    calibration_file = Path(args.calibration_file) if args.calibration_file else None

    annotate_clip(
        video_path=clip_path,
        output_path=output_path,
        model_name=args.model,
        ball_model_name=args.ball_model,
        device=args.device,
        conf_threshold=args.conf,
        calibration_file=calibration_file,
        player_model_name=args.player_model,
        player_tracker_backend=args.player_tracker_backend,
        player_conf_threshold=args.player_conf,
        pose_conf_threshold=args.pose_conf if args.pose_conf is not None else args.conf,
        bootstrap_foreground_backend=args.bootstrap_foreground_backend,
        bootstrap_foreground_model=args.bootstrap_foreground_model,
        bootstrap_foreground_prompt=args.bootstrap_foreground_prompt,
        player_recovery_backend=args.player_recovery_backend,
        player_recovery_model=args.player_recovery_model,
        player_recovery_prompt=args.player_recovery_prompt,
        ball_bootstrap_backend=args.ball_bootstrap_backend,
        ball_bootstrap_model=args.ball_bootstrap_model,
        ball_bootstrap_prompt=args.ball_bootstrap_prompt,
    )
    print(output_path)


if __name__ == "__main__":
    main()

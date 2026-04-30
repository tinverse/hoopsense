"""Weak-feature continuous court-pose tracking.

This module intentionally does not assume clean, visible court lines. It carries
the best available court-plane pose through a clip using calibration when
available and weaker support from scene priors plus player foot anchors when
court markings are sparse.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pipelines.geometry import homography_sanity


LEFT_ANKLE = 15
RIGHT_ANKLE = 16


@dataclass
class CourtPoseState:
    state: str
    confidence: float
    frame_idx: int
    source: str
    update_reason: str
    homography_image_to_court: list[list[float]] | None
    homography_determinant: float | None
    visible_features: list[str]
    foot_anchor_count: int
    supported_foot_anchor_count: int
    scene_prior_status: str
    temporal_age_frames: int
    detector_policy: str | None = None

    def to_payload(self) -> dict:
        return {
            "kind": "court_pose_v1",
            "state": self.state,
            "confidence": round(float(self.confidence), 4),
            "frame_idx": int(self.frame_idx),
            "source": self.source,
            "update_reason": self.update_reason,
            "camera_model": "planar_homography_or_weak_scene_prior",
            "homography_image_to_court": self.homography_image_to_court,
            "homography_determinant": (
                round(float(self.homography_determinant), 6)
                if self.homography_determinant is not None
                else None
            ),
            "visible_features": list(self.visible_features),
            "foot_anchor_count": int(self.foot_anchor_count),
            "supported_foot_anchor_count": int(self.supported_foot_anchor_count),
            "scene_prior_status": self.scene_prior_status,
            "temporal_age_frames": int(self.temporal_age_frames),
            "detector_policy": self.detector_policy,
        }


def _homography_payload(h_matrix):
    if h_matrix is None:
        return None, None
    h = np.asarray(h_matrix, dtype=float)
    if h.shape != (3, 3):
        return None, None
    try:
        sanity = homography_sanity(h)
    except Exception:
        return None, None
    if not sanity.get("invertible"):
        return None, float(sanity.get("determinant") or 0.0)
    return [[round(float(value), 8) for value in row] for row in h.tolist()], float(sanity["determinant"])


def _footpoint(detection):
    bbox = detection.get("bbox_xyxy") or [0.0, 0.0, 0.0, 0.0]
    x1, _y1, x2, y2 = [float(v) for v in bbox]
    keypoints = detection.get("keypoints_xy") or []
    confidences = detection.get("keypoints_conf") or []
    ankle_points = []
    for idx in (LEFT_ANKLE, RIGHT_ANKLE):
        if idx >= len(keypoints):
            continue
        confidence = float(confidences[idx]) if idx < len(confidences) else 1.0
        if confidence >= 0.2:
            ankle_points.append((float(keypoints[idx][0]), float(keypoints[idx][1]), confidence))
    if ankle_points:
        return {
            "xy": [
                sum(point[0] for point in ankle_points) / len(ankle_points),
                sum(point[1] for point in ankle_points) / len(ankle_points),
            ],
            "source": "ankles",
            "confidence": sum(point[2] for point in ankle_points) / len(ankle_points),
        }
    return {
        "xy": [(x1 + x2) * 0.5, y2],
        "source": "bbox_bottom",
        "confidence": 0.25,
    }


def _mask_prior(scene_prior, x, y, frame_width, frame_height):
    mask_grid = (scene_prior or {}).get("region_mask_grid") or []
    mask_shape = (scene_prior or {}).get("region_mask_shape") or []
    if not mask_grid or len(mask_shape) != 2 or frame_width <= 0 or frame_height <= 0:
        return 0.0
    height, width = int(mask_shape[0]), int(mask_shape[1])
    if height <= 0 or width <= 0:
        return 0.0
    gx = min(width - 1, max(0, int((float(x) / max(float(frame_width), 1.0)) * width)))
    gy = min(height - 1, max(0, int((float(y) / max(float(frame_height), 1.0)) * height)))
    return float(mask_grid[gy][gx])


def _foot_anchor_support(detections, scene_prior, frame_width, frame_height):
    anchors = []
    for detection in detections or []:
        if detection.get("class_id") not in (None, 0):
            continue
        foot = _footpoint(detection)
        x, y = foot["xy"]
        prior = _mask_prior(scene_prior, x, y, frame_width, frame_height)
        anchors.append(
            {
                "xy": [round(float(x), 3), round(float(y), 3)],
                "source": foot["source"],
                "confidence": round(float(foot["confidence"]), 4),
                "scene_prior": round(float(prior), 4),
                "supported": bool(prior >= 0.5),
            }
        )
    return anchors


class CourtPoseTracker:
    """Maintain court pose through sparse visible geometry."""

    def __init__(self, *, carry_max_age_frames=45):
        self.carry_max_age_frames = int(carry_max_age_frames)
        self.last_state: CourtPoseState | None = None

    def update(self, frame, *, video_meta=None) -> CourtPoseState:
        video_meta = video_meta or {}
        frame_idx = int(frame.get("frame_idx") or 0)
        scene_prior = frame.get("scene_prior") or {}
        detections = frame.get("detections") or []
        frame_width = int(video_meta.get("width") or 0)
        frame_height = int(video_meta.get("height") or 0)
        if frame_width <= 0 or frame_height <= 0:
            for detection in detections:
                bbox = detection.get("bbox_xyxy") or []
                if len(bbox) == 4:
                    frame_width = max(frame_width, int(max(float(bbox[0]), float(bbox[2]))))
                    frame_height = max(frame_height, int(max(float(bbox[1]), float(bbox[3]))))
        homography, determinant = _homography_payload(frame.get("_h_matrix"))
        anchors = _foot_anchor_support(detections, scene_prior, frame_width, frame_height)
        supported_anchor_count = sum(1 for anchor in anchors if anchor["supported"])
        scene_ready = scene_prior.get("prior_status") == "ready"
        detector_policy = (frame.get("frame_quality") or {}).get("detector_policy")

        visible_features = []
        if homography is not None:
            visible_features.append("calibrated_homography")
        if scene_ready:
            visible_features.append("grounding_scene_prior")
        if anchors:
            visible_features.append("player_foot_anchors")
        if supported_anchor_count:
            visible_features.append("scene_supported_foot_anchors")

        if homography is not None:
            confidence = 0.68
            confidence += min(0.12, 0.02 * len(anchors))
            confidence += min(0.12, 0.03 * supported_anchor_count)
            confidence += 0.08 if scene_ready else 0.0
            state = "calibrated"
            source = "calibration_plus_weak_features"
            update_reason = "calibration_available"
            temporal_age = 0
        elif scene_ready or anchors:
            confidence = 0.18
            confidence += 0.18 if scene_ready else 0.0
            confidence += min(0.18, 0.03 * len(anchors))
            confidence += min(0.14, 0.04 * supported_anchor_count)
            state = "weak_scene_pose"
            source = "grounding_and_player_foot_anchors"
            update_reason = "weak_features_only"
            temporal_age = 0
        elif self.last_state is not None and self.last_state.temporal_age_frames < self.carry_max_age_frames:
            carried = self.last_state
            temporal_age = carried.temporal_age_frames + 1
            decay = max(0.0, 1.0 - float(temporal_age) / float(max(self.carry_max_age_frames, 1)))
            state = "carried"
            confidence = float(carried.confidence) * decay
            source = carried.source
            update_reason = "temporal_carry_forward"
            homography = carried.homography_image_to_court
            determinant = carried.homography_determinant
            visible_features = list(carried.visible_features)
        else:
            state = "unknown"
            confidence = 0.0
            source = "none"
            update_reason = "no_reliable_court_features"
            temporal_age = 0

        if detector_policy in {"bridge_measurements", "downweight_measurements"}:
            confidence *= 0.85
            if "low_quality_frame" not in visible_features:
                visible_features.append("low_quality_frame")

        pose = CourtPoseState(
            state=state,
            confidence=max(0.0, min(1.0, float(confidence))),
            frame_idx=frame_idx,
            source=source,
            update_reason=update_reason,
            homography_image_to_court=homography,
            homography_determinant=determinant,
            visible_features=visible_features,
            foot_anchor_count=len(anchors),
            supported_foot_anchor_count=supported_anchor_count,
            scene_prior_status=str(scene_prior.get("prior_status") or "missing"),
            temporal_age_frames=temporal_age,
            detector_policy=detector_policy,
        )
        self.last_state = pose
        return pose


def annotate_court_pose(frames, video_meta=None) -> dict:
    tracker = CourtPoseTracker()
    counts = {}
    confidence_sum = 0.0
    for frame in frames:
        pose = tracker.update(frame, video_meta=video_meta)
        frame["court_pose"] = pose.to_payload()
        counts[pose.state] = int(counts.get(pose.state, 0)) + 1
        confidence_sum += float(pose.confidence)
    frame_count = len(frames)
    return {
        "enabled": True,
        "kind": "court_pose_tracking_v1",
        "frame_count": int(frame_count),
        "state_counts": dict(sorted(counts.items())),
        "mean_confidence": round(confidence_sum / max(float(frame_count), 1.0), 4),
        "strategy": "calibration_first_grounding_and_player_foot_anchor_support",
        "line_visibility_assumption": "sparse_or_intermittent",
    }

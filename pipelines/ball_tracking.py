"""Runtime ball-state tracking primitives.

The live pipeline needs a dedicated ball subsystem because ball state has a
different cadence and failure mode than player tracking. This module starts by
owning the existing conservative state selector so future ROI scheduling and
dedicated ball models can evolve outside the main inference loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pipelines.geometry import play_region_prior_for_point

BALL_CLASS_ID = 32
BALL_MIN_OBSERVED_CONFIDENCE = 0.30
BALL_MAX_PREDICT_GAP_FRAMES = 4
BALL_MAX_JUMP_PX = 180.0
BALL_MIN_SIZE_PX = 4.0
BALL_MAX_SIZE_PX = 80.0
BALL_RECOVERY_SOURCES = {"airborne_player_reacquisition", "pose_shot_pass_corridor"}


@dataclass(frozen=True)
class BallSearchRoi:
    bbox_xyxy: list[int]
    source: str
    track_id: int | None = None
    motion_mode: str | None = None


@dataclass(frozen=True)
class BallSearchPlan:
    run_full_frame: bool
    reason: str
    rois: list[BallSearchRoi] = field(default_factory=list)
    motion_mode: str = "unknown_recent"
    motion_mode_confidence: float = 0.0

    def to_payload(self) -> dict:
        return {
            "kind": "runtime_ball_search_plan_v1",
            "run_full_frame": bool(self.run_full_frame),
            "reason": self.reason,
            "roi_count": len(self.rois),
            "rois": [
                {
                    "bbox_xyxy": list(roi.bbox_xyxy),
                    "source": roi.source,
                    "track_id": roi.track_id,
                    "motion_mode": roi.motion_mode,
                }
                for roi in self.rois
            ],
            "motion_mode": self.motion_mode,
            "motion_mode_confidence": self.motion_mode_confidence,
        }


@dataclass(frozen=True)
class BallSearchSchedulerConfig:
    full_frame_interval_frames: int = 1
    missing_full_frame_interval_frames: int = 1
    roi_search_enabled: bool = False
    max_roi_count: int = 4
    last_ball_radius_px: float = 96.0
    player_roi_width_scale: float = 1.25
    player_roi_height_scale: float = 1.10


@dataclass(frozen=True)
class BallMotionContext:
    motion_mode: str
    confidence: float
    track_id: int | None = None
    hand_xy: list[float] | None = None
    foot_y: float | None = None

    def to_payload(self) -> dict:
        return {
            "motion_mode": self.motion_mode,
            "confidence": round(float(self.confidence), 4),
            "track_id": self.track_id,
            "hand_xy": None if self.hand_xy is None else [round(float(v), 3) for v in self.hand_xy],
            "foot_y": None if self.foot_y is None else round(float(self.foot_y), 3),
        }


class BallMotionClassifier:
    """Infer coarse ball motion mode from recent ball state and player pose."""

    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16
    NOSE = 0

    def classify(self, ball_state=None, player_detections=None) -> BallMotionContext:
        ball_state = ball_state or {}
        center_xy = ball_state.get("center_xy")
        velocity_xy = ball_state.get("velocity_xy") or [0.0, 0.0]
        vx, vy = [float(v) for v in velocity_xy]
        speed = float(np.linalg.norm(np.array([vx, vy], dtype=np.float32)))
        if not center_xy:
            return BallMotionContext("unknown_recent", 0.0)

        nearest = self._nearest_pose_player(center_xy, player_detections or [])
        if speed >= 35.0 and vy <= -18.0:
            return BallMotionContext("shot_or_lob", 0.78, track_id=(nearest or {}).get("track_id"), hand_xy=(nearest or {}).get("hand_xy"))
        if nearest is None:
            if speed >= 45.0:
                return BallMotionContext("pass_or_loose", 0.55)
            return BallMotionContext("unknown_recent", 0.25)

        hand_xy = nearest.get("hand_xy")
        foot_y = nearest.get("foot_y")
        track_id = nearest.get("track_id")
        ball_y = float(center_xy[1])
        if hand_xy is not None and foot_y is not None:
            hand_y = float(hand_xy[1])
            if speed < 35.0 and abs(ball_y - hand_y) <= 0.30 * float(nearest["bbox_h"]):
                return BallMotionContext("carry_or_hold", 0.72, track_id=track_id, hand_xy=hand_xy, foot_y=foot_y)
            if vy >= 18.0 and ball_y >= hand_y + 0.22 * float(nearest["bbox_h"]):
                return BallMotionContext("dribble_like", 0.68, track_id=track_id, hand_xy=hand_xy, foot_y=foot_y)
        if speed >= 45.0:
            return BallMotionContext("pass_or_loose", 0.62, track_id=track_id, hand_xy=hand_xy, foot_y=foot_y)
        return BallMotionContext("unknown_recent", 0.35, track_id=track_id, hand_xy=hand_xy, foot_y=foot_y)

    def pose_corridor_rois(self, frame_shape, player_detections, motion_mode: str):
        if motion_mode not in {"shot_or_lob", "pass_or_loose", "unknown_recent"}:
            return []
        rois = []
        for detection in player_detections:
            keypoints_xy = detection.get("keypoints_xy")
            hand_points = self._points(keypoints_xy, [self.LEFT_WRIST, self.RIGHT_WRIST])
            head_points = self._points(keypoints_xy, [self.NOSE, self.LEFT_SHOULDER, self.RIGHT_SHOULDER])
            if not hand_points:
                continue
            hand = min(hand_points, key=lambda point: float(point[1]))
            if head_points:
                head_y = min(float(point[1]) for point in head_points)
            else:
                bbox = detection.get("bbox_xywh") or [hand[0], hand[1], 1.0, 1.0]
                head_y = float(bbox[1]) - 0.45 * float(bbox[3])
            x = float(hand[0])
            y = min(float(hand[1]), head_y + 30.0)
            bbox = detection.get("bbox_xywh") or [x, y, 80.0, 160.0]
            _, _, bw, bh = [float(v) for v in bbox]
            if motion_mode == "shot_or_lob":
                roi = [x - 1.35 * bw, y - 1.80 * bh, x + 1.35 * bw, y + 0.35 * bh]
            else:
                roi = [x - 1.65 * bw, y - 0.95 * bh, x + 1.65 * bw, y + 0.55 * bh]
            rois.append(
                BallSearchRoi(
                    bbox_xyxy=BallSearchScheduler._clip_roi(roi, frame_shape),
                    source="pose_shot_pass_corridor",
                    track_id=detection.get("track_id"),
                    motion_mode=motion_mode,
                )
            )
        return rois

    def _nearest_pose_player(self, center_xy, player_detections):
        best = None
        best_distance = None
        for detection in player_detections:
            hand_xy = self._nearest_hand(center_xy, detection.get("keypoints_xy"))
            if hand_xy is None:
                continue
            distance = float(np.linalg.norm(np.array(hand_xy, dtype=np.float32) - np.array(center_xy, dtype=np.float32)))
            bbox = detection.get("bbox_xywh") or [0.0, 0.0, 1.0, 1.0]
            bbox_h = max(1.0, float(bbox[3]))
            if distance > max(80.0, 0.85 * bbox_h):
                continue
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best = {
                    "track_id": detection.get("track_id"),
                    "hand_xy": [float(hand_xy[0]), float(hand_xy[1])],
                    "foot_y": self._foot_y(detection.get("keypoints_xy")),
                    "bbox_h": bbox_h,
                }
        return best

    def _nearest_hand(self, center_xy, keypoints_xy):
        points = self._points(keypoints_xy, [self.LEFT_WRIST, self.RIGHT_WRIST])
        if not points:
            return None
        center = np.array(center_xy, dtype=np.float32)
        return min(points, key=lambda point: float(np.linalg.norm(np.array(point, dtype=np.float32) - center)))

    def _foot_y(self, keypoints_xy):
        points = self._points(keypoints_xy, [self.LEFT_ANKLE, self.RIGHT_ANKLE])
        if not points:
            return None
        return max(float(point[1]) for point in points)

    @staticmethod
    def _points(keypoints_xy, indexes):
        if keypoints_xy is None:
            return []
        points = []
        for index in indexes:
            if index >= len(keypoints_xy):
                continue
            point = keypoints_xy[index]
            if point is None or len(point) < 2:
                continue
            x, y = [float(v) for v in point[:2]]
            if x <= 0.0 and y <= 0.0:
                continue
            points.append([x, y])
        return points


class BallSearchScheduler:
    """Plan when the runtime should spend full-frame or ROI ball detector work."""

    def __init__(self, config: BallSearchSchedulerConfig | None = None):
        self.config = config or BallSearchSchedulerConfig()
        self.motion_classifier = BallMotionClassifier()

    def plan(self, frame_shape, *, frame_idx: int, ball_state=None, player_detections=None) -> BallSearchPlan:
        player_detections = list(player_detections or [])
        motion_context = self.motion_classifier.classify(ball_state, player_detections)
        state_name = (ball_state or {}).get("state")
        center_xy = (ball_state or {}).get("center_xy")
        full_interval = max(1, int(self.config.full_frame_interval_frames))
        if not center_xy or state_name in {None, "missing"}:
            missing_interval = max(1, int(self.config.missing_full_frame_interval_frames))
            if missing_interval <= 1:
                run_full_frame = True
                reason = "no_recent_ball"
            elif int(frame_idx) % missing_interval == 0:
                run_full_frame = True
                reason = "missing_cadence_due"
            else:
                run_full_frame = False
                reason = "missing_roi_cadence"
        elif full_interval <= 1:
            run_full_frame = True
            reason = "cadence_every_frame"
        elif int(frame_idx) % full_interval == 0:
            run_full_frame = True
            reason = "cadence_due"
        else:
            run_full_frame = False
            reason = "roi_cadence"

        rois = []
        if self.config.roi_search_enabled:
            if center_xy:
                rois.append(
                    BallSearchRoi(
                        bbox_xyxy=self._roi_for_motion(frame_shape, center_xy, ball_state or {}, motion_context),
                        source="last_ball_state",
                        track_id=motion_context.track_id,
                        motion_mode=motion_context.motion_mode,
                    )
                )
            if state_name in {None, "missing"}:
                rois.extend(self._airborne_player_rois(frame_shape, center_xy, player_detections))
            if motion_context.motion_mode in {"shot_or_lob", "pass_or_loose", "unknown_recent"}:
                rois.extend(
                    self.motion_classifier.pose_corridor_rois(
                        frame_shape,
                        player_detections,
                        motion_context.motion_mode,
                    )
                )
            rois.extend(self._player_rois(frame_shape, center_xy, player_detections, motion_context))
        return BallSearchPlan(
            run_full_frame=run_full_frame,
            reason=reason,
            rois=rois[: max(0, int(self.config.max_roi_count))],
            motion_mode=motion_context.motion_mode,
            motion_mode_confidence=round(float(motion_context.confidence), 4),
        )

    def _player_rois(self, frame_shape, center_xy, player_detections, motion_context):
        ranked = []
        for detection in player_detections:
            bbox = detection.get("bbox_xywh")
            if bbox is None or len(bbox) < 4:
                continue
            cx, cy, bw, bh = [float(v) for v in bbox]
            if bw <= 1.0 or bh <= 1.0:
                continue
            distance = 0.0
            if center_xy:
                distance = float(np.linalg.norm(np.array([cx, cy], dtype=np.float32) - np.array(center_xy, dtype=np.float32)))
            if motion_context.track_id is not None and detection.get("track_id") == motion_context.track_id:
                distance -= 250.0
            ranked.append((distance, detection))
        ranked.sort(key=lambda item: item[0])
        rois = []
        for _distance, detection in ranked:
            rois.append(
                BallSearchRoi(
                    bbox_xyxy=self._roi_around_player(frame_shape, detection["bbox_xywh"], motion_context.motion_mode),
                    source="nearby_player",
                    track_id=detection.get("track_id"),
                    motion_mode=motion_context.motion_mode,
                )
            )
        return rois

    def _airborne_player_rois(self, frame_shape, center_xy, player_detections):
        ranked = []
        for detection in player_detections:
            bbox = detection.get("bbox_xywh")
            if bbox is None or len(bbox) < 4:
                continue
            cx, cy, bw, bh = [float(v) for v in bbox]
            if bw <= 1.0 or bh <= 1.0:
                continue
            distance = 0.0
            if center_xy:
                distance = float(np.linalg.norm(np.array([cx, cy], dtype=np.float32) - np.array(center_xy, dtype=np.float32)))
            ranked.append((distance, bbox, detection.get("track_id")))
        ranked.sort(key=lambda item: item[0])
        rois = []
        for _distance, bbox, track_id in ranked:
            cx, cy, bw, bh = [float(v) for v in bbox]
            rois.append(
                BallSearchRoi(
                    bbox_xyxy=self._clip_roi([cx - 1.15 * bw, cy - 1.85 * bh, cx + 1.15 * bw, cy + 0.20 * bh], frame_shape),
                    source="airborne_player_reacquisition",
                    track_id=track_id,
                    motion_mode="airborne_reacquisition",
                )
            )
        return rois

    def _roi_around_player(self, frame_shape, bbox_xywh, motion_mode="unknown_recent"):
        cx, cy, bw, bh = [float(v) for v in bbox_xywh]
        if motion_mode == "dribble_like":
            return self._clip_roi([cx - 0.65 * bw, cy - 0.05 * bh, cx + 0.65 * bw, cy + 0.75 * bh], frame_shape)
        if motion_mode == "shot_or_lob":
            return self._clip_roi([cx - 0.80 * bw, cy - 1.45 * bh, cx + 0.80 * bw, cy + 0.20 * bh], frame_shape)
        half_w = max(32.0, 0.5 * bw * float(self.config.player_roi_width_scale))
        half_h = max(40.0, 0.5 * bh * float(self.config.player_roi_height_scale))
        return self._clip_roi([cx - half_w, cy - half_h, cx + half_w, cy + half_h], frame_shape)

    def _roi_for_motion(self, frame_shape, center_xy, ball_state, motion_context):
        velocity = ball_state.get("velocity_xy") or [0.0, 0.0]
        vx, vy = [float(v) for v in velocity]
        speed = float(np.linalg.norm(np.array([vx, vy], dtype=np.float32)))
        cx, cy = [float(v) for v in center_xy]
        mode = motion_context.motion_mode
        if mode == "dribble_like":
            radius_x = max(72.0, self.config.last_ball_radius_px * 0.75)
            return self._clip_roi([cx - radius_x, cy - 24.0, cx + radius_x, cy + 160.0 + 0.25 * max(0.0, vy)], frame_shape)
        if mode == "shot_or_lob":
            radius_x = max(96.0, self.config.last_ball_radius_px + abs(vx) * 1.0)
            return self._clip_roi([cx - radius_x, cy - 220.0 - max(0.0, -vy) * 1.2, cx + radius_x, cy + 80.0], frame_shape)
        if mode == "pass_or_loose":
            predicted_x = cx + vx
            predicted_y = cy + vy
            radius_x = max(96.0, self.config.last_ball_radius_px + abs(vx) * 1.4)
            radius_y = max(80.0, self.config.last_ball_radius_px + abs(vy) * 1.1)
            return self._clip_roi([predicted_x - radius_x, predicted_y - radius_y, predicted_x + radius_x, predicted_y + radius_y], frame_shape)
        if mode == "carry_or_hold" and motion_context.hand_xy:
            hx, hy = [float(v) for v in motion_context.hand_xy]
            return self._clip_roi([hx - 72.0, hy - 72.0, hx + 72.0, hy + 72.0], frame_shape)
        return self._roi_around_center(frame_shape, center_xy, self.config.last_ball_radius_px + 0.45 * speed)

    @staticmethod
    def _roi_around_center(frame_shape, center_xy, radius):
        cx, cy = [float(v) for v in center_xy]
        radius = float(radius)
        return BallSearchScheduler._clip_roi([cx - radius, cy - radius, cx + radius, cy + radius], frame_shape)

    @staticmethod
    def _clip_roi(roi, frame_shape):
        height, width = frame_shape[:2]
        x1, y1, x2, y2 = roi
        x1 = int(max(0, min(width - 1, np.floor(x1))))
        y1 = int(max(0, min(height - 1, np.floor(y1))))
        x2 = int(max(x1 + 1, min(width, np.ceil(x2))))
        y2 = int(max(y1 + 1, min(height, np.ceil(y2))))
        return [x1, y1, x2, y2]


class BallStateTracker:
    """Keep one conservative ball state through short detector misses.

    Stage 1 intentionally does not attempt full multi-hypothesis ball tracking.
    It selects at most one candidate per frame, applies simple continuity-aware
    scoring, and predicts through only short gaps.
    """

    def __init__(
        self,
        *,
        min_observed_confidence=BALL_MIN_OBSERVED_CONFIDENCE,
        max_predict_gap_frames=BALL_MAX_PREDICT_GAP_FRAMES,
        max_jump_px=BALL_MAX_JUMP_PX,
    ):
        self.min_observed_confidence = float(min_observed_confidence)
        self.max_predict_gap_frames = int(max_predict_gap_frames)
        self.max_jump_px = float(max_jump_px)
        self.center_xy = None
        self.velocity_xy = np.array([0.0, 0.0], dtype=np.float32)
        self.confidence = 0.0
        self.missing_gap_frames = 0

    def _size_score(self, bbox_xywh):
        width = float(bbox_xywh[2]) if len(bbox_xywh) >= 3 else 0.0
        height = float(bbox_xywh[3]) if len(bbox_xywh) >= 4 else 0.0
        diameter = max(width, height)
        if diameter < BALL_MIN_SIZE_PX or diameter > BALL_MAX_SIZE_PX:
            return 0.0
        midpoint = 0.5 * (BALL_MIN_SIZE_PX + BALL_MAX_SIZE_PX)
        spread = max(1.0, midpoint - BALL_MIN_SIZE_PX)
        return max(0.0, 1.0 - abs(diameter - midpoint) / spread)

    def _continuity_score(self, candidate_center):
        if self.center_xy is None:
            return 0.5
        distance = float(np.linalg.norm(candidate_center - self.center_xy))
        if distance >= self.max_jump_px:
            return 0.0
        return max(0.0, 1.0 - distance / self.max_jump_px)

    def _candidate_score(self, bbox_xywh, confidence, bootstrap_context=None, metadata=None):
        candidate_center = np.array([float(bbox_xywh[0]), float(bbox_xywh[1])], dtype=np.float32)
        confidence_score = max(0.0, min(1.0, float(confidence)))
        raw_continuity_score = self._continuity_score(candidate_center)
        continuity_score = raw_continuity_score
        size_score = self._size_score(bbox_xywh)
        foreground_prior = 0.0
        metadata = metadata or {}
        if bootstrap_context and bootstrap_context.get("enabled"):
            foreground_prior = play_region_prior_for_point(
                bootstrap_context,
                candidate_center[0],
                candidate_center[1],
            )
        source = str(metadata.get("source") or "")
        motion_mode = str(metadata.get("motion_mode") or "")
        airspace_prior = 0.20 if source in {"airborne_player_reacquisition", "pose_shot_pass_corridor"} or motion_mode in {"shot_or_lob", "pass_or_loose", "airborne_reacquisition"} else 0.0
        stale_reacquisition = self._is_stale_reacquisition_source(source)
        if stale_reacquisition:
            continuity_score = max(continuity_score, 0.35)
        score = (
            0.60 * confidence_score
            + 0.20 * continuity_score
            + 0.10 * size_score
            + 0.10 * max(foreground_prior, airspace_prior)
        )
        return score, candidate_center, {
            "confidence_score": confidence_score,
            "continuity_score": continuity_score,
            "raw_continuity_score": raw_continuity_score,
            "size_score": size_score,
            "foreground_prior": foreground_prior,
            "airspace_prior": airspace_prior,
            "stale_reacquisition": stale_reacquisition,
            "acceptance_threshold": self._candidate_acceptance_threshold(source, confidence_score, size_score),
            "mask_semantics": "ball_airspace_bonus_only",
        }

    def _is_stale_reacquisition_source(self, source):
        return (
            source in BALL_RECOVERY_SOURCES
            and self.center_xy is not None
            and self.missing_gap_frames >= self.max_predict_gap_frames
        )

    def _candidate_acceptance_threshold(self, source, confidence_score, size_score):
        if not self._is_stale_reacquisition_source(source):
            return self.min_observed_confidence
        if source == "pose_shot_pass_corridor" and confidence_score >= 0.07 and size_score >= 0.20:
            return 0.24
        if source == "airborne_player_reacquisition" and confidence_score >= 0.12 and size_score >= 0.30:
            return 0.26
        return self.min_observed_confidence

    def update(self, boxes_xywh, cls, conf, bootstrap_context=None, candidate_metadata=None):
        candidates = []
        candidate_metadata = candidate_metadata or []
        for idx, (bbox_xywh, class_id, confidence) in enumerate(zip(boxes_xywh, cls, conf)):
            if int(class_id) != BALL_CLASS_ID:
                continue
            metadata = candidate_metadata[idx] if idx < len(candidate_metadata) else {}
            score, center_xy, score_parts = self._candidate_score(
                bbox_xywh,
                confidence,
                bootstrap_context=bootstrap_context,
                metadata=metadata,
            )
            candidates.append(
                {
                    "bbox_xywh": [float(v) for v in bbox_xywh],
                    "center_xy": center_xy,
                    "confidence": float(confidence),
                    "score": float(score),
                    "score_parts": score_parts,
                    "source": metadata.get("source") or "unknown",
                    "roi_bbox_xyxy": metadata.get("roi_bbox_xyxy"),
                    "roi_track_id": metadata.get("track_id"),
                    "roi_motion_mode": metadata.get("motion_mode"),
                }
            )

        candidates.sort(key=lambda c: c["score"], reverse=True)
        selected = candidates[0] if candidates else None
        if selected and self._candidate_is_accepted(selected):
            self._annotate_candidate_decisions(candidates, selected)
            next_center = selected["center_xy"]
            if self.center_xy is not None:
                self.velocity_xy = next_center - self.center_xy
            self.center_xy = next_center
            self.confidence = float(selected["confidence"])
            self.missing_gap_frames = 0
            return {
                "state": "observed",
                "confidence": round(float(selected["confidence"]), 4),
                "center_xy": [round(float(next_center[0]), 3), round(float(next_center[1]), 3)],
                "bbox_xywh": [round(float(v), 3) for v in selected["bbox_xywh"]],
                "velocity_xy": [round(float(self.velocity_xy[0]), 3), round(float(self.velocity_xy[1]), 3)],
                "speed_px": round(float(np.linalg.norm(self.velocity_xy)), 3),
                "missing_gap_frames": 0,
                "source": "detector",
                "candidate_count": len(candidates),
                "candidate_scores": self._candidate_scores(candidates),
                "selected_candidate": self._candidate_payload(selected),
                "rejected_candidates": self._rejected_candidates(candidates),
            }

        self._annotate_candidate_decisions(candidates, None)
        if self.center_xy is not None and self.missing_gap_frames < self.max_predict_gap_frames:
            self.missing_gap_frames += 1
            self.center_xy = self.center_xy + self.velocity_xy
            return {
                "state": "predicted_short_gap",
                "confidence": round(float(self.confidence * 0.85), 4),
                "center_xy": [round(float(self.center_xy[0]), 3), round(float(self.center_xy[1]), 3)],
                "bbox_xywh": None,
                "velocity_xy": [round(float(self.velocity_xy[0]), 3), round(float(self.velocity_xy[1]), 3)],
                "speed_px": round(float(np.linalg.norm(self.velocity_xy)), 3),
                "missing_gap_frames": int(self.missing_gap_frames),
                "source": "smoothed_prediction",
                "candidate_count": len(candidates),
                "candidate_scores": self._candidate_scores(candidates),
                "selected_candidate": None,
                "rejected_candidates": self._rejected_candidates(candidates),
            }

        self.missing_gap_frames = self.max_predict_gap_frames + 1
        self.confidence = 0.0
        return {
            "state": "missing",
            "confidence": 0.0,
            "center_xy": None,
            "bbox_xywh": None,
            "velocity_xy": [0.0, 0.0],
            "speed_px": 0.0,
            "missing_gap_frames": int(self.missing_gap_frames),
            "source": "smoothed_prediction",
            "candidate_count": len(candidates),
            "candidate_scores": self._candidate_scores(candidates),
            "selected_candidate": None,
            "rejected_candidates": self._rejected_candidates(candidates),
        }

    def _annotate_candidate_decisions(self, candidates, selected):
        for candidate in candidates:
            accepted = candidate is selected
            candidate["accepted"] = bool(accepted)
            candidate["rejection_reasons"] = [] if accepted else self._candidate_rejection_reasons(candidate)

    def _candidate_is_accepted(self, candidate):
        score_parts = candidate.get("score_parts") or {}
        threshold = float(score_parts.get("acceptance_threshold") or self.min_observed_confidence)
        return float(candidate.get("score") or 0.0) >= threshold

    def _candidate_rejection_reasons(self, candidate):
        reasons = []
        score_parts = candidate.get("score_parts") or {}
        threshold = float(score_parts.get("acceptance_threshold") or self.min_observed_confidence)
        if float(candidate.get("score") or 0.0) < threshold:
            reasons.append("below_min_score")
        if float(score_parts.get("size_score") or 0.0) <= 0.0:
            reasons.append("invalid_ball_size")
        if self.center_xy is not None and float(score_parts.get("raw_continuity_score", score_parts.get("continuity_score") or 0.0) or 0.0) <= 0.0:
            reasons.append("large_jump_from_last_ball")
        if not reasons:
            reasons.append("lower_ranked_candidate")
        return reasons

    @staticmethod
    def _candidate_scores(candidates):
        return [
            BallStateTracker._candidate_payload(candidate)
            for candidate in candidates[:3]
        ]

    @staticmethod
    def _rejected_candidates(candidates):
        return [
            BallStateTracker._candidate_payload(candidate)
            for candidate in candidates
            if not candidate.get("accepted")
        ][:5]

    @staticmethod
    def _candidate_payload(candidate):
        score_parts = candidate.get("score_parts") or {}
        return {
            "bbox_xywh": [round(float(v), 3) for v in candidate.get("bbox_xywh") or []],
            "confidence": round(float(candidate.get("confidence") or 0.0), 4),
            "score": round(float(candidate.get("score") or 0.0), 4),
            "accepted": bool(candidate.get("accepted")),
            "source": candidate.get("source"),
            "roi_bbox_xyxy": candidate.get("roi_bbox_xyxy"),
            "roi_track_id": candidate.get("roi_track_id"),
            "roi_motion_mode": candidate.get("roi_motion_mode"),
            "rejection_reasons": list(candidate.get("rejection_reasons") or []),
            "score_parts": {
                "confidence_score": round(float(score_parts.get("confidence_score") or 0.0), 4),
                "continuity_score": round(float(score_parts.get("continuity_score") or 0.0), 4),
                "raw_continuity_score": round(float(score_parts.get("raw_continuity_score", score_parts.get("continuity_score") or 0.0) or 0.0), 4),
                "size_score": round(float(score_parts.get("size_score") or 0.0), 4),
                "foreground_prior": round(float(score_parts.get("foreground_prior") or 0.0), 4),
                "airspace_prior": round(float(score_parts.get("airspace_prior") or 0.0), 4),
                "stale_reacquisition": bool(score_parts.get("stale_reacquisition")),
                "acceptance_threshold": round(float(score_parts.get("acceptance_threshold") or 0.0), 4),
                "mask_semantics": score_parts.get("mask_semantics"),
            },
        }

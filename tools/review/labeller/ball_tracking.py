from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BallTrackingConfig:
    max_stale_frames: int = 8
    base_radius_px: float = 80.0


class BallTrackSmoother:
    """Own ball motion-mode inference and predictive search-window geometry."""

    def __init__(self, config=None):
        self.config = config or BallTrackingConfig()

    def infer_motion_mode(self, ball_detection, velocity_xy, nearby_player):
        vx, vy = [float(v) for v in velocity_xy]
        speed = float(np.linalg.norm(np.array([vx, vy], dtype=np.float32)))
        if speed >= 35.0 and vy <= -18.0:
            return "shot_or_lob"
        if nearby_player is None:
            return "pass_or_loose" if speed >= 45.0 else "unknown_recent"
        bbox = nearby_player.get("bbox_xyxy") or [0.0, 0.0, 0.0, 0.0]
        x1, y1, _x2, y2 = [float(v) for v in bbox]
        _, cy = [float(v) for v in (ball_detection.get("center_xy") or [0.0, 0.0])]
        box_h = max(1.0, y2 - y1)
        if speed < 35.0:
            return "carry_or_hold"
        if abs(vy) >= abs(vx) * 0.8 and cy >= y1 + 0.35 * box_h:
            return "dribble_like"
        if speed >= 45.0:
            return "pass_or_loose"
        return "unknown_recent"

    def state_motion_mode_from_velocity(self, velocity_xy, detection=None):
        explicit_mode = None
        if detection is not None:
            explicit_mode = detection.get("ball_motion_mode") or detection.get("motion_mode")
        if explicit_mode:
            return explicit_mode
        vx, vy = [float(v) for v in velocity_xy]
        speed = float(np.linalg.norm(np.array([vx, vy], dtype=np.float32)))
        if speed >= 35.0 and vy <= -18.0:
            return "shot_or_lob"
        if speed >= 45.0:
            return "pass_or_loose"
        return "unknown_recent"

    def predicted_confidence(self, previous_confidence, missing_gap_frames):
        return float(previous_confidence) * (0.85 ** float(missing_gap_frames))

    def keep_alive_kind(self, missing_gap_frames, motion_mode):
        if int(missing_gap_frames) > 4 and motion_mode in {"pass_or_loose", "shot_or_lob"}:
            return "extended_pass_shot"
        return "short_gap"

    def predictive_search_roi(self, frame_shape, predictive_state, *, frame_idx=None):
        if predictive_state is None:
            return None
        center_xy = predictive_state.get("center_xy")
        if not center_xy:
            return None
        cx, cy = [float(v) for v in center_xy]
        vx, vy = [float(v) for v in (predictive_state.get("velocity_xy") or [0.0, 0.0])]
        speed = float(np.linalg.norm(np.array([vx, vy], dtype=np.float32)))
        mode = predictive_state.get("motion_mode") or "unknown_recent"
        last_seen = int(predictive_state.get("last_seen_frame_idx") or 0)
        stale_frames = max(1, int(frame_idx) - last_seen) if frame_idx is not None else 1
        stale_frames = min(stale_frames, int(self.config.max_stale_frames))
        predicted_cx = cx + vx * stale_frames
        predicted_cy = cy + vy * stale_frames
        nearby_bbox = predictive_state.get("nearby_player_bbox_xyxy")
        uncertainty = 1.0 + 0.22 * max(0, stale_frames - 1)
        base_radius = float(self.config.base_radius_px)
        if mode == "carry_or_hold" and nearby_bbox and stale_frames <= 2:
            x1, y1, x2, y2 = [float(v) for v in nearby_bbox]
            box_w = max(1.0, x2 - x1)
            box_h = max(1.0, y2 - y1)
            roi = [x1 - 0.30 * box_w, y1 - 0.60 * box_h, x2 + 0.30 * box_w, y2 + 0.18 * box_h]
        elif mode == "dribble_like" and nearby_bbox and stale_frames <= 3:
            x1, y1, x2, y2 = [float(v) for v in nearby_bbox]
            box_w = max(1.0, x2 - x1)
            box_h = max(1.0, y2 - y1)
            roi = [x1 - 0.25 * box_w, y1 + 0.02 * box_h, x2 + 0.25 * box_w, y2 + 0.75 * box_h]
        elif mode == "shot_or_lob":
            radius_x = (base_radius + 1.15 * abs(vx) + 0.35 * speed * stale_frames) * uncertainty
            radius_y_up = (base_radius + 1.35 * max(0.0, -vy) + 0.55 * speed * stale_frames) * uncertainty
            radius_y_down = (base_radius + 0.65 * max(0.0, vy) + 0.35 * speed * stale_frames) * uncertainty
            roi = [predicted_cx - radius_x, predicted_cy - radius_y_up, predicted_cx + radius_x, predicted_cy + radius_y_down]
        elif mode == "pass_or_loose":
            radius_x = (base_radius + 1.35 * abs(vx) + 0.45 * speed * stale_frames) * uncertainty
            radius_y = (base_radius + 1.05 * abs(vy) + 0.35 * speed * stale_frames) * uncertainty
            roi = [predicted_cx - radius_x, predicted_cy - radius_y, predicted_cx + radius_x, predicted_cy + radius_y]
        else:
            radius = (base_radius + 0.9 * speed + 0.30 * speed * stale_frames) * uncertainty
            roi = [predicted_cx - radius, predicted_cy - radius, predicted_cx + radius, predicted_cy + radius]
        return self._clip_roi_xyxy(roi, frame_shape)

    @staticmethod
    def _clip_roi_xyxy(roi, frame_shape):
        height, width = frame_shape[:2]
        x1, y1, x2, y2 = roi
        x1 = int(max(0, min(width - 1, np.floor(x1))))
        y1 = int(max(0, min(height - 1, np.floor(y1))))
        x2 = int(max(x1 + 1, min(width, np.ceil(x2))))
        y2 = int(max(y1 + 1, min(height, np.ceil(y2))))
        return [x1, y1, x2, y2]

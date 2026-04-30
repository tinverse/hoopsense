"""Frame quality metrics for detection reliability analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FrameQualityRaw:
    frame_idx: int
    laplacian_variance: float
    tenengrad_score: float
    global_motion_px: float


def _percentile_rank(values: np.ndarray, value: float) -> float:
    if values.size == 0:
        return 100.0
    min_value = float(np.nanmin(values))
    max_value = float(np.nanmax(values))
    if max_value == min_value:
        return 100.0
    value = float(value)
    if value <= min_value:
        return 0.0
    if value >= max_value:
        return 100.0
    denom = max(float(values.size), 1.0)
    less = float(np.count_nonzero(values < value))
    equal = float(np.count_nonzero(values == value))
    return 100.0 * (less + 0.5 * equal) / denom


def _quality_label(sharpness_percentile: float, global_motion_percentile: float, global_motion_px: float) -> tuple[str, str]:
    high_global_motion = global_motion_percentile >= 85.0 or global_motion_px >= 8.0
    if sharpness_percentile <= 5.0:
        return "severe_blur", "camera_or_subject_motion" if high_global_motion else "low_edge_energy"
    if sharpness_percentile <= 15.0:
        return "blurred", "camera_or_subject_motion" if high_global_motion else "low_edge_energy"
    if sharpness_percentile <= 30.0:
        return "soft", "camera_or_subject_motion" if high_global_motion else "low_edge_energy"
    return "sharp", "normal"


def detector_policy_for_quality(label: str) -> str:
    if label == "severe_blur":
        return "bridge_measurements"
    if label == "blurred":
        return "downweight_measurements"
    if label == "soft":
        return "normal_with_low_confidence_evidence"
    return "normal"


def _prepare_gray(frame_bgr, *, long_side=320):
    import cv2

    if frame_bgr is None:
        return None
    frame = np.asarray(frame_bgr)
    if frame.size == 0:
        return None
    height, width = frame.shape[:2]
    if height <= 0 or width <= 0:
        return None
    scale = min(1.0, float(long_side) / float(max(height, width)))
    if scale < 1.0:
        frame = cv2.resize(
            frame,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    return np.asarray(gray, dtype=np.float32)


def _global_motion_px(previous_gray, current_gray, scale_to_original: float) -> float:
    if previous_gray is None or current_gray is None or previous_gray.shape != current_gray.shape:
        return 0.0
    try:
        import cv2

        shift, _response = cv2.phaseCorrelate(
            np.asarray(previous_gray, dtype=np.float32),
            np.asarray(current_gray, dtype=np.float32),
        )
        dx, dy = float(shift[0]), float(shift[1])
        return float((dx * dx + dy * dy) ** 0.5) * float(scale_to_original)
    except Exception:
        return 0.0


def compute_frame_quality_raw(frame_bgr, *, frame_idx: int, previous_gray=None, long_side=320) -> tuple[FrameQualityRaw, np.ndarray | None]:
    import cv2

    gray = _prepare_gray(frame_bgr, long_side=long_side)
    if gray is None:
        return FrameQualityRaw(int(frame_idx), 0.0, 0.0, 0.0), None
    height, width = np.asarray(frame_bgr).shape[:2]
    scale_to_original = float(max(height, width)) / float(max(gray.shape[:2]))
    laplacian_depth = getattr(cv2, "CV_32F", cv2.CV_64F)
    laplacian = cv2.Laplacian(gray, laplacian_depth)
    laplacian_variance = float(np.asarray(laplacian, dtype=np.float32).var())
    grad_y, grad_x = np.gradient(gray)
    tenengrad_score = float(np.mean((grad_x * grad_x) + (grad_y * grad_y)))
    global_motion_px = _global_motion_px(previous_gray, gray, scale_to_original)
    return (
        FrameQualityRaw(
            frame_idx=int(frame_idx),
            laplacian_variance=laplacian_variance,
            tenengrad_score=tenengrad_score,
            global_motion_px=global_motion_px,
        ),
        gray,
    )


def annotate_frame_quality(frames, *, frame_key="_frame_bgr", long_side=320) -> dict:
    raws = []
    previous_gray = None
    for frame in frames:
        raw, previous_gray = compute_frame_quality_raw(
            frame.get(frame_key),
            frame_idx=int(frame.get("frame_idx") or len(raws)),
            previous_gray=previous_gray,
            long_side=long_side,
        )
        raws.append(raw)
    return apply_frame_quality_raws(frames, raws, long_side=long_side)


def apply_frame_quality_raws(frames, raws, *, long_side=320) -> dict:
    lap_values = np.asarray([raw.laplacian_variance for raw in raws], dtype=np.float32)
    ten_values = np.asarray([raw.tenengrad_score for raw in raws], dtype=np.float32)
    motion_values = np.asarray([raw.global_motion_px for raw in raws], dtype=np.float32)
    labels = []
    for frame, raw in zip(frames, raws):
        lap_percentile = _percentile_rank(lap_values, raw.laplacian_variance)
        ten_percentile = _percentile_rank(ten_values, raw.tenengrad_score)
        sharpness_percentile = 0.5 * lap_percentile + 0.5 * ten_percentile
        motion_percentile = _percentile_rank(motion_values, raw.global_motion_px)
        label, blur_kind = _quality_label(sharpness_percentile, motion_percentile, raw.global_motion_px)
        labels.append(label)
        frame["frame_quality"] = {
            "laplacian_variance": round(float(raw.laplacian_variance), 4),
            "tenengrad_score": round(float(raw.tenengrad_score), 4),
            "global_motion_px": round(float(raw.global_motion_px), 4),
            "laplacian_percentile": round(float(lap_percentile), 2),
            "tenengrad_percentile": round(float(ten_percentile), 2),
            "sharpness_percentile": round(float(sharpness_percentile), 2),
            "global_motion_percentile": round(float(motion_percentile), 2),
            "quality_label": label,
            "blur_kind": blur_kind,
            "detector_policy": detector_policy_for_quality(label),
        }

    label_counts = {label: labels.count(label) for label in sorted(set(labels))}
    return {
        "enabled": True,
        "frame_count": len(frames),
        "method": "laplacian_variance_tenengrad_phase_correlation_v1",
        "long_side": int(long_side),
        "label_counts": label_counts,
        "threshold_policy": {
            "severe_blur": "sharpness_percentile <= 5",
            "blurred": "sharpness_percentile <= 15",
            "soft": "sharpness_percentile <= 30",
            "sharp": "sharpness_percentile > 30",
        },
    }


def summarize_detection_misses_by_quality(frames) -> dict:
    by_label = {}
    for frame in frames:
        label = ((frame.get("frame_quality") or {}).get("quality_label")) or "unknown"
        bucket = by_label.setdefault(
            label,
            {
                "frame_count": 0,
                "ball_missing_count": 0,
                "ball_observed_or_predicted_count": 0,
                "player_detection_count": 0,
                "active_player_detection_count": 0,
            },
        )
        bucket["frame_count"] += 1
        ball = frame.get("ball_state") or frame.get("ball_detection") or {}
        if ball.get("state") == "missing" or not ball:
            bucket["ball_missing_count"] += 1
        else:
            bucket["ball_observed_or_predicted_count"] += 1
        detections = frame.get("detections") or []
        bucket["player_detection_count"] += len(detections)
        bucket["active_player_detection_count"] += sum(1 for det in detections if det.get("active_player_candidate"))

    for bucket in by_label.values():
        frame_count = max(1, int(bucket["frame_count"]))
        bucket["ball_missing_rate"] = round(float(bucket["ball_missing_count"]) / float(frame_count), 4)
        bucket["avg_player_detections"] = round(float(bucket["player_detection_count"]) / float(frame_count), 4)
        bucket["avg_active_player_detections"] = round(float(bucket["active_player_detection_count"]) / float(frame_count), 4)
    return dict(sorted(by_label.items()))

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from pipelines.geometry import lift_keypoints_to_3d, project_pixel_to_court


REPO_ROOT = Path(__file__).resolve().parents[3]
CLIPS_DIR = REPO_ROOT / "data" / "raw_clips"
OUTPUT_DIR = REPO_ROOT / "data" / "review_artifacts" / "layer1"
CALIBRATION_FILE = REPO_ROOT / "data" / "training" / "camera_calibration.json"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COURT_X_RANGE = (-50.0, 2890.0)
COURT_Y_RANGE = (-50.0, 1575.0)
ACTIVE_PLAYER_SCORE_THRESHOLD = 0.55
MIN_PLAYER_BBOX_HEIGHT_RATIO = 0.08
EDGE_MARGIN_RATIO = 0.06
SHORT_GAP_REPAIR_MAX_GAP = 3
MOTION_SCORE_THRESHOLD_PX = 6.0
IDENTITY_REPAIR_SCORE_THRESHOLD = 0.72
IDENTITY_REPAIR_MAX_COURT_DISTANCE = 260.0
IDENTITY_REPAIR_MAX_CENTER_DISTANCE_RATIO = 1.35
JERSEY_OCR_MIN_CROP_WIDTH = 18
JERSEY_OCR_MIN_CROP_HEIGHT = 18
JERSEY_OCR_MIN_SHARPNESS = 25.0
JERSEY_OCR_MAX_SAMPLES_PER_IDENTITY = 5
JERSEY_OCR_CONSENSUS_MIN_VOTES = 2
JERSEY_OCR_CONSENSUS_MIN_SHARE = 0.62
_EASYOCR_READER = None
_EASYOCR_READER_ATTEMPTED = False


def _to_float_list(values):
    return [float(v) for v in values]


def _lerp(a, b, alpha):
    return (1.0 - alpha) * float(a) + alpha * float(b)


def _interpolate_point_list(points_a, points_b, alpha):
    if not points_a or not points_b or len(points_a) != len(points_b):
        return None
    interpolated = []
    for pt_a, pt_b in zip(points_a, points_b):
        if len(pt_a) != len(pt_b):
            return None
        interpolated.append([_lerp(a, b, alpha) for a, b in zip(pt_a, pt_b)])
    return interpolated


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


def _vector_norm(vec):
    if not vec or len(vec) < 2:
        return 0.0
    return float(np.linalg.norm([float(vec[0]), float(vec[1])]))


def _initialize_identity_fields(detection):
    canonical_track_id = detection.get("track_id")
    detection["identity_track_id"] = canonical_track_id
    detection["identity_track_source"] = "tracker"
    detection["identity_repair"] = None
    detection["identity_jersey_number"] = detection.get("identity_jersey_number")
    detection["identity_jersey_number_confidence"] = detection.get("identity_jersey_number_confidence")
    detection["identity_jersey_number_source"] = detection.get("identity_jersey_number_source")
    detection["identity_jersey_evidence_count"] = detection.get("identity_jersey_evidence_count")


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


def annotate_identity_jersey_numbers(frames):
    reader = get_easyocr_reader()
    identity_samples = {}
    for frame in frames:
        for detection in frame.get("detections", []):
            identity_track_id = detection.get("identity_track_id")
            if identity_track_id is None:
                detection["identity_jersey_evidence_count"] = 0
                continue
            detection["identity_jersey_evidence_count"] = 0
            evidence = _collect_jersey_evidence(reader, detection)
            if evidence is None:
                continue
            sample = {
                "frame_idx": frame.get("frame_idx"),
                "track_id": detection.get("track_id"),
                "identity_track_id": identity_track_id,
                **evidence,
            }
            bucket = identity_samples.setdefault(identity_track_id, [])
            bucket.append(sample)
            bucket.sort(key=lambda item: (item["sharpness"], item["ocr_confidence"]), reverse=True)
            del bucket[JERSEY_OCR_MAX_SAMPLES_PER_IDENTITY:]

    identity_consensus = {}
    for identity_track_id, samples in identity_samples.items():
        consensus = _resolve_identity_jersey_consensus(samples)
        if consensus is None:
            continue
        identity_consensus[identity_track_id] = {
            **consensus,
            "samples": samples,
        }

    for frame in frames:
        for detection in frame.get("detections", []):
            identity_track_id = detection.get("identity_track_id")
            consensus = identity_consensus.get(identity_track_id)
            if consensus is not None:
                detection["identity_jersey_number"] = consensus["number"]
                detection["identity_jersey_number_confidence"] = consensus["confidence"]
                detection["identity_jersey_number_source"] = consensus["source"]
                detection["identity_jersey_evidence_count"] = consensus["evidence_count"]
            detection.pop("_jersey_crop_bgr", None)
            detection.pop("_jersey_crop_sharpness", None)
    return {
        "reader_available": reader is not None,
        "identity_count_with_consensus": len(identity_consensus),
        "identity_consensus": {
            str(identity_track_id): {
                "number": consensus["number"],
                "confidence": consensus["confidence"],
                "vote_count": consensus["vote_count"],
                "evidence_count": consensus["evidence_count"],
                "votes": consensus["votes"],
                "samples": [
                    {
                        "frame_idx": sample["frame_idx"],
                        "track_id": sample["track_id"],
                        "candidate": sample["candidate"],
                        "raw_text": sample["raw_text"],
                        "ocr_confidence": sample["ocr_confidence"],
                        "sharpness": sample["sharpness"],
                    }
                    for sample in consensus["samples"]
                ],
            }
            for identity_track_id, consensus in sorted(identity_consensus.items())
        },
    }


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
):
    confidence = float(detection.get("confidence") or 0.0)
    bbox_xyxy = detection.get("bbox_xyxy") or [0.0, 0.0, 0.0, 0.0]
    x1, y1, x2, y2 = bbox_xyxy
    bbox_h = max(0.0, y2 - y1)
    bbox_cx = (x1 + x2) * 0.5
    bbox_cy = (y1 + y2) * 0.5

    reasons = {}
    score = 0.0

    reasons["confidence"] = round(confidence, 4)
    score += 0.35 * _clip01(confidence)

    height_ratio = bbox_h / max(float(frame_height), 1.0)
    height_score = _clip01((height_ratio - MIN_PLAYER_BBOX_HEIGHT_RATIO) / 0.18)
    reasons["height_ratio"] = round(height_ratio, 4)
    score += 0.20 * height_score

    edge_margin_x = frame_width * EDGE_MARGIN_RATIO
    edge_margin_y = frame_height * EDGE_MARGIN_RATIO
    edge_penalty = 0.0
    if bbox_cx < edge_margin_x or bbox_cx > frame_width - edge_margin_x:
        edge_penalty += 0.15
    if bbox_cy < edge_margin_y:
        edge_penalty += 0.05
    reasons["edge_penalty"] = round(edge_penalty, 4)
    score -= edge_penalty

    court_xy = detection.get("court_xy")
    if court_xy is not None and len(court_xy) == 2:
        cx, cy = float(court_xy[0]), float(court_xy[1])
        in_bounds = (
            COURT_X_RANGE[0] <= cx <= COURT_X_RANGE[1]
            and COURT_Y_RANGE[0] <= cy <= COURT_Y_RANGE[1]
        )
        reasons["court_in_bounds"] = in_bounds
        score += 0.20 if in_bounds else -0.20
    else:
        reasons["court_in_bounds"] = None

    persistence_score = min(track_frame_count, 5) / 5.0
    reasons["track_frame_count"] = int(track_frame_count)
    score += 0.10 * persistence_score

    motion_speed = float(detection.get("motion_speed_px") or 0.0)
    motion_score = _clip01(motion_speed / MOTION_SCORE_THRESHOLD_PX)
    reasons["motion_speed_px"] = round(motion_speed, 3)
    score += 0.15 * motion_score

    active_score = round(_clip01(score), 4)
    return {
        "score": active_score,
        "candidate": active_score >= ACTIVE_PLAYER_SCORE_THRESHOLD,
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

    if result.keypoints is not None and det_idx < len(result.keypoints):
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

    return detection


def _track_frame_counts(frames):
    counts = {}
    for frame in frames:
        seen = set()
        for detection in frame.get("detections", []):
            track_id = detection.get("track_id")
            if track_id is None or track_id in seen:
                continue
            counts[track_id] = counts.get(track_id, 0) + 1
            seen.add(track_id)
    return counts


def annotate_active_players(frames, video_meta):
    track_counts = _track_frame_counts(frames)
    frame_width = int(video_meta["width"])
    frame_height = int(video_meta["height"])
    for frame in frames:
        for detection in frame.get("detections", []):
            score_info = score_active_player(
                detection,
                frame_width=frame_width,
                frame_height=frame_height,
                track_frame_count=track_counts.get(detection.get("track_id"), 1),
            )
            detection["active_player_score"] = score_info["score"]
            detection["active_player_candidate"] = score_info["candidate"]
            detection["active_player_reasons"] = score_info["reasons"]
    return frames


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


def _score_identity_link(start_det, end_det, gap_frames, fps):
    if not start_det.get("active_player_candidate") or not end_det.get("active_player_candidate"):
        return None

    start_bucket = start_det.get("uniform_bucket")
    end_bucket = end_det.get("uniform_bucket")
    if (
        start_bucket in {"dark", "light"}
        and end_bucket in {"dark", "light"}
        and start_bucket != end_bucket
    ):
        return None

    dt = max(1.0 / max(float(fps), 1.0), 1e-3)
    elapsed_s = max(gap_frames + 1, 1) * dt
    start_cx, start_cy = _bbox_center_xy(start_det)
    end_cx, end_cy = _bbox_center_xy(end_det)
    vel_x, vel_y = (start_det.get("smoothed_velocity_xy") or [0.0, 0.0])[:2]
    predicted_x = float(start_cx) + float(vel_x) * elapsed_s
    predicted_y = float(start_cy) + float(vel_y) * elapsed_s
    center_distance = float(np.linalg.norm([end_cx - predicted_x, end_cy - predicted_y]))

    _start_w, start_h = _bbox_size_xy(start_det)
    _end_w, end_h = _bbox_size_xy(end_det)
    avg_height = max(1.0, (start_h + end_h) * 0.5)
    max_center_distance = max(40.0, avg_height * IDENTITY_REPAIR_MAX_CENTER_DISTANCE_RATIO)
    if center_distance > max_center_distance:
        return None
    position_score = _clip01(1.0 - (center_distance / max_center_distance))

    start_w, start_h = _bbox_size_xy(start_det)
    end_w, end_h = _bbox_size_xy(end_det)
    width_ratio = min(start_w, end_w) / max(start_w, end_w, 1.0)
    height_ratio = min(start_h, end_h) / max(start_h, end_h, 1.0)
    size_score = (width_ratio + height_ratio) * 0.5
    if size_score < 0.55:
        return None

    start_vel = np.array((start_det.get("smoothed_velocity_xy") or [0.0, 0.0])[:2], dtype=float)
    end_vel = np.array((end_det.get("smoothed_velocity_xy") or [0.0, 0.0])[:2], dtype=float)
    start_speed = float(np.linalg.norm(start_vel))
    end_speed = float(np.linalg.norm(end_vel))
    if start_speed > 1.0 and end_speed > 1.0:
        velocity_score = _clip01((float(np.dot(start_vel, end_vel)) / (start_speed * end_speed) + 1.0) * 0.5)
    else:
        velocity_score = 0.5

    if start_bucket == end_bucket and start_bucket in {"dark", "light"}:
        uniform_score = 1.0
    elif start_bucket == "unknown" or end_bucket == "unknown":
        uniform_score = 0.55
    else:
        uniform_score = 0.0

    court_score = 0.5
    if start_det.get("court_xy") and end_det.get("court_xy"):
        court_distance = float(
            np.linalg.norm(
                [
                    float(end_det["court_xy"][0]) - float(start_det["court_xy"][0]),
                    float(end_det["court_xy"][1]) - float(start_det["court_xy"][1]),
                ]
            )
        )
        if court_distance > IDENTITY_REPAIR_MAX_COURT_DISTANCE:
            return None
        court_score = _clip01(1.0 - (court_distance / IDENTITY_REPAIR_MAX_COURT_DISTANCE))

    gap_score = _clip01(1.0 - ((gap_frames - 1) / max(SHORT_GAP_REPAIR_MAX_GAP, 1)))
    confidence_score = min(float(start_det.get("confidence") or 0.0), float(end_det.get("confidence") or 0.0))
    link_score = (
        0.34 * position_score
        + 0.18 * velocity_score
        + 0.14 * size_score
        + 0.14 * uniform_score
        + 0.10 * court_score
        + 0.06 * gap_score
        + 0.04 * confidence_score
    )
    return {
        "score": round(float(link_score), 4),
        "gap_frames": int(gap_frames),
        "reasons": {
            "position_score": round(position_score, 4),
            "velocity_score": round(velocity_score, 4),
            "size_score": round(size_score, 4),
            "uniform_score": round(uniform_score, 4),
            "court_score": round(court_score, 4),
            "gap_score": round(gap_score, 4),
            "confidence_score": round(confidence_score, 4),
            "predicted_center_distance_px": round(center_distance, 3),
        },
    }


def _find_identity_links(track_frames, fps, *, max_gap=SHORT_GAP_REPAIR_MAX_GAP):
    track_ids = sorted(track_frames)
    candidates = []
    for predecessor_track_id in track_ids:
        predecessor_obs = sorted(track_frames.get(predecessor_track_id, []), key=lambda item: item[0])
        if not predecessor_obs:
            continue
        predecessor_last_frame, predecessor_last_det = predecessor_obs[-1]
        for successor_track_id in track_ids:
            if successor_track_id == predecessor_track_id:
                continue
            successor_obs = sorted(track_frames.get(successor_track_id, []), key=lambda item: item[0])
            if not successor_obs:
                continue
            successor_first_frame, successor_first_det = successor_obs[0]
            gap_frames = successor_first_frame - predecessor_last_frame - 1
            if gap_frames <= 0 or gap_frames > max_gap:
                continue
            link = _score_identity_link(predecessor_last_det, successor_first_det, gap_frames, fps)
            if link is None or link["score"] < IDENTITY_REPAIR_SCORE_THRESHOLD:
                continue
            candidates.append(
                {
                    "predecessor_track_id": predecessor_track_id,
                    "successor_track_id": successor_track_id,
                    "start_det": predecessor_last_det,
                    "end_det": successor_first_det,
                    **link,
                }
            )

    links_by_predecessor = {}
    links_by_successor = {}
    for candidate in sorted(candidates, key=lambda item: (-item["score"], item["gap_frames"], item["successor_track_id"])):
        predecessor_track_id = candidate["predecessor_track_id"]
        successor_track_id = candidate["successor_track_id"]
        if predecessor_track_id in links_by_predecessor or successor_track_id in links_by_successor:
            continue
        competing_predecessor_scores = sorted(
            other["score"]
            for other in candidates
            if other["predecessor_track_id"] == predecessor_track_id and other["successor_track_id"] != successor_track_id
        )
        competing_successor_scores = sorted(
            other["score"]
            for other in candidates
            if other["successor_track_id"] == successor_track_id and other["predecessor_track_id"] != predecessor_track_id
        )
        best_competitor = max(competing_predecessor_scores + competing_successor_scores, default=0.0)
        if best_competitor >= candidate["score"] - 0.05:
            continue
        links_by_predecessor[predecessor_track_id] = candidate
        links_by_successor[successor_track_id] = candidate
    return list(links_by_successor.values())


def repair_short_track_gaps(frames, video_meta=None, *, max_gap=SHORT_GAP_REPAIR_MAX_GAP):
    frame_index = {frame["frame_idx"]: frame for frame in frames}
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

    identity_links = _find_identity_links(track_frames, fps, max_gap=max_gap)

    for link in identity_links:
        predecessor_track_id = link["predecessor_track_id"]
        successor_track_id = link["successor_track_id"]
        canonical_track_id = track_frames[predecessor_track_id][0][1].get("identity_track_id", predecessor_track_id)
        repair_meta = {
            "kind": "short_gap_identity_bridge",
            "predecessor_track_id": predecessor_track_id,
            "successor_track_id": successor_track_id,
            "canonical_track_id": canonical_track_id,
            "gap_frames": link["gap_frames"],
            "link_score": link["score"],
            "reasons": link["reasons"],
        }
        for _frame_idx, detection in track_frames[successor_track_id]:
            detection["identity_track_id"] = canonical_track_id
            detection["identity_track_source"] = "repaired"
            detection["identity_repair"] = repair_meta

    for track_id, observations in track_frames.items():
        observations.sort(key=lambda item: item[0])
        for (start_idx, start_det), (end_idx, end_det) in zip(observations, observations[1:]):
            gap = end_idx - start_idx - 1
            if gap <= 0 or gap > max_gap:
                continue
            if not start_det.get("active_player_candidate") or not end_det.get("active_player_candidate"):
                continue
            for missing_frame_idx in range(start_idx + 1, end_idx):
                frame = frame_index.get(missing_frame_idx)
                if frame is None:
                    continue
                existing_track_ids = {
                    detection.get("track_id")
                    for detection in frame.get("detections", [])
                }
                if track_id in existing_track_ids:
                    continue
                alpha = (missing_frame_idx - start_idx) / float(end_idx - start_idx)
                synthesized = _interpolate_detection(
                    start_det,
                    end_det,
                    missing_frame_idx,
                    frame["t_ms"],
                    alpha,
                )
                frame.setdefault("detections", []).append(synthesized)

    for link in identity_links:
        start_idx = link["start_det"]["_frame_idx"]
        end_idx = link["end_det"]["_frame_idx"]
        start_det = link["start_det"]
        end_det = link["end_det"]
        canonical_track_id = start_det.get("identity_track_id", start_det.get("track_id"))
        for missing_frame_idx in range(start_idx + 1, end_idx):
            frame = frame_index.get(missing_frame_idx)
            if frame is None:
                continue
            existing_identity_ids = {
                detection.get("identity_track_id", detection.get("track_id"))
                for detection in frame.get("detections", [])
            }
            if canonical_track_id in existing_identity_ids:
                continue
            alpha = (missing_frame_idx - start_idx) / float(end_idx - start_idx)
            synthesized = _interpolate_detection(
                start_det,
                end_det,
                missing_frame_idx,
                frame["t_ms"],
                alpha,
            )
            synthesized["identity_track_id"] = canonical_track_id
            synthesized["identity_track_source"] = "repaired"
            synthesized["identity_repair"] = {
                "kind": "short_gap_identity_bridge",
                "predecessor_track_id": link["predecessor_track_id"],
                "successor_track_id": link["successor_track_id"],
                "canonical_track_id": canonical_track_id,
                "gap_frames": link["gap_frames"],
                "link_score": link["score"],
                "reasons": link["reasons"],
            }
            frame.setdefault("detections", []).append(synthesized)

    for frame in frames:
        frame["detections"].sort(
            key=lambda detection: (
                detection.get("identity_track_id") is None,
                detection.get("identity_track_id") if detection.get("identity_track_id") is not None else 1_000_000,
                detection.get("track_id") is None,
                detection.get("track_id") if detection.get("track_id") is not None else 1_000_000,
                not detection.get("synthesized", False),
            )
        )
        for detection in frame.get("detections", []):
            detection.pop("_frame_idx", None)
            detection.pop("_t_ms", None)
    return frames


def annotate_clip(video_path, output_path, model_name, device, conf_threshold, calibration_file):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    model = YOLO(model_name)
    frames = []
    frame_idx = 0
    clip_id = video_path.stem
    calibration = load_calibration(clip_id, calibration_file)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t_ms = int((frame_idx / fps) * 1000)
        results = model.track(
            frame,
            persist=True,
            classes=[0],
            conf=conf_threshold,
            device=device,
            verbose=False,
        )

        detections = []
        h_matrix = get_frame_homography(calibration, frame_idx)
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            result = results[0]
            for det_idx in range(len(result.boxes)):
                detections.append(build_detection(result, det_idx, model.names, frame, h_matrix=h_matrix))

        frames.append({
            "frame_idx": frame_idx,
            "t_ms": t_ms,
            "calibrated": h_matrix is not None,
            "detections": detections,
        })
        frame_idx += 1

    cap.release()

    relative_path = str(video_path.relative_to(CLIPS_DIR))
    video_meta = {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
    }
    frames = smooth_track_motion(frames, video_meta)
    frames = annotate_active_players(frames, video_meta)
    frames = repair_short_track_gaps(frames, video_meta)
    frames = smooth_track_motion(frames, video_meta)
    frames = annotate_active_players(frames, video_meta)
    jersey_ocr = annotate_identity_jersey_numbers(frames)

    artifact = {
        "schema_version": "1.0.0",
        "clip_id": clip_id,
        "video_path": relative_path,
        "video": video_meta,
        "model": {
            "name": model_name,
            "task": "pose_track",
            "device": device,
            "classes": ["person"],
        },
        "postprocess": {
            "active_player_score_threshold": ACTIVE_PLAYER_SCORE_THRESHOLD,
            "short_gap_repair_max_gap": SHORT_GAP_REPAIR_MAX_GAP,
            "identity_repair_score_threshold": IDENTITY_REPAIR_SCORE_THRESHOLD,
            "motion_score_threshold_px": MOTION_SCORE_THRESHOLD_PX,
            "track_motion_smoother": "kalman_constant_velocity_v1",
            "identity_repair": "trajectory_bridge_v1",
            "jersey_ocr": {
                "backend": "easyocr_v1",
                "reader_available": jersey_ocr["reader_available"],
                "max_samples_per_identity": JERSEY_OCR_MAX_SAMPLES_PER_IDENTITY,
                "min_sharpness": JERSEY_OCR_MIN_SHARPNESS,
                "consensus_min_votes": JERSEY_OCR_CONSENSUS_MIN_VOTES,
                "consensus_min_share": JERSEY_OCR_CONSENSUS_MIN_SHARE,
                "identity_count_with_consensus": jersey_ocr["identity_count_with_consensus"],
            },
        },
        "calibration": {
            "enabled": calibration is not None,
            "source": str(calibration_file.relative_to(REPO_ROOT)) if calibration_file and calibration_file.exists() else None,
            "type": calibration.get("type") if calibration else None,
        },
        "identity_jersey_consensus": jersey_ocr["identity_consensus"],
        "frames": frames,
    }

    with open(output_path, "w") as f:
        json.dump(artifact, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Generate Layer 1 annotations for the labeller.")
    parser.add_argument("clip_path", help="Absolute or repo-relative path to the clip")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--model", default="yolov8n-pose.pt", help="Ultralytics model name or path")
    parser.add_argument("--device", default="cuda:0", help="Ultralytics device selector")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
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
        device=args.device,
        conf_threshold=args.conf,
        calibration_file=calibration_file,
    )
    print(output_path)


if __name__ == "__main__":
    main()

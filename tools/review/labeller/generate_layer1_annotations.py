import argparse
import json
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


def _to_float_list(values):
    return [float(v) for v in values]


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

    return detection


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
    artifact = {
        "schema_version": "1.0.0",
        "clip_id": clip_id,
        "video_path": relative_path,
        "video": {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
        },
        "model": {
            "name": model_name,
            "task": "pose_track",
            "device": device,
            "classes": ["person"],
        },
        "calibration": {
            "enabled": calibration is not None,
            "source": str(calibration_file.relative_to(REPO_ROOT)) if calibration_file and calibration_file.exists() else None,
            "type": calibration.get("type") if calibration else None,
        },
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

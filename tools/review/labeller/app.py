import json
import cv2
import numpy as np
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from tools.review.labeller.sam_refiner import (
    DEFAULT_SAM3_CHECKPOINT,
    DEFAULT_SAM3_REPO_MODEL,
    DEFAULT_SAM3_TEXT_PROMPT,
    _is_hf_available,
    _is_sam3_available,
    _normalize_binary_mask,
)

app = Flask(__name__)

# Base paths
FILE_PATH = Path(__file__).resolve()
REPO_ROOT = FILE_PATH.parent.parent.parent.parent
CLIPS_DIR = REPO_ROOT / "data" / "raw_clips"
TRAINING_DIR = REPO_ROOT / "data" / "training"
PERCEPTION_DIR = REPO_ROOT / "data" / "review_artifacts" / "layer1"
PERCEPTION_FEEDBACK_FILE = TRAINING_DIR / "perception_feedback.jsonl"
GT_FILE = TRAINING_DIR / "manual_gt.jsonl"
CALIBRATION_FILE = TRAINING_DIR / "camera_calibration.json"
VALID_FEEDBACK_ISSUES = {
    "false_positive",
    "false_negative",
    "merge_error",
    "track_error",
    "pose_error",
    "live_play",
    "dead_ball",
    "uncertain_play_state",
    "general_note",
}
SAM3_INTERACTIVE_DEVICE = "cuda:0"
SAM3_INTERACTIVE_MAX_PROPOSALS = 12
_SAM3_INTERACTIVE_SESSION = None


def _line_primitive(label, family, endpoints_xy):
    return {
        "label": label,
        "family": family,
        "kind": "line",
        "world_points": [
            [float(endpoints_xy[0][0]), float(endpoints_xy[0][1])],
            [float(endpoints_xy[1][0]), float(endpoints_xy[1][1])],
        ],
    }


def _arc_primitive(label, family, center_xy, radius, start_deg, end_deg):
    return {
        "label": label,
        "family": family,
        "kind": "arc",
        "center_xy": [float(center_xy[0]), float(center_xy[1])],
        "radius": float(radius),
        "start_deg": float(start_deg),
        "end_deg": float(end_deg),
    }


def _point_primitive(label, family, xy):
    return {
        "label": label,
        "family": family,
        "kind": "point",
        "world_xy": [float(xy[0]), float(xy[1])],
    }

# Ensure directories exist
CLIPS_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_DIR.mkdir(parents=True, exist_ok=True)
PERCEPTION_DIR.mkdir(parents=True, exist_ok=True)

# Calibration primitives (NCAA cm). Reviewers can click arbitrary samples
# along visible lines and arcs instead of needing precise named court points.
CALIBRATION_PRIMITIVES = {
    "far_sideline": _line_primitive("Far Sideline", "sideline", [[0, 0], [2865, 0]]),
    "near_sideline": _line_primitive("Near Sideline", "sideline", [[0, 1524], [2865, 1524]]),
    "left_baseline": _line_primitive("Left Baseline", "baseline", [[0, 0], [0, 1524]]),
    "right_baseline": _line_primitive("Right Baseline", "baseline", [[2865, 0], [2865, 1524]]),
    "left_lane_line": _line_primitive("Left Lane Edge", "lane", [[580, 517], [580, 1007]]),
    "right_lane_line": _line_primitive("Right Lane Edge", "lane", [[2285, 517], [2285, 1007]]),
    "left_free_throw_line": _line_primitive("Left Free Throw Line", "lane", [[580, 517], [580, 1007]]),
    "right_free_throw_line": _line_primitive("Right Free Throw Line", "lane", [[2285, 517], [2285, 1007]]),
    "left_three_point_arc": _arc_primitive("Left 3PT Arc", "three_point_arc", [160, 762], 675, -72, 72),
    "right_three_point_arc": _arc_primitive("Right 3PT Arc", "three_point_arc", [2705, 762], 675, 108, 252),
    "left_rim": _point_primitive("Left Rim", "rim", [160, 762]),
    "right_rim": _point_primitive("Right Rim", "rim", [2705, 762]),
}


def get_point_identifier(point):
    return point.get("primitive_id") or point.get("landmark_id")


def get_primitive_spec(point_or_id):
    primitive_id = point_or_id if isinstance(point_or_id, str) else get_point_identifier(point_or_id)
    if primitive_id is None:
        return None
    return CALIBRATION_PRIMITIVES.get(primitive_id)


def get_world_point_for_sample(primitive_id, alpha=0.5):
    spec = get_primitive_spec(primitive_id)
    if spec is None:
        return None
    alpha = max(0.0, min(1.0, float(alpha)))
    if spec["kind"] == "point":
        return spec["world_xy"]
    if spec["kind"] == "line":
        start, end = spec["world_points"]
        return [
            (1.0 - alpha) * start[0] + alpha * end[0],
            (1.0 - alpha) * start[1] + alpha * end[1],
        ]
    if spec["kind"] == "arc":
        theta = np.deg2rad((1.0 - alpha) * spec["start_deg"] + alpha * spec["end_deg"])
        center_x, center_y = spec["center_xy"]
        radius = spec["radius"]
        return [
            center_x + radius * np.cos(theta),
            center_y + radius * np.sin(theta),
        ]
    return None


def summarize_calibration_points(points_data):
    families = {}
    unique_primitive_ids = set()
    for point in points_data:
        primitive_id = get_point_identifier(point)
        spec = get_primitive_spec(primitive_id)
        if spec is None:
            continue
        unique_primitive_ids.add(primitive_id)
        family = spec["family"]
        families[family] = families.get(family, 0) + 1
    return {
        "point_count": len(points_data),
        "unique_primitive_count": len(unique_primitive_ids),
        "family_counts": families,
    }


def validate_calibration_points(points_data):
    unknown = []
    unique_primitive_ids = set()
    families = set()
    for point in points_data:
        primitive_id = get_point_identifier(point)
        spec = get_primitive_spec(primitive_id)
        if spec is None:
            unknown.append(primitive_id)
            continue
        unique_primitive_ids.add(primitive_id)
        families.add(spec["family"])

    if unknown:
        return {"ok": False, "reason": "unknown_primitives", "unknown_primitives": sorted(set(unknown))}
    if len(points_data) < 4:
        return {"ok": False, "reason": "too_few_points", "required_points": 4}
    if len(unique_primitive_ids) < 2:
        return {"ok": False, "reason": "too_few_unique_primitives", "required_primitives": 2}
    if len(families) < 2:
        return {"ok": False, "reason": "too_few_primitive_families", "required_families": 2}
    return {"ok": True}

def get_perception_artifact_path(clip_id):
    return PERCEPTION_DIR / f"{clip_id}.perception.json"


def load_perception_artifact(clip_id):
    artifact_path = get_perception_artifact_path(clip_id)
    if not artifact_path.exists():
        return None
    with open(artifact_path, "r") as f:
        return json.load(f)


def build_disabled_perception_payload(clip_id, reason):
    return {
        "enabled": False,
        "clip_id": clip_id,
        "frames": [],
        "status": reason,
    }


def find_artifact_frame(artifact, frame_idx):
    for frame in artifact.get("frames", []):
        if frame.get("frame_idx") == frame_idx:
            return frame
    return None


class Sam3InteractiveSession:
    def __init__(
        self,
        *,
        model_name=DEFAULT_SAM3_REPO_MODEL,
        checkpoint_filename=DEFAULT_SAM3_CHECKPOINT,
        device=SAM3_INTERACTIVE_DEVICE,
    ):
        self.model_name = model_name
        self.checkpoint_filename = checkpoint_filename
        self.device = device
        self.model = None
        self.processor = None

    def _gpu_ready(self):
        if not str(self.device).startswith("cuda"):
            return False
        import torch

        return bool(torch.cuda.is_available())

    def load(self):
        if self.model is not None and self.processor is not None:
            return self
        if not _is_sam3_available():
            raise RuntimeError("sam3 repo package is not installed")
        if not _is_hf_available():
            raise RuntimeError("huggingface_hub is not available")
        if not self._gpu_ready():
            raise RuntimeError("cuda_unavailable")

        from huggingface_hub import hf_hub_download
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        checkpoint_path = hf_hub_download(
            repo_id=self.model_name,
            filename=self.checkpoint_filename,
        )
        runtime_device = "cuda" if str(self.device).startswith("cuda") else self.device
        self.model = build_sam3_image_model(
            checkpoint_path=str(Path(checkpoint_path)),
            load_from_HF=False,
            device=runtime_device,
            eval_mode=True,
        )
        self.processor = Sam3Processor(self.model, device=runtime_device)
        return self

    def detect(self, frame_bgr, prompt):
        self.load()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image_height, image_width = frame_bgr.shape[:2]
        torch = None
        autocast_context = None
        if str(self.device).startswith("cuda"):
            import torch as _torch

            torch = _torch
            autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        from PIL import Image

        if autocast_context is not None:
            with autocast_context:
                state = self.processor.set_image(Image.fromarray(frame_rgb))
                output = self.processor.set_text_prompt(state=state, prompt=prompt)
        else:
            state = self.processor.set_image(Image.fromarray(frame_rgb))
            output = self.processor.set_text_prompt(state=state, prompt=prompt)

        masks = output.get("masks")
        scores = output.get("scores")
        if masks is None or scores is None or len(scores) == 0:
            return []
        score_values = scores.detach().float()
        proposals = []
        for idx in range(len(score_values)):
            mask = _normalize_binary_mask(masks[idx].detach().float().cpu().numpy())
            if mask is None or mask.sum() <= 0:
                continue
            if mask.shape[:2] != frame_bgr.shape[:2]:
                mask = cv2.resize(mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
                mask = np.array(mask > 0, dtype=np.uint8)
            ys, xs = np.where(mask > 0)
            if len(xs) == 0 or len(ys) == 0:
                continue
            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            area_px = int(mask.sum())
            bbox_w = max(1, x2 - x1)
            bbox_h = max(1, y2 - y1)
            mask_points = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            polygons = []
            for contour in mask_points:
                if len(contour) < 3:
                    continue
                epsilon = 0.01 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                polygons.append([[int(pt[0][0]), int(pt[0][1])] for pt in approx])
            proposals.append({
                "score": round(float(score_values[idx].item()), 4),
                "bbox_xyxy": [x1, y1, x2, y2],
                "bbox_xywh": [round(float((x1 + x2) * 0.5), 2), round(float((y1 + y2) * 0.5), 2), float(bbox_w), float(bbox_h)],
                "mask_area_px": area_px,
                "mask_area_ratio": round(area_px / max(float(image_width * image_height), 1.0), 6),
                "polygons_xy": polygons,
            })
        proposals.sort(key=lambda item: float(item["score"]), reverse=True)
        if torch is not None:
            torch.cuda.empty_cache()
        return proposals[:SAM3_INTERACTIVE_MAX_PROPOSALS]


def get_sam3_interactive_session():
    global _SAM3_INTERACTIVE_SESSION
    if _SAM3_INTERACTIVE_SESSION is None:
        _SAM3_INTERACTIVE_SESSION = Sam3InteractiveSession()
    return _SAM3_INTERACTIVE_SESSION


def resolve_clip_video_path(clip_id, relative_path=None):
    if relative_path:
        candidate = CLIPS_DIR / relative_path
        if candidate.exists():
            return candidate
    for domain in ("nba", "ncaa", "youth", "playground"):
        candidate = CLIPS_DIR / domain / f"{clip_id}.mp4"
        if candidate.exists():
            return candidate
    return None


def load_frame_bgr(video_path, frame_idx):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_idx)))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")
    return frame


def resolve_frame_index(video_path, frame_idx=None, t_ms=None):
    if frame_idx is not None:
        return max(0, int(frame_idx))
    if t_ms is None:
        raise RuntimeError("frame_idx_missing")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return max(0, int(round((float(t_ms) / 1000.0) * float(fps))))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/clips')
def list_clips():
    clips = []
    domains = ["nba", "ncaa", "youth", "playground"]
    for domain in domains:
        domain_path = CLIPS_DIR / domain
        if domain_path.exists():
            for video in domain_path.glob("*.mp4"):
                if not get_perception_artifact_path(video.stem).exists():
                    continue
                clips.append({
                    "id": video.stem,
                    "domain": domain,
                    "path": f"{domain}/{video.name}"
                })
    return jsonify(clips)


@app.route('/api/landmarks')
def get_landmarks():
    return jsonify(CALIBRATION_PRIMITIVES)


@app.route('/api/perception/<clip_id>')
def get_perception_overlay(clip_id):
    artifact = load_perception_artifact(clip_id)
    if not artifact:
        return jsonify(build_disabled_perception_payload(clip_id, "missing"))
    if artifact.get("clip_id") != clip_id:
        return jsonify(build_disabled_perception_payload(clip_id, "clip_id_mismatch")), 409
    artifact["enabled"] = True
    artifact["status"] = "ready"
    return jsonify(artifact)


@app.route('/api/sam_detect', methods=['POST'])
def sam_detect():
    data = request.json or {}
    clip_id = data.get("clip_id")
    prompt = str(data.get("prompt") or DEFAULT_SAM3_TEXT_PROMPT).strip()
    if not clip_id:
        return jsonify({"status": "error", "reason": "clip_id_missing"}), 400
    if not prompt:
        return jsonify({"status": "error", "reason": "prompt_missing"}), 400

    frame_idx = data.get("frame_idx")
    t_ms = data.get("t_ms")
    artifact = load_perception_artifact(clip_id)
    if frame_idx is None and t_ms is not None and artifact:
        target_ms = float(t_ms)
        best_frame = None
        best_delta = None
        for frame in artifact.get("frames", []):
            delta = abs(float(frame.get("t_ms") or 0.0) - target_ms)
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_frame = frame
        if best_frame is not None:
            frame_idx = int(best_frame.get("frame_idx") or 0)
    video_path = resolve_clip_video_path(clip_id, data.get("clip_path"))
    if video_path is None:
        return jsonify({"status": "error", "reason": "clip_missing"}), 404

    try:
        resolved_frame_idx = resolve_frame_index(video_path, frame_idx=frame_idx, t_ms=t_ms)
        frame_bgr = load_frame_bgr(video_path, resolved_frame_idx)
        session = get_sam3_interactive_session()
        proposals = session.detect(frame_bgr, prompt)
    except Exception as exc:
        return jsonify({
            "status": "error",
            "reason": "sam_detect_failed",
            "detail": str(exc),
        }), 500

    return jsonify({
        "status": "success",
        "clip_id": clip_id,
        "clip_path": data.get("clip_path"),
        "frame_idx": int(resolved_frame_idx),
        "t_ms": int(data.get("t_ms") or 0),
        "prompt": prompt,
        "proposal_count": len(proposals),
        "proposals": proposals,
    })


@app.route('/api/perception_feedback', methods=['POST'])
def save_perception_feedback():
    data = request.json or {}
    required = ["clip_id", "frame_idx", "t_ms", "issue_type", "timestamp"]
    missing = [field for field in required if field not in data]
    if missing:
        return jsonify({
            "status": "error",
            "missing": missing
        }), 400
    if data["issue_type"] not in VALID_FEEDBACK_ISSUES:
        return jsonify({
            "status": "error",
            "invalid_issue_type": data["issue_type"],
        }), 400
    artifact = load_perception_artifact(data["clip_id"])
    if not artifact:
        return jsonify({
            "status": "error",
            "reason": "artifact_missing",
        }), 404
    frame = find_artifact_frame(artifact, int(data["frame_idx"]))
    if frame is None:
        return jsonify({
            "status": "error",
            "reason": "frame_missing",
        }), 400
    track_id = data.get("track_id")
    if track_id not in (None, ""):
        valid_track_ids = {
            str(detection.get("track_id"))
            for detection in frame.get("detections", [])
            if detection.get("track_id") is not None
        }
        if str(track_id) not in valid_track_ids:
            return jsonify({
                "status": "error",
                "reason": "track_id_missing",
            }), 400

    with open(PERCEPTION_FEEDBACK_FILE, 'a') as f:
        f.write(json.dumps(data) + "\n")
    return jsonify({"status": "success"})


@app.route('/api/video/<path:filename>')
def serve_video(filename):
    return send_from_directory(CLIPS_DIR, filename)


@app.route('/api/save', methods=['POST'])
def save_annotation():
    data = request.json
    with open(GT_FILE, 'a') as f:
        f.write(json.dumps(data) + "\n")
    return jsonify({"status": "success"})


def track_landmarks(video_path, start_frame_idx, initial_points):
    """
    Tracks landmarks forward and backward using Lucas-Kanade optical flow.
    Returns: { frame_idx: [ [x, y], ... ] }
    """
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    lk_params = dict(winSize=(21, 21), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS |
                               cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    track_results = {start_frame_idx: initial_points}

    def track_direction(range_frames, points):
        curr_pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
        ret, prev_frame = cap.read()
        if not ret:
            return
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        for f_idx in range_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            n_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray,
                                                          curr_pts, None,
                                                          **lk_params)
            track_results[f_idx] = n_pts.reshape(-1, 2).tolist()
            curr_pts = n_pts
            prev_gray = gray

    track_direction(range(start_frame_idx + 1, total_frames), initial_points)
    track_direction(range(start_frame_idx - 1, -1, -1), initial_points)
    cap.release()
    return track_results


def sort_points_for_primitive(sample_points, primitive_kind):
    if primitive_kind == "point":
        return sample_points
    if primitive_kind == "line":
        xs = [point["image_xy"][0] for point in sample_points]
        ys = [point["image_xy"][1] for point in sample_points]
        sort_idx = 0 if (max(xs) - min(xs)) >= (max(ys) - min(ys)) else 1
        return sorted(sample_points, key=lambda point: point["image_xy"][sort_idx])
    return sorted(sample_points, key=lambda point: point.get("sample_order", 0))


def build_frame_correspondences(tracked_samples, frame_idx):
    grouped = {}
    for tracked_sample in tracked_samples.values():
        frames = tracked_sample["frames"]
        if frame_idx not in frames:
            continue
        primitive_id = tracked_sample["primitive_id"]
        grouped.setdefault(primitive_id, []).append({
            "image_xy": frames[frame_idx],
            "sample_order": tracked_sample.get("sample_order", 0),
        })

    image_points = []
    world_points = []
    families = set()

    for primitive_id, sample_points in grouped.items():
        spec = get_primitive_spec(primitive_id)
        if spec is None:
            continue
        ordered_points = sort_points_for_primitive(sample_points, spec["kind"])
        if spec["kind"] == "point":
            if not ordered_points:
                continue
            image_points.append(ordered_points[0]["image_xy"])
            world_points.append(get_world_point_for_sample(primitive_id, 0.5))
            families.add(spec["family"])
            continue
        if len(ordered_points) < 2:
            continue
        denom = max(len(ordered_points) - 1, 1)
        for index, sample_point in enumerate(ordered_points):
            alpha = index / denom
            image_points.append(sample_point["image_xy"])
            world_points.append(get_world_point_for_sample(primitive_id, alpha))
            families.add(spec["family"])

    return image_points, world_points, families


@app.route('/api/calibrate', methods=['POST'])
def solve_panning_calibration():
    """
    Temporal Aggregation Calibration:
    1. Collects points from different timestamps.
    2. Uses Optical Flow to project all points to frame_idx=0.
    3. Solves for a global H based on the aggregated points.
    4. Projects the global H back to all frames using tracked motion.
    """
    data = request.json
    clip_id = data["id"]
    clip_path = CLIPS_DIR / data["path"]
    points_data = data["points"]
    validation = validate_calibration_points(points_data)
    if not validation["ok"]:
        return jsonify({
            "status": "error",
            **validation,
        }), 400
    
    cap = cv2.VideoCapture(str(clip_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Group points by frame_idx
    frame_to_pts = {}
    for p in points_data:
        f_idx = int((p["t_ms"] / 1000.0) * fps)
        if f_idx not in frame_to_pts:
            frame_to_pts[f_idx] = []
        frame_to_pts[f_idx].append(p)

    # We need to project all points to a common reference (e.g., Frame 0)
    # to solve for a "canonical" Homography that we then adjust per frame.
    # For now, let's track all points across the whole clip.
    
    all_tracked = {} # sample_key -> { frame_idx: (x, y), primitive_id: str, sample_order: int }
    for f_idx, pts in frame_to_pts.items():
        initial_pts = [[p["x"], p["y"]] for p in pts]
        tracked = track_landmarks(clip_path, f_idx, initial_pts)
        for i, p in enumerate(pts):
            primitive_id = get_point_identifier(p)
            sample_key = p.get("point_key") or f'{primitive_id}@{f_idx}:{i}'
            if sample_key not in all_tracked:
                all_tracked[sample_key] = {
                    "primitive_id": primitive_id,
                    "sample_order": int(p.get("sample_order", i)),
                    "frames": {},
                }
            for res_f_idx, res_pts in tracked.items():
                all_tracked[sample_key]["frames"][res_f_idx] = res_pts[i]

    # Solve H for every frame where we have at least 4 landmarks tracked
    h_matrices = {}
    for f_idx in range(total_frames):
        img_pts, world_pts, frame_families = build_frame_correspondences(all_tracked, f_idx)
        if len(img_pts) >= 4 and len(frame_families) >= 2:
            H, _ = cv2.findHomography(np.array(img_pts, dtype=np.float32),
                                      np.array(world_pts, dtype=np.float32))
            if H is not None:
                h_matrices[str(f_idx)] = H.tolist()

    calibrations = {}
    if CALIBRATION_FILE.exists():
        with open(CALIBRATION_FILE, 'r') as f:
            calibrations = json.load(f)
    
    point_summary = summarize_calibration_points(points_data)
    calibrations[clip_id] = {
        "type": "temporal_aggregation_partial_court",
        "solver": "primitive_sample_correspondence_v1",
        "h_sequence": h_matrices,
        "primitive_count": len({sample["primitive_id"] for sample in all_tracked.values()}),
        "point_sample_count": len(all_tracked),
        "primitive_ids": sorted({sample["primitive_id"] for sample in all_tracked.values()}),
        "primitive_families": sorted(point_summary["family_counts"]),
        "point_summary": point_summary,
    }
    
    with open(CALIBRATION_FILE, 'w') as f:
        json.dump(calibrations, f, indent=2)

    return jsonify({
        "status": "success",
        "mode": "temporal_aggregation_partial_court",
        "primitives": sorted({sample["primitive_id"] for sample in all_tracked.values()}),
        "primitive_families": sorted(point_summary["family_counts"]),
        "frames_calibrated": len(h_matrices),
        "point_summary": point_summary,
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

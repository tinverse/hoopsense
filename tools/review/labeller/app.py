import json
import cv2
import numpy as np
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory

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
    "general_note",
}


def _landmark_spec(label, family, kind, xyz):
    return {
        "label": label,
        "family": family,
        "kind": kind,
        "court_xyz": [float(xyz[0]), float(xyz[1]), float(xyz[2])],
    }

# Ensure directories exist
CLIPS_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_DIR.mkdir(parents=True, exist_ok=True)
PERCEPTION_DIR.mkdir(parents=True, exist_ok=True)

# Landmark data (NCAA cm). These are intentionally easier-to-click partial-court
# correspondences so calibration does not depend on visible corners or center line.
LANDMARKS = {
    "rim_l": _landmark_spec("Left Rim", "rim", "point", [160, 762, 305]),
    "rim_r": _landmark_spec("Right Rim", "rim", "point", [2705, 762, 305]),
    "sideline_top_left_third": _landmark_spec("Far Sideline Left Third", "sideline", "line_point", [720, 0, 0]),
    "sideline_top_mid": _landmark_spec("Far Sideline Midcourt", "sideline", "line_point", [1432, 0, 0]),
    "sideline_top_right_third": _landmark_spec("Far Sideline Right Third", "sideline", "line_point", [2145, 0, 0]),
    "sideline_bottom_left_third": _landmark_spec("Near Sideline Left Third", "sideline", "line_point", [720, 1524, 0]),
    "sideline_bottom_mid": _landmark_spec("Near Sideline Midcourt", "sideline", "line_point", [1432, 1524, 0]),
    "sideline_bottom_right_third": _landmark_spec("Near Sideline Right Third", "sideline", "line_point", [2145, 1524, 0]),
    "baseline_left_top": _landmark_spec("Left Baseline Upper", "baseline", "line_point", [0, 280, 0]),
    "baseline_left_mid": _landmark_spec("Left Baseline Mid", "baseline", "line_point", [0, 762, 0]),
    "baseline_left_bottom": _landmark_spec("Left Baseline Lower", "baseline", "line_point", [0, 1244, 0]),
    "baseline_right_top": _landmark_spec("Right Baseline Upper", "baseline", "line_point", [2865, 280, 0]),
    "baseline_right_mid": _landmark_spec("Right Baseline Mid", "baseline", "line_point", [2865, 762, 0]),
    "baseline_right_bottom": _landmark_spec("Right Baseline Lower", "baseline", "line_point", [2865, 1244, 0]),
    "lane_left_top": _landmark_spec("Left Lane Upper Corner", "lane", "paint_corner", [580, 517, 0]),
    "lane_left_bottom": _landmark_spec("Left Lane Lower Corner", "lane", "paint_corner", [580, 1007, 0]),
    "lane_right_top": _landmark_spec("Right Lane Upper Corner", "lane", "paint_corner", [2285, 517, 0]),
    "lane_right_bottom": _landmark_spec("Right Lane Lower Corner", "lane", "paint_corner", [2285, 1007, 0]),
    "arc_left_upper": _landmark_spec("Left 3PT Arc Upper", "three_point_arc", "arc_point", [735.0, 409.0, 0]),
    "arc_left_center": _landmark_spec("Left 3PT Arc Center", "three_point_arc", "arc_point", [835.0, 762.0, 0]),
    "arc_left_lower": _landmark_spec("Left 3PT Arc Lower", "three_point_arc", "arc_point", [735.0, 1115.0, 0]),
    "arc_right_upper": _landmark_spec("Right 3PT Arc Upper", "three_point_arc", "arc_point", [2130.0, 409.0, 0]),
    "arc_right_center": _landmark_spec("Right 3PT Arc Center", "three_point_arc", "arc_point", [2030.0, 762.0, 0]),
    "arc_right_lower": _landmark_spec("Right 3PT Arc Lower", "three_point_arc", "arc_point", [2130.0, 1115.0, 0]),
    "corner_tl": _landmark_spec("Top-Left Corner", "corner", "point", [0, 0, 0]),
    "corner_bl": _landmark_spec("Bottom-Left Corner", "corner", "point", [0, 1524, 0]),
    "corner_tr": _landmark_spec("Top-Right Corner", "corner", "point", [2865, 0, 0]),
    "corner_br": _landmark_spec("Bottom-Right Corner", "corner", "point", [2865, 1524, 0]),
}


def get_landmark_world_xy(landmark_id):
    spec = LANDMARKS.get(landmark_id)
    if spec is None:
        return None
    return spec["court_xyz"][:2]


def summarize_calibration_points(points_data):
    families = {}
    unique_landmark_ids = set()
    for point in points_data:
        landmark_id = point.get("landmark_id")
        spec = LANDMARKS.get(landmark_id)
        if spec is None:
            continue
        unique_landmark_ids.add(landmark_id)
        family = spec["family"]
        families[family] = families.get(family, 0) + 1
    return {
        "point_count": len(points_data),
        "unique_landmark_count": len(unique_landmark_ids),
        "family_counts": families,
    }


def validate_calibration_points(points_data):
    unknown = []
    unique_landmark_ids = set()
    families = set()
    for point in points_data:
        landmark_id = point.get("landmark_id")
        spec = LANDMARKS.get(landmark_id)
        if spec is None:
            unknown.append(landmark_id)
            continue
        unique_landmark_ids.add(landmark_id)
        families.add(spec["family"])

    if unknown:
        return {"ok": False, "reason": "unknown_landmarks", "unknown_landmarks": sorted(set(unknown))}
    if len(points_data) < 4:
        return {"ok": False, "reason": "too_few_points", "required_points": 4}
    if len(unique_landmark_ids) < 4:
        return {"ok": False, "reason": "too_few_unique_landmarks", "required_landmarks": 4}
    if len(families) < 2:
        return {"ok": False, "reason": "too_few_landmark_families", "required_families": 2}
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
    return jsonify(LANDMARKS)


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
    
    all_tracked = {} # sample_key -> { frame_idx: (x, y), landmark_id: str }
    for f_idx, pts in frame_to_pts.items():
        initial_pts = [[p["x"], p["y"]] for p in pts]
        tracked = track_landmarks(clip_path, f_idx, initial_pts)
        for i, p in enumerate(pts):
            sample_key = p.get("point_key") or f'{p["landmark_id"]}@{f_idx}:{i}'
            if sample_key not in all_tracked:
                all_tracked[sample_key] = {"landmark_id": p["landmark_id"], "frames": {}}
            for res_f_idx, res_pts in tracked.items():
                all_tracked[sample_key]["frames"][res_f_idx] = res_pts[i]

    # Solve H for every frame where we have at least 4 landmarks tracked
    h_matrices = {}
    for f_idx in range(total_frames):
        img_pts = []
        world_pts = []
        frame_families = set()
        for tracked_sample in all_tracked.values():
            landmark_id = tracked_sample["landmark_id"]
            frames = tracked_sample["frames"]
            if f_idx in frames:
                img_pts.append(frames[f_idx])
                world_pts.append(get_landmark_world_xy(landmark_id))
                frame_families.add(LANDMARKS[landmark_id]["family"])

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
        "solver": "point_correspondence_partial_court_v1",
        "h_sequence": h_matrices,
        "landmark_count": len({sample["landmark_id"] for sample in all_tracked.values()}),
        "point_sample_count": len(all_tracked),
        "landmark_ids": sorted({sample["landmark_id"] for sample in all_tracked.values()}),
        "landmark_families": sorted(point_summary["family_counts"]),
        "point_summary": point_summary,
    }
    
    with open(CALIBRATION_FILE, 'w') as f:
        json.dump(calibrations, f, indent=2)

    return jsonify({
        "status": "success",
        "mode": "temporal_aggregation_partial_court",
        "landmarks": sorted({sample["landmark_id"] for sample in all_tracked.values()}),
        "landmark_families": sorted(point_summary["family_counts"]),
        "frames_calibrated": len(h_matrices),
        "point_summary": point_summary,
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

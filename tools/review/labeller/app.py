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
}

# Ensure directories exist
CLIPS_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_DIR.mkdir(parents=True, exist_ok=True)
PERCEPTION_DIR.mkdir(parents=True, exist_ok=True)

# Landmark data (NCAA cm)
LANDMARKS = {
    "rim_l": [160, 762, 305],
    "rim_r": [2705, 762, 305],
    "ft_l_top": [580, 517, 0],
    "ft_l_bot": [580, 1007, 0],
    "mid_top": [1432, 0, 0],
    "mid_bot": [1432, 1524, 0],
    "corner_tl": [0, 0, 0],
    "corner_bl": [0, 1524, 0],
    "corner_tr": [2865, 0, 0],
    "corner_br": [2865, 1524, 0]
}

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
    
    all_tracked = {} # landmark_id -> { frame_idx: (x, y) }
    for f_idx, pts in frame_to_pts.items():
        initial_pts = [[p["x"], p["y"]] for p in pts]
        tracked = track_landmarks(clip_path, f_idx, initial_pts)
        for i, p in enumerate(pts):
            lid = p["landmark_id"]
            if lid not in all_tracked:
                all_tracked[lid] = {}
            for res_f_idx, res_pts in tracked.items():
                all_tracked[lid][res_f_idx] = res_pts[i]

    # Solve H for every frame where we have at least 4 landmarks tracked
    h_matrices = {}
    for f_idx in range(total_frames):
        img_pts = []
        world_pts = []
        for lid, frames in all_tracked.items():
            if f_idx in frames:
                img_pts.append(frames[f_idx])
                world_pts.append(LANDMARKS[lid][:2])
        
        if len(img_pts) >= 4:
            H, _ = cv2.findHomography(np.array(img_pts, dtype=np.float32),
                                      np.array(world_pts, dtype=np.float32))
            if H is not None:
                h_matrices[str(f_idx)] = H.tolist()

    calibrations = {}
    if CALIBRATION_FILE.exists():
        with open(CALIBRATION_FILE, 'r') as f:
            calibrations = json.load(f)
    
    calibrations[clip_id] = {
        "type": "temporal_aggregation",
        "h_sequence": h_matrices,
        "landmark_count": len(all_tracked)
    }
    
    with open(CALIBRATION_FILE, 'w') as f:
        json.dump(calibrations, f, indent=2)

    return jsonify({
        "status": "success",
        "mode": "temporal_aggregation",
        "landmarks": list(all_tracked.keys()),
        "frames_calibrated": len(h_matrices)
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

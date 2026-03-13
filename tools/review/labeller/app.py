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
GT_FILE = TRAINING_DIR / "manual_gt.jsonl"
CALIBRATION_FILE = TRAINING_DIR / "camera_calibration.json"

# Ensure directories exist
CLIPS_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_DIR.mkdir(parents=True, exist_ok=True)

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
    """Tracks points across clip and solves H per frame."""
    data = request.json
    clip_id = data["id"]
    clip_path = CLIPS_DIR / data["path"]
    start_time_ms = data["t_ms"]
    points_data = data["points"]
    cap = cv2.VideoCapture(str(clip_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_frame_idx = int((start_time_ms / 1000.0) * fps)
    cap.release()
    initial_img_pts = [[p["x"], p["y"]] for p in points_data]
    world_pts = np.array([LANDMARKS[p["landmark_id"]][:2]
                          for p in points_data], dtype=np.float32)
    print(f"[INFO] Tracking {clip_id} from frame {start_frame_idx}...")
    tracked_pts = track_landmarks(clip_path, start_frame_idx, initial_img_pts)
    h_matrices = {}
    for f_idx, img_pts in tracked_pts.items():
        H, _ = cv2.findHomography(np.array(img_pts, dtype=np.float32),
                                  world_pts)
        if H is not None:
            h_matrices[str(f_idx)] = H.tolist()
    calibrations = {}
    if CALIBRATION_FILE.exists():
        with open(CALIBRATION_FILE, 'r') as f:
            calibrations = json.load(f)
    calibrations[clip_id] = {"type": "panning", "h_sequence": h_matrices}
    with open(CALIBRATION_FILE, 'w') as f:
        json.dump(calibrations, f, indent=2)
    return jsonify({"status": "success", "mode": "panning",
                    "frames_tracked": len(h_matrices)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

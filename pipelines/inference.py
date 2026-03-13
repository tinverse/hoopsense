import json
import numpy as np
import os
import yaml
from collections import deque
from pipelines.behavior_engine import BehaviorStateMachine, PossessionEngine
from pipelines.geometry import lift_keypoints_to_3d
from pipelines.geometry import project_pixel_to_court

# Constants for Feature Schema V2
LABEL_MAP_INV = {
    0: "jump_shot", 1: "crossover", 2: "rebound", 3: "block", 4: "steal"
}


def get_label_map(spec_path="specs/basketball_ncaa.yaml"):
    if not os.path.exists(spec_path):
        return {
            0: "jump_shot", 1: "crossover", 2: "rebound", 3: "block", 4: "steal"
        }
    with open(spec_path, 'r') as f:
        spec = yaml.safe_load(f)
    labels = []
    for cat in spec.get('categories', []):
        labels.extend(cat.get('rules', []))
    labels = sorted(list(set(labels)))
    return {i: label for i, label in enumerate(labels)}


LABEL_MAP_INV = get_label_map()


class KalmanFilter:
    def __init__(self, process_variance=1e-4, measurement_variance=1e-2):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.posteri_estimate = 0.0
        self.posteri_error_estimate = 1.0
        self.initialized = False

    def update(self, measurement):
        if not self.initialized:
            self.posteri_estimate = measurement
            self.initialized = True
            return self.posteri_estimate
        pri_est = self.posteri_estimate
        pri_err = self.posteri_error_estimate + self.process_variance
        blending = pri_err / (pri_err + self.measurement_variance)
        self.posteri_estimate = pri_est + blending * (measurement - pri_est)
        self.posteri_error_estimate = (1 - blending) * pri_err
        return self.posteri_estimate


class TrackManager:
    def __init__(self, tid):
        self.tid = tid
        self.kf_x = KalmanFilter()
        self.kf_y = KalmanFilter()
        self.kpt_history = deque(maxlen=30)
        self.state_machine = BehaviorStateMachine()
        self.court_x = 0.0
        self.court_y = 0.0
        self.pos_3d = np.zeros(3)
        self.team = 1  # Default to Home for now

    def update_position(self, x, y):
        self.court_x = self.kf_x.update(x)
        self.court_y = self.kf_y.update(y)
        return self.court_x, self.court_y

    def add_keypoints(self, kpts, H):
        if kpts is not None:
            self.kpt_history.append(kpts)
            kpts_np = np.array(kpts)
            if np.any(kpts_np[11:13] > 0):
                lifted = lift_keypoints_to_3d(kpts_np, H)
                self.pos_3d = lifted[11:13].mean(axis=0)

    def is_ready(self):
        return len(self.kpt_history) == 30


def construct_features_v2(kpt_history, last_ball_2d, H, player_court_pos,
                          kpts_3d_approx, ball_3d_approx, device):
    import torch
    if len(kpt_history) < 30:
        return None
    kpts_2d = np.array(kpt_history)
    features = []
    ball_court_xy = project_pixel_to_court(last_ball_2d[0], last_ball_2d[1], H)
    ball_z = ball_3d_approx[2] if ball_3d_approx is not None else 100.0
    ball_3d = np.array([ball_court_xy[0], ball_court_xy[1], ball_z])
    for t in range(30):
        pose = kpts_2d[t].flatten()
        delta = kpts_2d[t] - kpts_2d[max(0, t-1)]
        vel = delta.flatten() * 0.1
        kpts_3d = lift_keypoints_to_3d(kpts_2d[t], H)
        dist_l = np.linalg.norm(kpts_3d[9] - ball_3d) * 0.01
        dist_r = np.linalg.norm(kpts_3d[10] - ball_3d) * 0.01
        court_context = player_court_pos * 0.001
        row = np.concatenate([pose, vel, [dist_l, dist_r], court_context])
        features.append(row)
    return torch.FloatTensor(np.array(features)).unsqueeze(0).to(device)


def extract_game_dna(video_path=None, output_dir="data", smoke_test=False):
    import cv2
    import torch
    from ultralytics import YOLO
    from core.vision.action_brain import ActionBrain
    if video_path is None:
        with open("hoops_config.yaml", "r") as f:
            config = yaml.safe_load(f)
        video_path = config.get("local_video_path", "data/sample.mp4")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pose_model = YOLO('yolov8n-pose.pt')
    if smoke_test:
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        pose_model(dummy)
        return
    H = np.eye(3)
    if os.path.exists("data/calibration.json"):
        with open("data/calibration.json", 'r') as f:
            H = np.array(json.load(f)["h_matrix"])
    brain = None
    brain_path = "data/models/action_brain.pt"
    if os.path.exists(brain_path):
        brain = ActionBrain(num_classes=len(LABEL_MAP_INV)).to(device)
        brain.load_state_dict(torch.load(brain_path, map_location=device))
        brain.eval()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "intelligent_game_dna.jsonl")
    frame_idx, tracks = 0, {}
    possession_engine = PossessionEngine()
    last_ball_2d = np.array([0.0, 0.0])
    with open(out_file, 'w') as f:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            frame_idx += 1
            t_ms = int(frame_idx/fps*1000)
            res = pose_model.track(frame, persist=True, classes=[0, 32],
                                   verbose=False)
            if res[0].boxes.id is not None:
                ball_3d = None
                boxes_xywh = res[0].boxes.xywh
                cls = res[0].boxes.cls
                conf = res[0].boxes.conf
                for b, c, cn in zip(boxes_xywh, cls, conf):
                    if int(c) == 32:
                        last_ball_2d = np.array([float(b[0]), float(b[1])])
                        b_xy = project_pixel_to_court(last_ball_2d[0],
                                                      last_ball_2d[1], H)
                        ball_3d = np.array([b_xy[0], b_xy[1], 50.0])
                        f.write(json.dumps({
                            "kind": "ball", "t_ms": t_ms,
                            "x": float(b[0]), "y": float(b[1]),
                            "confidence_bps": int(cn*10000)}) + "\n")
                boxes = res[0].boxes.xywh.cpu().numpy()
                tids = res[0].boxes.id.int().cpu().numpy()
                classes = res[0].boxes.cls.int().cpu().numpy()
                keypoints = res[0].keypoints.xyn.cpu().numpy() \
                    if res[0].keypoints is not None else None
                player_map = {}
                for i, tid in enumerate(tids):
                    if classes[i] == 0:
                        if tid not in tracks:
                            tracks[tid] = TrackManager(tid)
                        tm = tracks[tid]
                        r_cp = project_pixel_to_court(float(boxes[i][0]),
                                                      float(boxes[i][1]), H)
                        tm.update_position(r_cp[0], r_cp[1])
                        kpts = keypoints[i].tolist() \
                            if keypoints is not None else None
                        tm.add_keypoints(kpts, H)
                        player_map[tid] = {'pos_3d': tm.pos_3d,
                                           'team': tm.team}
                pos_events = possession_engine.update(player_map, ball_3d, t_ms)
                for ev in pos_events:
                    f.write(json.dumps(ev) + "\n")
                for tid in player_map.keys():
                    tm = tracks[tid]
                    learned_label = None
                    if tm.is_ready() and brain:
                        player_cp = np.array([tm.court_x, tm.court_y])
                        feat = construct_features_v2(tm.kpt_history,
                                                     last_ball_2d, H,
                                                     player_cp, None,
                                                     None, device)
                        with torch.no_grad():
                            out = brain(feat)
                            learned_label = LABEL_MAP_INV[int(torch.argmax(out))]
                    is_h = possession_engine.current_handler == tid
                    context = {"has_possession": is_h}
                    tm.state_machine.update(tm.kpt_history,
                                            learned_label=learned_label,
                                            context=context)
                    f.write(json.dumps({
                        "kind": "player", "track_id": int(tid), "t_ms": t_ms,
                        "action": tm.state_machine.get_label(),
                        "court_x": tm.court_x, "court_y": tm.court_y}) + "\n")
    cap.release()
    print(f"[INFO] Complete. Output: {out_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="HoopSense Inference Pipeline")
    parser.add_argument("video_path", nargs='?', default=None,
                        help="Path to input video file")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run a quick model initialization check")
    parser.add_argument("--output-dir", default="data",
                        help="Output directory for results")
    args = parser.parse_args()
    extract_game_dna(video_path=args.video_path, output_dir=args.output_dir,
                     smoke_test=args.smoke_test)

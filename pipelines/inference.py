import cv2
import json
import numpy as np
import re
import os
import yaml
from collections import Counter, deque
from pipelines.behavior_engine import BehaviorStateMachine
from pipelines.geometry import lift_keypoints_to_3d
from pipelines.geometry import project_pixel_to_court

# Constants for Feature Schema V2
LABEL_MAP_INV = {0: "jump_shot", 1: "crossover", 2: "rebound", 3: "block", 4: "steal"}

def get_label_map(spec_path="specs/basketball_ncaa.yaml"):
    """Dynamically builds the label map from the DSL categories."""
    if not os.path.exists(spec_path):
        return {0: "jump_shot", 1: "crossover", 2: "rebound", 3: "block", 4: "steal"}
    with open(spec_path, 'r') as f:
        spec = yaml.safe_load(f)
    labels = []
    for cat in spec.get('categories', []):
        labels.extend(cat.get('rules', []))
    labels = sorted(list(set(labels)))
    return {i: label for i, label in enumerate(labels)}

LABEL_MAP_INV = get_label_map()

def construct_features_v2(kpt_history, last_ball_2d, H, player_court_pos, kpts_3d_approx, ball_3d_approx, device):
    """
    Constructs the D=72 multimodal feature tensor for the ActionBrain.
    Schema: [17*2(pose), 17*2(vel), 2(ball_dist), 2(court_pos)]
    """
    import torch
    if len(kpt_history) < 30: return None
    kpts_2d = np.array(kpt_history)
    features = []
    
    # 1. Project ball to court. 
    # If ball_3d_approx is available (from a trajectory filter), use it. 
    # Otherwise, assume it's near the floor (100cm as a proxy for 'held' or 'dribbled').
    ball_court_xy = project_pixel_to_court(last_ball_2d[0], last_ball_2d[1], H)
    ball_z = ball_3d_approx[2] if ball_3d_approx is not None else 100.0
    ball_3d = np.array([ball_court_xy[0], ball_court_xy[1], ball_z])
    
    for t in range(30):
        # Normalized 2D Pose
        pose = kpts_2d[t].flatten()
        # 2D Velocity
        vel = (kpts_2d[t] - kpts_2d[max(0, t-1)]).flatten() * 0.1
        
        # 3D Context for distance metrics
        kpts_3d = lift_keypoints_to_3d(kpts_2d[t], H)
        # Distance from wrists to ball (normalized to meters for the model)
        dist_l = np.linalg.norm(kpts_3d[9] - ball_3d) * 0.01
        dist_r = np.linalg.norm(kpts_3d[10] - ball_3d) * 0.01
        
        # Global court context (normalized to meters)
        court_context = player_court_pos * 0.001
        
        row = np.concatenate([pose, vel, [dist_l, dist_r], court_context])
        features.append(row)
        
    return torch.FloatTensor(np.array(features)).unsqueeze(0).to(device)

def match_pose_to_box(box, poses, frame_w, frame_h):
    if not poses: return None
    cx, cy, _, _ = box
    box_center = np.array([cx / frame_w, cy / frame_h])
    best_idx = -1
    min_dist = float('inf')
    for i, pose in enumerate(poses):
        valid_kpts = pose[np.all(pose > 0, axis=1)]
        if len(valid_kpts) == 0: continue
        pose_center = np.mean(valid_kpts, axis=0)
        dist = np.linalg.norm(box_center - pose_center)
        if dist < min_dist:
            min_dist = dist
            best_idx = i
    return poses[best_idx].tolist() if min_dist < 0.1 else None

from pipelines.audio_head import AudioHead

def extract_game_dna(video_path=None, output_dir="data"):
    import torch
    from ultralytics import YOLO
    import easyocr
    from core.vision.action_brain import ActionBrain
    
    # Load config if no path provided
    if video_path is None:
        with open("hoops_config.yaml", "r") as f:
            config = yaml.safe_load(f)
        video_path = config.get("local_video_path", "data/sample.mp4")

    # 1. Initialize Audio Head (Concurrent Stream)
    a_head = AudioHead()
    audio_file = a_head.extract_audio(video_path)
    audio_cues = a_head.spot_keywords(audio_file)
    print(f"[INFO] Audio Pre-processing complete. Found {len(audio_cues)} cues.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Loading Perception Models on {device}...")
    det_model = YOLO('yolov8n.pt')
    pose_model = YOLO('yolov8n-pose.pt')
    reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    
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
    fps, w, h = cap.get(cv2.CAP_PROP_FPS) or 30.0, cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "intelligent_game_dna.jsonl")
    checkpoint_path = os.path.join(output_dir, "checkpoint.json")

    frame_idx, resolved_ids = 0, {}
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
            frame_idx = checkpoint.get("last_frame", 0)
            resolved_ids = {int(k): v for k, v in checkpoint.get("identities", {}).items()}
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    kpt_history, state_machines = {}, {}
    last_ball_2d = np.array([0.0, 0.0])
    
    mode = 'a' if frame_idx > 0 else 'w'
    with open(out_file, mode) as f:
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            frame_idx += 1
            t_ms = int(frame_idx/fps*1000)
            det_results = det_model.track(frame, persist=True, classes=[0, 32], verbose=False)
            pose_results = pose_model(frame, verbose=False)
            
            if det_results[0].boxes.id is not None:
                for b, c, conf in zip(det_results[0].boxes.xywh, det_results[0].boxes.cls, det_results[0].boxes.conf):
                    if int(c) == 32:
                        last_ball_2d = np.array([float(b[0]), float(b[1])])
                        f.write(json.dumps({"kind": "ball", "t_ms": t_ms, "x": float(b[0]), "y": float(b[1]), "confidence_bps": int(conf*10000)}) + "\n")

                poses = pose_results[0].keypoints.xyn.cpu().numpy() if pose_results[0].keypoints is not None else []
                for i, (box, tid) in enumerate(zip(det_results[0].boxes.xywh, det_results[0].boxes.id)):
                    if int(det_results[0].boxes.cls[i]) == 0:
                        tid = int(tid)
                        player_kpts = match_pose_to_box(box, poses, w, h)
                        if tid not in kpt_history: kpt_history[tid] = deque(maxlen=30)
                        if player_kpts: kpt_history[tid].append(player_kpts)
                        
                        # Real Court Projection
                        player_court_pos = project_pixel_to_court(float(box[0]), float(box[1]), H)
                        
                        # Active Player Gating
                        # NCAA: 2865 x 1524 cm. We allow a 2m buffer for active sideline play.
                        is_active = (-200 <= player_court_pos[0] <= 3065) and (-200 <= player_court_pos[1] <= 1724)

                        learned_label = None
                        if is_active and brain and len(kpt_history[tid]) == 30:
                            kpts_3d_mock = [np.zeros((17, 3)) for _ in range(30)]
                            ball_3d_mock = [np.zeros(3) for _ in range(30)]
                            feat_tensor = construct_features_v2(kpt_history[tid], last_ball_2d, H, player_court_pos, kpts_3d_mock, ball_3d_mock, device)
                            with torch.no_grad():
                                output = brain(feat_tensor)
                                learned_label = LABEL_MAP_INV[int(torch.argmax(output))]
                        elif not is_active:
                            learned_label = "idle_bystander"

                        if tid not in state_machines: state_machines[tid] = BehaviorStateMachine()
                        state_machines[tid].update(kpt_history[tid], learned_label=learned_label)
                        f.write(json.dumps({"kind": "player", "track_id": tid, "t_ms": t_ms, "action": state_machines[tid].get_label()}) + "\n")


            if frame_idx % 500 == 0:
                with open(checkpoint_path, 'w') as cp_f:
                    json.dump({"last_frame": frame_idx, "identities": {str(k): v for k, v in resolved_ids.items()}}, cp_f)

    cap.release()
    print(f"[INFO] Complete. Output: {out_file}")

if __name__ == "__main__":
    import sys
    extract_game_dna(sys.argv[1] if len(sys.argv) > 1 else "data/sample.mp4")

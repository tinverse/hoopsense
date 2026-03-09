import cv2
import json
import numpy as np
import re
import os
from collections import Counter, deque
from ultralytics import YOLO
import easyocr
import torch
from behavior_engine import BehaviorStateMachine
from core.vision.action_brain import ActionBrain

# Constants for Feature Schema V2
LABEL_MAP_INV = {0: "jump_shot", 1: "crossover", 2: "rebound", 3: "block", 4: "steal"}

def project_pixel_to_court(u, v, H):
    """
    Python implementation of the Rust SpatialResolver logic.
    Projects 2D pixel (u,v) to 3D court (x,y) using Homography H.
    """
    p = np.array([u, v, 1.0])
    p_world = H @ p
    if abs(p_world[2]) < 1e-6: return np.array([0.0, 0.0])
    return p_world[:2] / p_world[2]

def construct_features_v2(kpt_history, ball_court_pos, player_court_pos, kpts_3d_approx, ball_3d_approx, device):
    """
    Real-time construction of 72-dim feature tensor.
    Matches the 'Ground Truth' logic of generate_data.py.
    """
    if len(kpt_history) < 30: return None
    kpts_2d = np.array(kpt_history) # (30, 17, 2)
    features = []
    
    for t in range(30):
        # 1. Local Pose (34)
        pose = kpts_2d[t].flatten()
        
        # 2. Temporal (34)
        if t > 0:
            velocity = (kpts_2d[t] - kpts_2d[t-1]).flatten() * 0.1
        else:
            velocity = np.zeros(34)
            
        # 3. Interaction (2) - Wrist-to-Ball in CM (Scaled)
        # Using 3D approximations for real distance
        dist_l = np.linalg.norm(kpts_3d_approx[t][9] - ball_3d_approx[t]) * 0.01
        dist_r = np.linalg.norm(kpts_3d_approx[t][10] - ball_3d_approx[t]) * 0.01
        
        # 4. Global (2) - Court Position in CM (Scaled)
        court_context = player_court_pos * 0.001
        
        row = np.concatenate([pose, velocity, [dist_l, dist_r], court_context])
        features.append(row)
        
    return torch.FloatTensor(np.array(features)).unsqueeze(0).to(device)

def extract_game_dna(video_path, output_dir="data"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Loading Perception Models on {device}...")
    det_model = YOLO('yolov8n.pt')
    pose_model = YOLO('yolov8n-pose.pt')
    reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    
    # Mock Homography for Prototyping (Identity for now, in prod calibrated via Rust)
    H = np.eye(3) 

    brain = None
    brain_path = "data/models/action_brain.pt"
    if os.path.exists(brain_path):
        brain = ActionBrain(num_classes=5).to(device)
        brain.load_state_dict(torch.load(brain_path, map_location=device))
        brain.eval()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "intelligent_game_dna.jsonl")

    resolved_ids, history, kpt_history = {}, {}, {}
    state_machines = {}
    last_ball_court_pos = np.array([0.0, 0.0])
    frame_idx = 0

    with open(out_file, 'w') as f:
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            frame_idx += 1
            t_ms = int(frame_idx/fps*1000)
            
            det_results = det_model.track(frame, persist=True, classes=[0, 32], verbose=False)
            pose_results = pose_model(frame, verbose=False)
            
            if det_results[0].boxes.id is not None:
                # Update Ball
                for b, c, conf in zip(det_results[0].boxes.xywh, det_results[0].boxes.cls, det_results[0].boxes.conf):
                    if int(c) == 32:
                        last_ball_court_pos = project_pixel_to_court(float(b[0]), float(b[1]), H)
                        f.write(json.dumps({"kind": "ball", "t_ms": t_ms, "x": float(b[0]), "y": float(b[1]), "confidence_bps": int(conf*10000)}) + "\n")

                boxes, track_ids = det_results[0].boxes.xywh, det_results[0].boxes.id
                poses = pose_results[0].keypoints.xyn.cpu().numpy() if pose_results[0].keypoints is not None else []

                for i, (box, tid) in enumerate(zip(boxes, track_ids)):
                    tid = int(tid)
                    if int(det_results[0].boxes.cls[i]) == 0:
                        player_kpts = poses[0].tolist() if len(poses) > 0 else None # Simp match
                        
                        if tid not in kpt_history: kpt_history[tid] = deque(maxlen=30)
                        if player_kpts: kpt_history[tid].append(player_kpts)
                        
                        # Real Court Projection
                        player_court_pos = project_pixel_to_court(float(box[0]), float(box[1]), H)

                        learned_label = None
                        if brain and len(kpt_history[tid]) == 30:
                            # Mock 3D for feature construction until lifting is ported
                            kpts_3d_mock = [np.zeros((17, 3)) for _ in range(30)]
                            ball_3d_mock = [np.zeros(3) for _ in range(30)]
                            feat_tensor = construct_features_v2(kpt_history[tid], last_ball_court_pos, player_court_pos, kpts_3d_mock, ball_3d_mock, device)
                            with torch.no_grad():
                                output = brain(feat_tensor)
                                learned_label = LABEL_MAP_INV[int(torch.argmax(output))]

                        row = {"kind": "player", "track_id": tid, "t_ms": t_ms, "action": learned_label or "unknown"}
                        f.write(json.dumps(row) + "\n")

    cap.release()
    print(f"[INFO] Complete. Output: {out_file}")

if __name__ == "__main__":
    import sys
    extract_game_dna(sys.argv[1] if len(sys.argv) > 1 else "data/sample.mp4")

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
    """Projects 2D pixel (u,v) to 3D court (x,y) using Homography H."""
    p = np.array([u, v, 1.0])
    p_world = H @ p
    if abs(p_world[2]) < 1e-6: return np.array([0.0, 0.0])
    return p_world[:2] / p_world[2]

def lift_to_3d(kpts_2d, H):
    """
    Kinematic Lifting: Estimates 3D world coordinates from 2D pixels.
    Uses the homography to find (X,Y) on floor, then approximates Z.
    """
    kpts_3d = []
    # 1. Map feet to floor to get world scale
    l_ank, r_ank = kpts_2d[15], kpts_2d[16]
    floor_xy = project_pixel_to_court((l_ank[0]+r_ank[0])/2, (l_ank[1]+r_ank[1])/2, H)
    
    for u, v in kpts_2d:
        world_xy = project_pixel_to_court(u, v, H)
        # Approximate Z based on vertical displacement from floor in world space
        # (Simplified height heuristic for single-camera depth)
        z_est = abs(world_xy[1] - floor_xy[1]) * 0.5 
        kpts_3d.append([world_xy[0], world_xy[1], z_est])
    return np.array(kpts_3d)

def construct_features_v2(kpt_history, last_ball_2d, H, player_court_pos, device):
    """
    Real-time construction of 72-dim feature tensor.
    Uses real geometry (Lifting + Projection) to match training semantics.
    """
    if len(kpt_history) < 30: return None
    kpts_2d = np.array(kpt_history) # (30, 17, 2)
    features = []
    
    # Pre-calculate 3D ball approximation
    ball_court_xy = project_pixel_to_court(last_ball_2d[0], last_ball_2d[1], H)
    ball_3d = np.array([ball_court_xy[0], ball_court_xy[1], 100.0]) # Assume dribble height proxy
    
    for t in range(30):
        # 1. Local Pose (34)
        pose = kpts_2d[t].flatten()
        
        # 2. Temporal (34)
        vel = (kpts_2d[t] - kpts_2d[max(0, t-1)]).flatten() * 0.1
        
        # 3. Interaction (2) - Real 3D Wrist-to-Ball distance in CM
        kpts_3d = lift_to_3d(kpts_2d[t], H)
        dist_l = np.linalg.norm(kpts_3d[9] - ball_3d) * 0.01
        dist_r = np.linalg.norm(kpts_3d[10] - ball_3d) * 0.01
        
        # 4. Global (2) - Real Court Position in CM
        court_context = player_court_pos * 0.001
        
        row = np.concatenate([pose, vel, [dist_l, dist_r], court_context])
        features.append(row)
        
    return torch.FloatTensor(np.array(features)).unsqueeze(0).to(device)

def extract_game_dna(video_path, output_dir="data"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Loading Perception Models on {device}...")
    det_model = YOLO('yolov8n.pt')
    pose_model = YOLO('yolov8n-pose.pt')
    reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    
    # Load Real Calibration if exists, else default to NCAA Identity
    H = np.eye(3)
    if os.path.exists("data/calibration.json"):
        with open("data/calibration.json", 'r') as f:
            H = np.array(json.load(f)["h_matrix"])
            print("[INFO] Calibrated Homography Loaded.")

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
    last_ball_2d = np.array([0.0, 0.0])
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
                        last_ball_2d = np.array([float(b[0]), float(b[1])])
                        f.write(json.dumps({"kind": "ball", "t_ms": t_ms, "x": last_ball_2d[0], "y": last_ball_2d[1], "confidence_bps": int(conf*10000)}) + "\n")

                boxes, track_ids = det_results[0].boxes.xywh, det_results[0].boxes.id
                poses = pose_results[0].keypoints.xyn.cpu().numpy() if pose_results[0].keypoints is not None else []

                for i, (box, tid) in enumerate(zip(boxes, track_ids)):
                    tid = int(tid)
                    if int(det_results[0].boxes.cls[i]) == 0:
                        player_kpts = poses[0].tolist() if len(poses) > 0 else None
                        
                        if tid not in kpt_history: kpt_history[tid] = deque(maxlen=30)
                        if player_kpts: kpt_history[tid].append(player_kpts)
                        
                        # Real Court Projection
                        player_court_pos = project_pixel_to_court(float(box[0]), float(box[1]), H)

                        learned_label = None
                        if brain and len(kpt_history[tid]) == 30:
                            feat_tensor = construct_features_v2(kpt_history[tid], last_ball_2d, H, player_court_pos, device)
                            with torch.no_grad():
                                output = brain(feat_tensor)
                                learned_label = LABEL_MAP_INV[int(torch.argmax(output))]

                        ident = resolved_ids.get(tid, {"jersey": None, "team": "unknown", "is_ref": False})
                        if tid not in state_machines:
                            state_machines[tid] = BehaviorStateMachine(is_ref=ident["is_ref"])
                        
                        state_machines[tid].update(kpt_history[tid], learned_label=learned_label)
                        action_label = state_machines[tid].get_label()

                        row = {"kind": "player", "track_id": tid, "t_ms": t_ms, "action": action_label}
                        f.write(json.dumps(row) + "\n")

    cap.release()
    print(f"[INFO] Complete. Output: {out_file}")

if __name__ == "__main__":
    import sys
    extract_game_dna(sys.argv[1] if len(sys.argv) > 1 else "data/sample.mp4")

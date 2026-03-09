import cv2
import json
import numpy as np
import re
import os
import yaml
from collections import Counter, deque
from ultralytics import YOLO
import easyocr
import torch
from pipelines.behavior_engine import BehaviorStateMachine
from core.vision.action_brain import ActionBrain

def get_label_map(spec_path="specs/basketball_ncaa.yaml"):
    """Dynamically builds the label map from the DSL categories."""
    if not os.path.exists(spec_path):
        return {0: "jump_shot", 1: "crossover", 2: "rebound", 3: "block", 4: "steal"}
    with open(spec_path, 'r') as f:
        spec = yaml.safe_load(f)
    
    # Flatten all rule IDs from all categories into a single index map
    labels = []
    for cat in spec.get('categories', []):
        labels.extend(cat.get('rules', []))
    
    # Ensure uniqueness and stability by sorting
    labels = sorted(list(set(labels)))
    return {i: label for i, label in enumerate(labels)}

LABEL_MAP_INV = get_label_map()

def project_pixel_to_court(u, v, H):
    p = np.array([u, v, 1.0])
    p_world = H @ p
    if abs(p_world[2]) < 1e-6: return np.array([0.0, 0.0])
    return p_world[:2] / p_world[2]

def lift_to_3d(kpts_2d, H):
    kpts_3d = []
    l_ank, r_ank = kpts_2d[15], kpts_2d[16]
    floor_xy = project_pixel_to_court((l_ank[0]+r_ank[0])/2, (l_ank[1]+r_ank[1])/2, H)
    for u, v in kpts_2d:
        world_xy = project_pixel_to_court(u, v, H)
        z_est = abs(world_xy[1] - floor_xy[1]) * 0.5 
        kpts_3d.append([world_xy[0], world_xy[1], z_est])
    return np.array(kpts_3d)

def construct_features_v2(kpt_history, last_ball_2d, H, player_court_pos, device):
    if len(kpt_history) < 30: return None
    kpts_2d = np.array(kpt_history)
    features = []
    ball_court_xy = project_pixel_to_court(last_ball_2d[0], last_ball_2d[1], H)
    ball_3d = np.array([ball_court_xy[0], ball_court_xy[1], 100.0])
    for t in range(30):
        pose = kpts_2d[t].flatten()
        vel = (kpts_2d[t] - kpts_2d[max(0, t-1)]).flatten() * 0.1
        kpts_3d = lift_to_3d(kpts_2d[t], H)
        dist_l = np.linalg.norm(kpts_3d[9] - ball_3d) * 0.01
        dist_r = np.linalg.norm(kpts_3d[10] - ball_3d) * 0.01
        court_context = player_court_pos * 0.001
        row = np.concatenate([pose, vel, [dist_l, dist_r], court_context])
        features.append(row)
    return torch.FloatTensor(np.array(features)).unsqueeze(0).to(device)

def match_pose_to_box(box, poses, frame_w, frame_h):
    """Associates a YOLO detection box with the closest skeletal pose center."""
    if not poses: return None
    cx, cy, _, _ = box
    box_center = np.array([cx / frame_w, cy / frame_h])
    
    best_idx = -1
    min_dist = float('inf')
    for i, pose in enumerate(poses):
        # Use hip center (avg of 11, 12) as pose anchor
        valid_kpts = pose[np.all(pose > 0, axis=1)]
        if len(valid_kpts) == 0: continue
        pose_center = np.mean(valid_kpts, axis=0)
        dist = np.linalg.norm(box_center - pose_center)
        if dist < min_dist:
            min_dist = dist
            best_idx = i
            
    # Distance threshold: 10% of frame width
    return poses[best_idx].tolist() if min_dist < 0.1 else None

def extract_game_dna(video_path, output_dir="data"):
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

    kpt_history, state_machines = {}, {}
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
                        
                        player_court_pos = project_pixel_to_court(float(box[0]), float(box[1]), H)
                        learned_label = None
                        if brain and len(kpt_history[tid]) == 30:
                            kpts_3d_mock = [np.zeros((17, 3)) for _ in range(30)]
                            ball_3d_mock = [np.zeros(3) for _ in range(30)]
                            feat_tensor = construct_features_v2(kpt_history[tid], last_ball_2d, H, player_court_pos, device)
                            with torch.no_grad():
                                output = brain(feat_tensor)
                                learned_label = LABEL_MAP_INV[int(torch.argmax(output))]

                        if tid not in state_machines: state_machines[tid] = BehaviorStateMachine()
                        state_machines[tid].update(kpt_history[tid], learned_label=learned_label)
                        f.write(json.dumps({"kind": "player", "track_id": tid, "t_ms": t_ms, "action": state_machines[tid].get_label()}) + "\n")
    cap.release()

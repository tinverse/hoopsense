import cv2
import json
import numpy as np
import re
import os
from collections import Counter, deque
from ultralytics import YOLO
import easyocr
import torch

def get_team_color(crop_img):
    if crop_img is None or crop_img.size == 0: return "unknown"
    h, w = crop_img.shape[:2]
    center = crop_img[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
    hsv = cv2.cvtColor(center if center.size > 0 else crop_img, cv2.COLOR_BGR2HSV)
    return "light" if np.mean(hsv[:,:,2]) > 120 else "dark"

def perform_ocr_voting(crops, reader):
    votes = []
    for crop in crops:
        if crop.shape[0] < 20: continue
        results = reader.readtext(crop)
        for (_, text, prob) in results:
            digits = re.sub(r'[^0-9]', '', text)
            if 1 <= len(digits) <= 2 and prob >= 0.6:
                votes.append(digits.zfill(2))
    return Counter(votes).most_common(1)[0][0] if votes else None

def recognize_action(history_kpts):
    """
    Heuristic-based action recognition using skeletal keypoint history.
    """
    if len(history_kpts) < 15: return "idle"
    kpts = np.array(history_kpts)
    valid_mask = (kpts[:, 11, 0] > 0) & (kpts[:, 12, 0] > 0)
    if np.sum(valid_mask) < 5: return "idle"
    v_kpts = kpts[valid_mask]
    wrists_above_head = np.mean(v_kpts[:, [9, 10], 1] < v_kpts[:, 0, 1]) > 0.6
    hips_y = np.mean(v_kpts[:, [11, 12], 1], axis=1)
    is_rising = (hips_y[-1] - hips_y[0]) < -0.02
    return "jump_shot" if wrists_above_head and is_rising else "idle"

def recognize_ref_signal(history_kpts):
    """
    Decodes NCAA Referee Hand Signals from skeletal keypoints.
    """
    if len(history_kpts) < 10: return None
    kpts = np.array(history_kpts)
    valid_mask = (kpts[:, 5, 0] > 0) & (kpts[:, 6, 0] > 0)
    if np.sum(valid_mask) < 5: return None
    v_kpts = kpts[valid_mask]
    left_arm_up = np.mean(v_kpts[:, 9, 1] < v_kpts[:, 5, 1]) > 0.7
    right_arm_up = np.mean(v_kpts[:, 10, 1] < v_kpts[:, 6, 1]) > 0.7
    if left_arm_up and right_arm_up: return "ref_3pt_success"
    if left_arm_up or right_arm_up: return "ref_3pt_attempt"
    return None

def is_referee(crop_img):
    """
    Identifies a referee based on visual cues (NCAA stripes or solid neutral).
    """
    if crop_img is None or crop_img.size == 0: return False
    hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
    return np.std(hsv[:,:,2]) > 40

def extract_game_dna(video_path, output_dir="data"):
    print(f"Loading Models on {'GPU' if torch.cuda.is_available() else 'CPU'}...")
    model = YOLO('yolov8n-pose.pt')
    reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "intelligent_game_dna.jsonl")

    resolved_ids = {}
    history = {}
    kpt_history = {}
    is_ref_map = {}
    frame_idx = 0

    with open(out_file, 'w') as f:
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            frame_idx += 1
            results = model.track(frame, persist=True, verbose=False)

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh
                track_ids = results[0].boxes.id
                class_ids = results[0].boxes.cls
                confidences = results[0].boxes.conf
                xyxy_boxes = results[0].boxes.xyxy
                keypoints = results[0].keypoints.xyn.cpu().numpy() if results[0].keypoints is not None else None

                for i, (box, tid, cls, conf, xyxy) in enumerate(zip(boxes, track_ids, class_ids, confidences, xyxy_boxes)):
                    tid = int(tid)
                    if cls == 0: # Person
                        x1, y1, x2, y2 = map(int, xyxy)
                        crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                        if tid not in is_ref_map: is_ref_map[tid] = is_referee(crop)
                        
                        p_kpts = keypoints[i].tolist() if keypoints is not None and i < len(keypoints) else None
                        if tid not in kpt_history: kpt_history[tid] = deque(maxlen=60)
                        if p_kpts: kpt_history[tid].append(p_kpts)
                        
                        if is_ref_map[tid]:
                            signal = recognize_ref_signal(kpt_history[tid])
                            row = {"kind": "referee", "track_id": tid, "t_ms": int(frame_idx/fps*1000), "x": float(box[0]), "y": float(box[1]), "w": float(box[2]), "h": float(box[3]), "signal": signal}
                            f.write(json.dumps(row) + "\n")
                            continue

                        action = recognize_action(kpt_history[tid])
                        if str(tid) not in resolved_ids:
                            if tid not in history: history[tid] = deque(maxlen=30)
                            history[tid].append(frame[max(0, y1):min(frame.shape[0], int(y1+(y2-y1)*0.6)), max(0, x1):min(frame.shape[1], x2)])
                            if len(history[tid]) == 30:
                                resolved_ids[str(tid)] = {"team": get_team_color(list(history[tid])[0]), "jersey_number": perform_ocr_voting(history[tid], reader)}
                                print(f"Locked Track {tid} -> #{resolved_ids[str(tid)]['jersey_number']}")

                        ident = resolved_ids.get(str(tid), {})
                        row = {"kind": "player", "track_id": tid, "t_ms": int(frame_idx/fps*1000), "x": float(box[0]), "y": float(box[1]), "w": float(box[2]), "h": float(box[3]), "confidence_bps": int(conf*10000), "actor_jersey_number": ident.get("jersey_number"), "team_color": ident.get("team"), "action": action, "keypoints": p_kpts}
                        f.write(json.dumps(row) + "\n")
                    elif cls == 32: # Ball
                        row = {"kind": "ball", "track_id": tid, "t_ms": int(frame_idx/fps*1000), "x": float(box[0]), "y": float(box[1]), "w": float(box[2]), "h": float(box[3]), "confidence_bps": int(conf*10000)}
                        f.write(json.dumps(row) + "\n")
    cap.release()
    print(f"Processing Complete. Output: {out_file}")

if __name__ == "__main__":
    import sys
    video = sys.argv[1] if len(sys.argv) > 1 else "data/sample.mp4"
    extract_game_dna(video)

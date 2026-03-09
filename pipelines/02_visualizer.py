import cv2
import json
import os
import sys

def visualize(video_path, jsonl_path, output_path="data/review.mp4"):
    dna = {}
    print("Loading DNA records...")
    with open(jsonl_path, 'r') as f:
        for line in f:
            d = json.loads(line)
            dna.setdefault(d['frame_idx'], []).append(d)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    # Define the skeletal connections (pairs of keypoints indices to draw lines)
    SKELETON_CONNECTIONS = [
        (0, 1), (0, 2), (1, 3), (2, 4), # Face
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # Upper body
        (5, 11), (6, 12), (11, 12), # Torso
        (11, 13), (13, 15), (12, 14), (14, 16) # Lower body
    ]

    idx = 0
    print(f"Rendering: {output_path}")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        idx += 1
        for ent in dna.get(idx, []):
            cx, cy, ew, eh = ent['x'], ent['y'], ent['w'], ent['h']
            x1, y1 = int(cx-ew/2), int(cy-eh/2)
            
            color = (0, 255, 0) # Default Green
            label_prefix = f"ID:{ent['track_id']}"
            
            if ent['kind'] == 'referee':
                color = (255, 0, 255) # Magenta for Ref
                signal = ent.get('signal')
                label = f"{label_prefix} REF [{'SIGNAL:' + signal if signal else 'IDLE'}]"
            elif ent['kind'] == 'player':
                action_label = ent.get('action', 'unknown')
                label = f"{label_prefix} #{ent['actor_jersey_number'] or '??'} [{action_label}]"
            else: # Ball
                color = (0, 165, 255) # Orange for Ball
                label = "BALL"

            cv2.rectangle(frame, (x1, y1), (int(cx+ew/2), int(cy+eh/2)), color, 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw Skeleton
            if 'keypoints' in ent and ent['keypoints'] is not None:
                kpts = ent['keypoints']
                # Draw connections
                for (a, b) in SKELETON_CONNECTIONS:
                    if a < len(kpts) and b < len(kpts):
                        p1 = (int(kpts[a][0] * w), int(kpts[a][1] * h))
                        p2 = (int(kpts[b][0] * w), int(kpts[b][1] * h))
                        # Only draw if both points are valid (not 0,0)
                        if p1 != (0, 0) and p2 != (0, 0):
                            cv2.line(frame, p1, p2, (255, 255, 0), 2)
                # Draw points
                for kpt in kpts:
                    pt = (int(kpt[0] * w), int(kpt[1] * h))
                    if pt != (0, 0):
                        cv2.circle(frame, pt, 3, (0, 0, 255), -1)
        out.write(frame)
    cap.release()
    out.release()
    print("Visualization complete.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 02_visualizer.py <video_path> <jsonl_path>")
    else:
        visualize(sys.argv[1], sys.argv[2])

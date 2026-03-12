import os

# Define the full HoopSense OS project structure with complete source code
project_structure = {
    "hoopsense": {
        "README.md": """# HoopSense OS
Practical game understanding from basketball video.

## Project Structure
- `core/`: Native perception and logic internals.
- `pipelines/`: Deterministic execution stages.
- `docs/`: Design-first documentation (Requirements, Architecture, Book).
- `cli/`: Local orchestration and Gemini Agent tools.
- `data/`: Local storage for training manifests and video links.
- `TASK_STATUS.md`: Current execution frontier.
""",
        ".envrc": """# HoopSense Environment Config
export HOOPS_REMOTE_HOST="root@your-tunnel-address.ngrok.io"
export HOOPS_REMOTE_PORT="22"
export HOOPS_REMOTE_DIR="/content/hoopsense"
export GEMINI_API_KEY=""
""",
        "TASK_STATUS.md": """# Task Status: HoopSense MVP
Frontier: L3.81.4 (Real-world event extraction)

- [x] L1.1 Platform Foundation & CLI
- [x] L1.2 Identity Fusion Prototype
- [ ] L1.3 Native Vision Migration (In Progress)
- [ ] L1.6 Training Data Preparation
""",
        "GEMINI_BOOTSTRAP_PROMPT.md": """# Gemini Bootstrap Prompt for HoopSense
## System Prompt
You are the lead Machine Learning architect for HoopSense.
Mission: Build a basketball analysis system producing trustworthy statistics.
Standards: Design-first SDLC (Requirements -> Arch -> Code -> Tests).

## Bootstrap Task Prompt
You are bootstrapping the project. Current focus:
- detector/tracker internals
- jersey OCR voting
- event inference (the HoopSense JSONL contract)
""",
        "docs/plan/REQUIREMENTS.md": """# Requirements
- R1: Track player/ball in 2D with unique track_id.
- R2: Identify players via Jersey OCR voting and team color clustering.
- R3: Emit HoopSense JSONL contract rows for downstream event inference.
- R4: Support deterministic re-entry and checkpointing for long video processing.
""",
        "docs/architecture/ARCHITECTURE_BLUEPRINT.md": """# Architecture Blueprint
- **Input**: Video (MP4) + Team Rosters (Optional)
- **Perception Layer**: YOLOv8 (Detection) + BoT-SORT (Tracking)
- **Identity Layer**: EasyOCR + HSV Color Clustering + Majority Voting
- **Contract Layer**: HoopSense JSONL Protocol
- **Output**: Event DNA, Game Stats, and Review Visualization
""",
        "docs/book/CHAPTER_01_IDENTITY_FUSION.md": "# Chapter 1: Identity Fusion\nExplaining the gap between Track IDs and Player IDs. We use a buffered voting mechanism to resolve identity.\n",
        "cli/hoops.py": """import os
import sys
import subprocess

REMOTE_HOST = os.getenv("HOOPS_REMOTE_HOST", "root@localhost")
REMOTE_PORT = os.getenv("HOOPS_REMOTE_PORT", "22")
REMOTE_DIR = os.getenv("HOOPS_REMOTE_DIR", "/content/hoopsense")

def sync():
    if "your-tunnel" in REMOTE_HOST:
        print("❌ Update .envrc first.")
        sys.exit(1)
    print(f"🚀 Syncing to {REMOTE_HOST}:{REMOTE_PORT}...")
    subprocess.run([
        "rsync", "-avz",
        "-e", f"ssh -p {REMOTE_PORT} -o StrictHostKeyChecking=no",
        "--exclude", ".git",
        "--exclude", "data/",
        "./", f"{REMOTE_HOST}:{REMOTE_DIR}"
    ])

def run_remote(script_path):
    sync()
    print(f"🏃 Executing {script_path} on GPU...")
    subprocess.run([
        "ssh", "-p", REMOTE_PORT,
        "-o StrictHostKeyChecking=no",
        REMOTE_HOST,
        f"cd {REMOTE_DIR} && python3 {script_path}"
    ])

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        if len(sys.argv) > 2:
            run_remote(sys.argv[2])
        else:
            print("Usage: hoops run <script_path>")
    elif len(sys.argv) > 1 and sys.argv[1] == "sync":
        sync()
    else:
        print("Usage: hoops [sync | run <script>]")
""",
        "pipelines/01_inference.py": """import cv2
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

def extract_game_dna(video_path, output_dir="data"):
    print(f"Loading Models on {'GPU' if torch.cuda.is_available() else 'CPU'}...")
    model = YOLO('yolov8n.pt')
    reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "intelligent_game_dna.jsonl")

    resolved_ids = {}
    history = {}
    frame_idx = 0

    with open(out_file, 'w') as f:
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            frame_idx += 1
            results = model.track(frame, persist=True, verbose=False)

            if results[0].boxes.id is not None:
                for box, tid, cls, conf, xyxy in zip(results[0].boxes.xywh, results[0].boxes.id, results[0].boxes.cls, results[0].boxes.conf, results[0].boxes.xyxy):
                    tid = int(tid)
                    if cls == 0: # Player
                        if str(tid) not in resolved_ids:
                            if tid not in history: history[tid] = deque(maxlen=30)
                            x1, y1, x2, y2 = map(int, xyxy)
                            history[tid].append(frame[y1:int(y1+(y2-y1)*0.6), x1:x2])
                            if len(history[tid]) == 30:
                                resolved_ids[str(tid)] = {"team": get_team_color(list(history[tid])[0]), "jersey_number": perform_ocr_voting(history[tid], reader)}
                                print(f"Locked Track {tid} -> #{resolved_ids[str(tid)]['jersey_number']}")

                        ident = resolved_ids.get(str(tid), {})
                        row = {"kind": "player", "track_id": tid, "t_ms": int(frame_idx/fps*1000), "frame_idx": frame_idx,
                               "x": float(box[0]), "y": float(box[1]), "w": float(box[2]), "h": float(box[3]),
                               "confidence_bps": int(conf*10000), "actor_jersey_number": ident.get("jersey_number"), "team_color": ident.get("team")}
                        f.write(json.dumps(row) + "\\n")
    cap.release()
    print(f"Processing Complete. Output: {out_file}")

if __name__ == "__main__":
    import sys
    video = sys.argv[1] if len(sys.argv) > 1 else "data/sample.mp4"
    extract_game_dna(video)
""",
        "pipelines/02_visualizer.py": """import cv2
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

    idx = 0
    print(f"Rendering: {output_path}")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        idx += 1
        for ent in dna.get(idx, []):
            cx, cy, ew, eh = ent['x'], ent['y'], ent['w'], ent['h']
            x1, y1 = int(cx-ew/2), int(cy-eh/2)
            cv2.rectangle(frame, (x1, y1), (int(cx+ew/2), int(cy+eh/2)), (0, 255, 0), 2)
            label = f"ID:{ent['track_id']} #{ent['actor_jersey_number'] or '??'}"
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        out.write(frame)
    cap.release()
    out.release()
    print("Visualization complete.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 02_visualizer.py <video_path> <jsonl_path>")
    else:
        visualize(sys.argv[1], sys.argv[2])
""",
        "core/vision.py": """# HoopSense Core Vision Internals
# Future home of native NMS, Voting, and Re-ID logic.
""",
        "data/.gitkeep": ""
    }
}

def build_structure(base_path, structure):
    for name, content in structure.items():
        path = os.path.join(base_path, name)
        if isinstance(content, dict):
            os.makedirs(path, exist_ok=True)
            build_structure(path, content)
        else:
            # FIX: Ensure parent directories exist for the file
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(content)
            print(f"Created: {path}")

if __name__ == "__main__":
    # Ensure we use the absolute path for the home directory
    target = os.path.expanduser("~/hoopsense")
    os.makedirs(target, exist_ok=True)
    build_structure(target, project_structure["hoopsense"])
    print(f"\\n✅ HoopSense OS Initialized in {target}")

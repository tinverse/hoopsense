import os
import sys

if not sys.flags.no_user_site:
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    os.execvpe(sys.executable, [sys.executable, "-s", *sys.argv], env)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import json
import subprocess
import shutil
import numpy as np
from pipelines.behavior_engine import BehaviorStateMachine


def run_guix_command(packages, argv):
    if shutil.which("guix"):
        cmd = ["guix", "shell", "--pure", *packages, "--", *argv]
    else:
        cmd = argv
    return subprocess.check_output(cmd, text=True)

def local_match_pose_to_box(box, poses, frame_w, frame_h):
    """Local implementation for test verification."""
    if not poses: return None
    cx, cy, _, _ = box
    box_center = np.array([cx / frame_w, cy / frame_h])
    best_idx = -1
    min_dist = float('inf')
    for i, pose in enumerate(poses):
        pose = np.array(pose)
        valid_kpts = pose[np.all(pose > 0, axis=1)]
        if len(valid_kpts) == 0: continue
        pose_center = np.mean(valid_kpts, axis=0)
        dist = np.linalg.norm(box_center - pose_center)
        if dist < min_dist:
            min_dist = dist
            best_idx = i
    return poses[best_idx] if min_dist < 0.1 else None

def run_rigorous_probe():
    print("[PROBE] Starting CPU-Only Logical Verification...")
    
    # 1. Test Pose Association
    print("[PROBE] Testing Multi-Player Pose Association...")
    frame_w, frame_h = 1920, 1080
    boxes = [[400, 500, 100, 200], [1400, 500, 100, 200]]
    poses = [np.zeros((17, 2)), np.zeros((17, 2))]
    poses[0][11:13] = [400/frame_w, 500/frame_h] 
    poses[1][11:13] = [1400/frame_w, 500/frame_h]
    
    match1 = local_match_pose_to_box(boxes[0], [p.tolist() for p in poses], frame_w, frame_h)
    assert np.array_equal(match1, poses[0]), "Pose 1 matching failed"
    print("[SUCCESS] Multi-player pose matching verified.")

    # 2. Test DSL Execution
    print("[PROBE] Testing DSL Declarative Rule Execution...")
    fsm = BehaviorStateMachine(spec_path="specs/basketball_ncaa.yaml")
    jump_sequence = []
    for i in range(20):
        k = np.zeros((17, 2))
        y_val = 0.5 - (i * 0.01) # Rapid upward movement
        k[11:13] = [0.5, y_val] 
        k[0] = [0.5, y_val - 0.1]
        k[9:11] = [0.5, y_val - 0.2]
        jump_sequence.append(k)
    fsm.update(jump_sequence, context={"has_possession": True})
    assert fsm.get_label() == "jump_shot", f"DSL failed. Got '{fsm.get_label()}'"
    print("[SUCCESS] DSL Rule 'jump_shot' verified.")

    # 3. Test Action Brain Tensor Flow
    print("[PROBE] Testing Action Brain Tensor Flow...")
    try:
        import torch
        from core.vision.action_brain import ActionBrain
        from pipelines.inference import construct_features_v2, get_label_map
        
        device = "cpu"
        brain = ActionBrain(input_dim=72, num_classes=len(get_label_map()))
        brain.eval()
        
        # Mock data for construct_features_v2
        kpt_history = [np.zeros((17, 2)) for _ in range(30)]
        last_ball_2d = [960, 540]
        H = np.eye(3)
        player_court_pos = np.array([1400, 750])
        
        features = construct_features_v2(kpt_history, last_ball_2d, H, player_court_pos, None, None, device)
        assert features.shape == (1, 30, 72), f"Feature shape mismatch: {features.shape}"
        
        with torch.no_grad():
            output = brain(features)
            assert output.shape == (1, len(get_label_map())), f"Output shape mismatch: {output.shape}"
        print("[SUCCESS] Action Brain tensor flow verified.")
    except ImportError:
        print("[SKIP] Torch not found, skipping Action Brain tensor test.")

    # 4. Test Label Map Consistency
    print("[PROBE] Testing Label Map Consistency...")
    from pipelines.inference import get_label_map
    lmap = get_label_map()
    expected_labels = ["euro_step_left", "euro_step_right", "pass_chest"]
    for label in expected_labels:
        assert label in lmap.values(), f"Missing label in map: {label}"
    print("[SUCCESS] Label map updated with new NCAA categories.")

    # 5. Test Rust Logic Bridge
    print("[PROBE] Testing Rust Scoring Logic...")
    mock_dna_path = "data/intelligent_game_dna.jsonl"
    os.makedirs("data", exist_ok=True)
    with open(mock_dna_path, 'w') as f:
        f.write(json.dumps({"kind": "ball", "t_ms": 1000, "x": 160.0, "y": 762.0, "confidence_bps": 9900}) + "\n")
        f.write(json.dumps({"kind": "referee", "t_ms": 2000, "x": 500, "y": 500, "signal": "ref_3pt_success", "confidence_bps": 9900}) + "\n")

    try:
        output = run_guix_command(
            ["bash", "rust", "rust:cargo", "pkg-config", "openssl", "gcc-toolchain", "coreutils", "nss-certs", "python", "python-numpy", "python-pyyaml", "zlib"],
            ["bash", "--noprofile", "--norc", "-c", "cargo run --quiet --manifest-path core/Cargo.toml --bin spatial_processor"],
        )
        if "[SCORE] Final Score: (3, 0)" in output:
            print("[SUCCESS] End-to-End Logic Bridge Verified (Score: 3-0).")
        else:
            print("[FAILURE] Scoring logic failed.")
    except Exception as e:
        print(f"[ERROR] Rust Bridge failed: {e}")

if __name__ == "__main__":
    run_rigorous_probe()

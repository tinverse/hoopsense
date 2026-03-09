import json
import os
import subprocess
import numpy as np
from pipelines.inference import match_pose_to_box
from pipelines.behavior_engine import BehaviorStateMachine

def run_rigorous_probe():
    print("[PROBE] Starting Rigorous Foundation Verification...")
    
    # 1. Test Pose Association (Fix for Finding #5)
    print("[PROBE] Testing Multi-Player Pose Association...")
    frame_w, frame_h = 1920, 1080
    # Two detection boxes
    boxes = [
        [400, 500, 100, 200], # Player 1
        [1400, 500, 100, 200] # Player 2
    ]
    # Two poses (normalized coordinates)
    poses = [
        np.zeros((17, 2)), # Pose at 400/1920, 500/1080
        np.zeros((17, 2))  # Pose at 1400/1920, 500/1080
    ]
    poses[0][11:13] = [400/frame_w, 500/frame_h] # Hip center for box 1
    poses[1][11:13] = [1400/frame_w, 500/frame_h] # Hip center for box 2
    
    match1 = match_pose_to_box(boxes[0], [p.tolist() for p in poses], frame_w, frame_h)
    match2 = match_pose_to_box(boxes[1], [p.tolist() for p in poses], frame_w, frame_h)
    
    # Assert that box 1 matched pose 0 and box 2 matched pose 1
    assert match1 == poses[0].tolist(), "Pose 1 matching failed"
    assert match2 == poses[1].tolist(), "Pose 2 matching failed"
    print("[SUCCESS] Pose association verified for multiple players.")

    # 2. Test DSL Execution (Fix for Finding #4)
    print("[PROBE] Testing Declarative DSL Rule Execution...")
    fsm = BehaviorStateMachine(spec_path="specs/basketball_ncaa.yaml")
    
    # Simulate a jump shot sequence (Matching the DSL predicates)
    # Rule id: "jump_shot" -> metric: "velocity_z" operator: ">" threshold: 0.02
    # In normalized Y, Z-up is Y-decreasing.
    jump_sequence = []
    for i in range(20):
        k = np.zeros((17, 2))
        y_val = 0.5 - (i * 0.01) # Rapid upward movement
        k[11:13] = [0.5, y_val] # Hips
        k[0] = [0.5, y_val - 0.1] # Head
        k[9:11] = [0.5, y_val - 0.2] # Wrists (Above head)
        jump_sequence.append(k)
        
    fsm.update(jump_sequence)
    # Should trigger the DSL rule "jump_shot"
    assert fsm.get_label() == "jump_shot", f"DSL Execution failed. Expected 'jump_shot', got '{fsm.get_label()}'"
    print("[SUCCESS] DSL Rule 'jump_shot' successfully executed via predicates.")

    # 3. Trigger Rust Logic Bridge (Logic & Bridge Verification)
    print("[PROBE] Running Rust Logic Bridge (Spatial + Ledger)...")
    mock_dna_path = "data/intelligent_game_dna.jsonl"
    os.makedirs("data", exist_ok=True)
    with open(mock_dna_path, 'w') as f:
        # Ball detection at rim (triggering Rust physics)
        f.write(json.dumps({"kind": "ball", "t_ms": 1000, "x": 160.0, "y": 762.0, "confidence_bps": 9900}) + "\n")
        # Ref confirmation
        f.write(json.dumps({"kind": "referee", "t_ms": 2000, "x": 500, "y": 500, "signal": "ref_3pt_success", "confidence_bps": 9900}) + "\n")

    try:
        cmd = "guix shell -m guix.scm -- bash -lc 'cargo run --quiet --manifest-path core/Cargo.toml --bin spatial_processor'"
        output = subprocess.check_output(cmd, shell=True).decode("utf-8")
        print(output)
        if "[SCORE] Final Score: (3, 0)" in output:
            print("[SUCCESS] End-to-End Logic Bridge Verified.")
        else:
            print("[FAILURE] Scoring logic failed.")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Rust Bridge failed: {e}")

if __name__ == "__main__":
    run_rigorous_probe()

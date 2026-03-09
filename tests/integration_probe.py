import json
import os
import subprocess
import numpy as np
from pipelines.behavior_engine import BehaviorStateMachine

def run_rigorous_probe():
    print("[PROBE] Starting Rigorous End-to-End Verification...")
    
    # 1. Setup Ground Truth
    # A point at the left rim center in NCAA (160.0, 762.0, 304.8)
    # Using generate_data.py projection logic:
    # cam_pos = (0, -600, 250), K = [[1000, 0, 960], [0, 1000, 540], [0, 0, 1]]
    # Since generate_data.py uses identity R, we mock a 'perfect' detection.
    
    # Simulate a ball passing through the rim at t=1000ms
    # Then a referee signal at t=2000ms
    mock_dna_path = "data/intelligent_game_dna.jsonl"
    os.makedirs("data", exist_ok=True)
    
    with open(mock_dna_path, 'w') as f:
        # Ball detection (Exactly at left rim X,Y with identity H)
        ball_row = {
            "kind": "ball", "track_id": 99, "t_ms": 1000,
            "x": 160.0, "y": 762.0,
            "confidence_bps": 9900
        }
        f.write(json.dumps(ball_row) + "\n")
        
        # Referee signal (3-pt Success)
        ref_row = {
            "kind": "referee", "track_id": 1, "t_ms": 2000,
            "x": 500.0, "y": 500.0,
            "confidence_bps": 9900,
            "signal": "ref_3pt_success"
        }
        f.write(json.dumps(ref_row) + "\n")

    # 2. Trigger Rust Bridge
    print("[PROBE] Running Rust Spatial Processor...")
    try:
        cmd = "guix shell -m guix.scm -- bash -lc 'cargo run --quiet --manifest-path core/Cargo.toml --bin spatial_processor'"
        output = subprocess.check_output(cmd, shell=True).decode("utf-8")
        print(output)
        
        # 3. Assertions
        if "[SCORE] Final Score: (3, 0)" in output:
            print("[SUCCESS] End-to-End Scoring Logic Verified: (Ball -> Rim -> Ref -> Score)")
        else:
            print("[FAILURE] Scoring logic failed. Check ledger/bridge integration.")
            
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Rust Bridge failed: {e}")

if __name__ == "__main__":
    run_rigorous_probe()

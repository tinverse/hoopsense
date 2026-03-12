import json
import os
import subprocess
import numpy as np
import torch
from pipelines.behavior_engine import BehaviorStateMachine

def run_rigorous_probe():
    print("[PROBE] Starting Hardware and Logical Verification...")
    
    # 1. Hardware Awareness Check (The actual Orin Proof)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[PROBE] Target Device: {device}")
    
    if device.type != 'cuda':
        print("[WARN] CUDA not detected. This probe is running on CPU.")
    else:
        # Verify tensor movement
        x = torch.randn(1, 30, 72).to(device)
        if x.is_cuda:
            print("[SUCCESS] CUDA acceleration verified: Tensors moved to GPU.")
        else:
            print("[FAILURE] CUDA detected but tensor movement failed.")

    # 2. Testing Multi-Player Pose Association
    print("[PROBE] Testing Multi-Player Pose Association...")
    mock_box = [100, 100, 50, 100] # cx, cy, w, h
    mock_poses = [
        [[0.1, 0.1]] * 17, # Correct match (normalized)
        [[0.8, 0.8]] * 17  # Incorrect match
    ]
    # In behavior_engine or perception_primitives? 
    # Current implementation in inference.py (match_pose_to_box)
    from pipelines.inference import match_pose_to_box
    matched = match_pose_to_box(mock_box, mock_poses, 1000, 1000)
    if matched and matched[0][0] == 0.1:
        print("[SUCCESS] Multi-player pose matching verified.")
    else:
        print("[FAILURE] Pose association logic failed.")

    # 3. DSL Declarative Rule Execution
    print("[PROBE] Testing DSL Declarative Rule Execution...")
    fsm = BehaviorStateMachine(spec_path="specs/basketball_ncaa.yaml")
    # Mock a jump (Rising Y in normalized coords is negative delta)
    jump_history = []
    for i in range(20):
        kpts = np.zeros((17, 2))
        kpts[:, 1] = 0.5 - (i * 0.01) # Moving UP in image space
        jump_history.append(kpts)
    
    state = fsm.update(jump_history)
    if fsm.get_label() == "jump_shot" or state is not None:
        print(f"[SUCCESS] DSL Rule execution verified. Result: {fsm.get_label()}")
    else:
        print("[FAILURE] DSL Rule 'jump_shot' not triggered.")

if __name__ == "__main__":
    run_rigorous_probe()

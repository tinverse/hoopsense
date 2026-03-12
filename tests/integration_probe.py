import os
import sys

if not sys.flags.no_user_site:
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    os.execvpe(sys.executable, [sys.executable, "-s", *sys.argv], env)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import subprocess

import numpy as np

from pipelines.behavior_engine import BehaviorStateMachine
from pipelines.perception_primitives import match_pose_to_box


def run_rust_bridge():
    output = subprocess.check_output(
        [
            "cargo",
            "run",
            "--quiet",
            "--manifest-path",
            "core/Cargo.toml",
            "--bin",
            "spatial_processor",
        ],
        text=True,
    )
    assert "[SCORE] Final Score: (3, 0)" in output, output
    print("[SUCCESS] Rust scoring bridge verified.")


def run_rigorous_probe():
    print("[PROBE] Starting hardware and logical verification...")

    try:
        import torch
    except ImportError:
        torch = None

    device = "cpu"
    if torch is not None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[PROBE] Target Device: {device}")
        if device.type == "cuda":
            tensor = torch.randn(1, 30, 72, device=device)
            assert tensor.is_cuda, "CUDA detected but tensor movement failed."
            print("[SUCCESS] CUDA acceleration verified: tensor moved to GPU.")
        else:
            print("[WARN] CUDA not detected. Continuing with CPU logical verification.")
    else:
        print("[SKIP] Torch not found. Continuing with CPU logical verification.")

    print("[PROBE] Testing Multi-Player Pose Association...")
    mock_box = [100, 100, 50, 100]
    mock_poses = [np.array([[0.1, 0.1]] * 17), np.array([[0.8, 0.8]] * 17)]
    matched = match_pose_to_box(mock_box, mock_poses, 1000, 1000)
    assert matched and matched[0][0] == 0.1, "Pose association logic failed."
    print("[SUCCESS] Multi-player pose matching verified.")

    print("[PROBE] Testing DSL Declarative Rule Execution...")
    fsm = BehaviorStateMachine(spec_path="specs/basketball_ncaa.yaml")
    jump_history = []
    for i in range(20):
        kpts = np.zeros((17, 2))
        y_val = 0.5 - (i * 0.01)
        kpts[11] = [0.4, y_val]
        kpts[12] = [0.6, y_val]
        kpts[0] = [0.5, 0.4]
        kpts[9] = [0.4, 0.2]
        kpts[10] = [0.6, 0.2]
        jump_history.append(kpts)
    fsm.update(jump_history, context={"has_possession": True})
    assert fsm.get_label() == "jump_shot", fsm.get_label()
    print("[SUCCESS] DSL Rule 'jump_shot' verified.")

    print("[PROBE] Testing Action Brain Tensor Flow...")
    if torch is None:
        print("[SKIP] Torch not found. Skipping Action Brain tensor flow test.")
    else:
        from core.vision.action_brain import ActionBrain
        from pipelines.inference import construct_features_v2, get_label_map

        brain = ActionBrain(input_dim=72, num_classes=len(get_label_map())).to(device)
        brain.eval()
        kpt_history = [np.zeros((17, 2)) for _ in range(30)]
        last_ball_2d = [960, 540]
        H = np.eye(3)
        player_court_pos = np.array([1400, 750])
        features = construct_features_v2(kpt_history, last_ball_2d, H, player_court_pos, None, None, str(device))
        assert features.shape == (1, 30, 72), features.shape
        with torch.no_grad():
            output = brain(features)
            assert output.shape == (1, len(get_label_map())), output.shape
        print("[SUCCESS] Action Brain tensor flow verified.")

    print("[PROBE] Testing Label Map Consistency...")
    from pipelines.inference import get_label_map

    label_values = set(get_label_map().values())
    for label in ["euro_step_left", "euro_step_right", "pass_chest"]:
        assert label in label_values, label
    print("[SUCCESS] Label map consistency verified.")

    print("[PROBE] Testing Rust Logic Bridge...")
    run_rust_bridge()


if __name__ == "__main__":
    run_rigorous_probe()

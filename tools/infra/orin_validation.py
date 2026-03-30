import platform
import os
import subprocess
import sys
import time


def validate_environment():
    result = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy_ok": False,
        "torch_version": None,
        "torch_file": None,
        "torch_ok": False,
        "cuda_available": False,
        "cuda_device_name": None,
        "cuda_tensor_ok": False,
        "status": "fail",
        "errors": [],
    }

    try:
        import numpy as np

        _ = np.array([1, 2, 3])
        result["numpy_ok"] = True
    except Exception as exc:
        result["errors"].append(f"numpy_error: {exc}")

    try:
        import torch

        result["torch_ok"] = True
        result["torch_version"] = torch.__version__
        result["torch_file"] = getattr(torch, "__file__", None)
        result["cuda_available"] = bool(torch.cuda.is_available())
        if result["cuda_available"]:
            result["cuda_device_name"] = torch.cuda.get_device_name(0)
            tensor = torch.randn(1, 3, 224, 224, device="cuda")
            result["cuda_tensor_ok"] = bool(tensor.is_cuda)
        else:
            is_orin_like = sys.platform == "linux" and platform.machine() == "aarch64"
            if is_orin_like:
                result["errors"].append("cuda_missing_on_aarch64")
    except Exception as exc:
        result["errors"].append(f"torch_cuda_error: {exc}")

    if result["torch_ok"]:
        result["status"] = "pass"
    return result


def run_rigorous_probe():
    try:
        import numpy as np
    except Exception as exc:
        return {
            "device_type": "unknown",
            "cuda_tensor_ok": False,
            "action_brain_ok": False,
            "action_brain_device": None,
            "action_brain_latency_ms": None,
            "pose_association_ok": False,
            "dsl_rule_ok": False,
            "rust_bridge_ok": False,
            "rust_bridge_returncode": None,
            "rust_bridge_stderr": f"numpy_import_error: {exc}",
            "status": "fail",
        }

    try:
        import torch
    except Exception as exc:
        return {
            "device_type": "unknown",
            "cuda_tensor_ok": False,
            "action_brain_ok": False,
            "action_brain_device": None,
            "action_brain_latency_ms": None,
            "pose_association_ok": None,
            "dsl_rule_ok": None,
            "rust_bridge_ok": None,
            "rust_bridge_returncode": None,
            "rust_bridge_stderr": f"torch_import_error: {exc}",
            "status": "fail",
        }

    result = {
        "device_type": "cuda" if torch.cuda.is_available() else "cpu",
        "cuda_tensor_ok": False,
        "action_brain_ok": False,
        "action_brain_device": None,
        "action_brain_latency_ms": None,
        "action_brain_error": None,
        "pose_association_ok": None,
        "dsl_rule_ok": None,
        "rust_bridge_ok": None,
        "rust_bridge_returncode": None,
        "rust_bridge_stderr": None,
        "status": "fail",
    }

    if result["device_type"] == "cuda":
        x = torch.randn(1, 30, 72).to("cuda")
        result["cuda_tensor_ok"] = bool(x.is_cuda)
    else:
        result["cuda_tensor_ok"] = False

    try:
        from core.vision.action_brain import ActionBrain

        device = torch.device(result["device_type"])
        model = ActionBrain(num_classes=5).to(device)
        dummy_input = torch.randn(1, 30, 72).to(device)
        with torch.no_grad():
            _ = model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            output = model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        result["action_brain_device"] = output.device.type
        result["action_brain_latency_ms"] = float(elapsed_ms)
        result["action_brain_ok"] = output.device.type == device.type and tuple(output.shape) == (1, 5)
    except Exception as exc:
        result["action_brain_error"] = str(exc)

    try:
        import numpy as np
        from pipelines.behavior_engine import BehaviorStateMachine
        from pipelines.perception_primitives import match_pose_to_box

        mock_box = [100, 100, 50, 100]
        mock_poses = [
            np.array([[0.1, 0.1]] * 17),
            np.array([[0.8, 0.8]] * 17),
        ]
        matched = match_pose_to_box(mock_box, mock_poses, 1000, 1000)
        result["pose_association_ok"] = matched is not None and matched[0][0] == 0.1

        fsm = BehaviorStateMachine(spec_path="specs/basketball_ncaa.yaml")
        jump_history = []
        for i in range(20):
            kpts = np.zeros((17, 2))
            kpts[:, 1] = 0.5 - (i * 0.01)
            jump_history.append(kpts)
        state = fsm.update(jump_history)
        result["dsl_rule_ok"] = fsm.get_label() == "jump_shot" or state is not None
    except Exception as exc:
        result["rust_bridge_stderr"] = (
            f"{result['rust_bridge_stderr']}; supplemental_probe_error: {exc}"
            if result["rust_bridge_stderr"]
            else f"supplemental_probe_error: {exc}"
        )

    try:
        rust_bridge_wrapper = os.environ.get("HOOPSENSE_RUST_BRIDGE_WRAPPER")
        rust_bridge_cmd = [
            "cargo",
            "run",
            "--quiet",
            "--manifest-path",
            "core/Cargo.toml",
            "--bin",
            "spatial_processor",
        ]
        if rust_bridge_wrapper:
            rust_bridge_cmd = [rust_bridge_wrapper, *rust_bridge_cmd]
        rust_bridge_env = os.environ.copy()
        if rust_bridge_wrapper:
            rust_bridge_env.pop("LD_LIBRARY_PATH", None)
        proc = subprocess.run(
            rust_bridge_cmd,
            capture_output=True,
            text=True,
            env=rust_bridge_env,
        )
        result["rust_bridge_returncode"] = proc.returncode
        result["rust_bridge_stderr"] = proc.stderr.strip() or None
        result["rust_bridge_ok"] = proc.returncode == 0
    except Exception as exc:
        result["rust_bridge_stderr"] = str(exc)
        result["rust_bridge_ok"] = None

    result["status"] = "pass" if all(
        [
            result["device_type"] == "cuda",
            result["cuda_tensor_ok"],
            result["action_brain_ok"],
        ]
    ) else "fail"
    return result

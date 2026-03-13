import os
import subprocess
import yaml
import sys


def verify_gpu():
    print("[COLAB] Verifying Hardware Accelerator...")
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"[SUCCESS] GPU Detected: {device_name}")
            return True
        else:
            print("[WARN] No GPU detected. Running on CPU.")
            return False
    except ImportError:
        print("[ERROR] PyTorch not installed correctly.")
        return False


def install_deps(packages):
    print(f"[COLAB] Installing dependencies: {packages}")
    cmd = ["pip", "install", "--quiet"] + packages
    subprocess.run(cmd, check=True)


def load_config(config_path):
    if not os.path.exists(config_path):
        print(f"[ERROR] Config file not found: {config_path}")
        return None
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def sync_outputs(output_dir, drive_path):
    print(f"[COLAB] Syncing {output_dir} to {drive_path}...")
    # Assume drive is already mounted by caller
    os.makedirs(drive_path, exist_ok=True)
    cmd = ["rsync", "-avz", output_dir + "/", drive_path + "/"]
    subprocess.run(cmd)


def run_pipeline(config_path):
    config = load_config(config_path)
    if not config:
        return

    # 1. Hardware Check
    has_gpu = verify_gpu()

    # 2. Package Injection (from config)
    runtime_pkgs = config.get("runtime", {}).get("packages", [])
    if runtime_pkgs:
        install_deps(runtime_pkgs)

    # 3. Execution logic (Simplified for bootstrap)
    # This would normally call pipelines/inference.py or training
    print(f"[COLAB] HoopSense OS Initialized for Project: {config.get('project_name')}")
    print(f"[INFO] Using {config.get('perception', {}).get('model')} for inference.")

    if has_gpu:
        print("[COLAB] Ready for high-performance video analysis.")
    else:
        print("[COLAB] Running in low-power logic-only mode.")


def cleanup():
    """Optional: cleanup temporary frames or unmount."""
    try:
        from google.colab import runtime
        runtime.unassign()
    except Exception:
        pass


if __name__ == "__main__":
    config_file = "hoops_config.yaml" if len(sys.argv) < 2 else sys.argv[1]
    run_pipeline(config_file)

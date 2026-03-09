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
            print(f"[INFO] GPU Detected: {device_name}")
            return True
        else:
            print("[ERROR] No GPU detected. Please set runtime to T4 GPU.")
            return False
    except ImportError:
        print("[WARN] Torch not installed yet, skipping hardware check.")
        return True

def mount_gdrive():
    print("[COLAB] Attempting to mount Google Drive...")
    try:
        from google.colab import drive
        # This will prompt for auth if not already mounted
        drive.mount('/content/drive', force_remount=True)
        return True
    except Exception as e:
        print(f"[WARN] GDrive mount failed or not in Colab env: {e}")
        # If we are in SSH, maybe it's already mounted? 
        return os.path.exists('/content/drive/MyDrive')

def setup_environment(packages):
    print("[COLAB] Installing Dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)

def run_pipeline(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 1. Setup Env
    runtime_cfg = config.get("runtime", {})
    setup_environment(runtime_cfg.get("packages", []))
    
    # 2. Check GPU
    if not verify_gpu():
        print("[CRITICAL] Aborting due to missing GPU.")
        return

    # 3. Mount Drive
    if config.get("video_source") == "gdrive":
        if not mount_gdrive():
            print("[ERROR] Google Drive required but not mounted.")
            return
        video_path = config.get("gdrive_video_path")
    else:
        video_path = config.get("video_path")

    output_dir = config.get("output_dir", "data")
    
    print(f"[COLAB] Starting Pipeline for {video_path}...")
    
    # Ensure local path is in sys.path for imports
    sys.path.append(os.getcwd())
    
    from pipelines.inference import extract_game_dna
    extract_game_dna(video_path, output_dir=output_dir)
    
    print(f"[COLAB] Pipeline Complete. Outputs in {output_dir}")

if __name__ == "__main__":
    config_file = "hoops_config.yaml" if len(sys.argv) < 2 else sys.argv[1]
    run_pipeline(config_file)

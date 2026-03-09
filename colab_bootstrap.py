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
        drive.mount('/content/drive', force_remount=True)
        return True
    except Exception as e:
        print(f"[WARN] GDrive mount failed or not in Colab env: {e}")
        return os.path.exists('/content/drive/MyDrive')

def setup_environment(packages):
    print("[COLAB] Installing Dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)

def validate_paths(config):
    print("[COLAB] Validating Data Paths...")
    if config.get("video_source") == "gdrive":
        gdrive_root = "/content/drive/MyDrive/HoopSense"
        if not os.path.exists(gdrive_root):
            print(f"[ERROR] Required Google Drive folder missing: {gdrive_root}")
            return False
        try:
            probe_path = os.path.join(gdrive_root, ".hoopsense_probe")
            with open(probe_path, "w") as f:
                f.write("HoopSense Connectivity Test")
            os.remove(probe_path)
            print("[SUCCESS] Google Drive Read/Write verified.")
        except Exception as e:
            print(f"[ERROR] Drive Write Test failed: {e}")
            return False
        video_path = config.get("gdrive_video_path")
        if not os.path.exists(video_path):
            print(f"[ERROR] Video file not found: {video_path}")
            return False
    output_dir = config.get("output_dir", "data")
    os.makedirs(output_dir, exist_ok=True)
    return True

def report_drive_stats(config):
    print("[COLAB] Auditing Google Drive Data...")
    gdrive_root = "/content/drive/MyDrive/HoopSense"
    if os.path.exists(gdrive_root):
        for root, dirs, files in os.walk(gdrive_root):
            for f in files:
                f_path = os.path.join(root, f)
                size_mb = os.path.getsize(f_path) / (1024 * 1024)
                print(f"  - {f}: {size_mb:.1f} MB")
    else:
        print("[WARN] HoopSense folder not found in Drive.")

def run_pipeline(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    setup_environment(config.get("runtime", {}).get("packages", []))
    verify_gpu()
    if config.get("video_source") == "gdrive":
        if not mount_gdrive():
            print("[ERROR] Google Drive required but not mounted.")
            return
        report_drive_stats(config)
    if not validate_paths(config):
        print("[CRITICAL] Aborting due to invalid paths.")
        return
    video_path = config.get("gdrive_video_path") if config.get("video_source") == "gdrive" else config.get("video_path")
    output_dir = config.get("output_dir", "data")
    print(f"[COLAB] Starting Pipeline for {video_path}...")
    sys.path.append(os.getcwd())
    from pipelines.inference import extract_game_dna
    extract_game_dna(video_path, output_dir=output_dir)
    print(f"[COLAB] Pipeline Complete. Outputs in {output_dir}")
    print("[COLAB] Terminating Instance...")
    try:
        from google.colab import runtime
        runtime.unassign()
    except Exception:
        pass

if __name__ == "__main__":
    config_file = "hoops_config.yaml" if len(sys.argv) < 2 else sys.argv[1]
    run_pipeline(config_file)

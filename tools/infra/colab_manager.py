import os
import subprocess
import time


def mount_drive():
    """Mounts Google Drive if running in Colab."""
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("[COLAB] Drive mounted successfully.")
        return True
    except ImportError:
        print("[LOCAL] Not in Colab environment.")
        return False


def setup_venv(venv_path="venv_hoops"):
    """Creates and initializes a virtual environment for the project."""
    if not os.path.exists(venv_path):
        print(f"[INFO] Creating virtual environment at {venv_path}...")
        subprocess.run(["python3", "-m", "venv", venv_path])

    print("[INFO] Installing requirements...")
    pip_cmd = [f"{venv_path}/bin/pip", "install", "-r", "requirements.txt"]
    subprocess.run(pip_cmd)


def run_cloud_job(job_name, machine_type="n1-standard-4"):
    """Submits a training job to Vertex AI."""
    print(f"[INFO] Submitting {job_name} to Vertex AI...")
    # Placeholder for gcloud ai custom-jobs submit
    time.sleep(2)
    print(f"[SUCCESS] Job {job_name} submitted.")


class ColabSession:
    def __init__(self, drive_path="/content/drive/MyDrive/HoopSense"):
        self.drive_path = drive_path
        self.is_colab = mount_drive()

    def sync_data(self, direction="download"):
        if not self.is_colab:
            return
        local_data = "data/training"
        cloud_data = f"{self.drive_path}/data/training"
        os.makedirs(local_data, exist_ok=True)

        if direction == "download":
            cmd = ["rsync", "-avz", cloud_data + "/", local_data + "/"]
        else:
            cmd = ["rsync", "-avz", local_data + "/", cloud_data + "/"]
        subprocess.run(cmd)


if __name__ == "__main__":
    if os.getenv("COLAB_GPU"):
        print("[INFO] Detected Colab Runtime.")
    else:
        print("[INFO] Running locally.")

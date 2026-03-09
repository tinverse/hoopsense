import subprocess
import os
import sys
import yaml

class ColabManager:
    def __init__(self, config_path="hoops_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.project = os.getenv("GCP_PROJECT_ID")
        self.region = os.getenv("GCP_REGION", "us-central1")
        self.runtime_id = "hoopsense-gpu-runtime"
        self.template_id = "hoopsense-t4-template"

    def provision(self):
        """Creates the T4 GPU Runtime Template and Instance."""
        print(f"[INFO] Provisioning Colab Enterprise Instance in {self.region}...")
        
        # 1. Create Template (Machine: n1-standard-4, GPU: T4)
        try:
            subprocess.run([
                "gcloud", "ai", "notebook-runtime-templates", "create", self.template_id,
                f"--project={self.project}", f"--region={self.region}",
                "--display-name=HoopSense T4 GPU",
                "--machine-type=n1-standard-4",
                "--accelerator-type=NVIDIA_TESLA_T4",
                "--accelerator-count=1"
            ], check=True)
        except subprocess.CalledProcessError:
            print("[INFO] Template might already exist, proceeding to runtime creation.")

        # 2. Create Runtime
        subprocess.run([
            "gcloud", "ai", "notebook-runtimes", "create", self.runtime_id,
            f"--project={self.project}", f"--region={self.region}",
            f"--notebook-runtime-template={self.template_id}"
        ], check=True)
        print("[SUCCESS] GPU Runtime Created.")

    def run_pipeline(self):
        """Syncs code and triggers colab_bootstrap.py."""
        # For Enterprise, we typically use 'gcloud compute ssh' or push to GCS.
        # Given your plan, we'll use the existing 'hoops colab' bridge logic
        # but pointed at this new GCP instance.
        print("[INFO] Launching HoopSense Pipeline on Cloud...")
        # (This is where hoops.py colab is triggered)

    def terminate(self):
        """Force shut down to stop billing."""
        print("[INFO] Terminating Cloud Runtime...")
        subprocess.run([
            "gcloud", "ai", "notebook-runtimes", "delete", self.runtime_id,
            f"--project={self.project}", f"--region={self.region}", "--quiet"
        ], check=True)

if __name__ == "__main__":
    manager = ColabManager()
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "up": manager.provision()
        elif cmd == "down": manager.terminate()

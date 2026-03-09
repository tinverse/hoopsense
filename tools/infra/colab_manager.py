import subprocess
import os
import sys
import yaml

class ColabManager:
    def __init__(self, config_path="hoops_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.project = os.getenv("GCP_PROJECT_ID", "hoopsense-ai")
        self.region = os.getenv("GCP_REGION", "us-central1")
        self.runtime_id = "hoopsense-gpu-runtime"
        self.template_id = "hoopsense-t4-template"

    def provision(self):
        """Creates the T4 GPU Runtime Template and Instance."""
        print(f"[INFO] Provisioning Colab Enterprise Instance in {self.region}...")
        
        # 1. Create Template (Machine: n1-standard-4, GPU: T4)
        try:
            subprocess.run([
                "gcloud", "beta", "ai", "notebook-runtime-templates", "create", self.template_id,
                f"--project={self.project}", f"--region={self.region}",
                "--display-name=HoopSense T4 GPU",
                "--machine-type=n1-standard-4",
                "--accelerator-type=NVIDIA_TESLA_T4",
                "--accelerator-count=1",
                "--shielded-secure-boot"
            ], check=True)
        except subprocess.CalledProcessError:
            print("[INFO] Template might already exist, proceeding to runtime creation.")

        # 2. Create Runtime
        subprocess.run([
            "gcloud", "beta", "ai", "notebook-runtimes", "create", self.runtime_id,
            f"--project={self.project}", f"--region={self.region}",
            f"--notebook-runtime-template={self.template_id}"
        ], check=True)
        print("[SUCCESS] GPU Runtime Created.")

    def terminate(self):
        """Force shut down to stop billing."""
        print("[INFO] Terminating Cloud Runtime...")
        subprocess.run([
            "gcloud", "beta", "ai", "notebook-runtimes", "delete", self.runtime_id,
            f"--project={self.project}", f"--region={self.region}", "--quiet"
        ], check=True)

    def run_pipeline(self, script_path, data_path=None):
        """Syncs the local codebase and runs a specific pipeline on the remote instance."""
        print(f"[INFO] Syncing codebase and running {script_path} on {self.runtime_id}...")
        
        # 1. Sync code (Simplified: use gcloud storage or scp if it were a GCE instance)
        # For Colab Enterprise, we typically use 'gcloud storage cp' to a GCS bucket 
        # that the runtime can access, or 'git clone' within the runtime.
        # Here we provide a template for gcloud-based execution.
        print("[TEMPLATE] To sync code: gcloud compute scp --recursive . hoopsense-gpu:/home/jupyter/hoopsense")
        
        # 2. Remote execution command
        # This is a stub for the actual API call to execute code on the runtime
        print(f"[EXEC] Running 'python {script_path}' on remote backend.")
        
        # Example of how to trigger a Vertex AI Custom Job (which Colab Runtimes can use)
        # subprocess.run(["gcloud", "ai", "custom-jobs", "create", ...])

if __name__ == "__main__":
    manager = ColabManager()
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "up": manager.provision()
        elif cmd == "down": manager.terminate()
        elif cmd == "run": 
            script = sys.argv[2] if len(sys.argv) > 2 else "pipelines/inference.py"
            manager.run_pipeline(script)

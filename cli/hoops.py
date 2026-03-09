import os
import sys
import subprocess

REMOTE_HOST = os.getenv("HOOPS_REMOTE_HOST", "root@localhost")
REMOTE_PORT = os.getenv("HOOPS_REMOTE_PORT", "22")
def sync():
    if "your-tunnel" in REMOTE_HOST:
        print("[ERROR] Update .envrc first.")
        sys.exit(1)
    print(f"[INFO] Syncing code to {REMOTE_HOST}:{REMOTE_PORT}...")
    subprocess.run([
        "rsync", "-avz",
        "-e", f"ssh -p {REMOTE_PORT} -o StrictHostKeyChecking=no",
        "--exclude", ".git",
        "--exclude", "data/",
        "--exclude", "core/target",
        "./", f"{REMOTE_HOST}:{REMOTE_DIR}"
    ], check=True)

def collect():
    print(f"[INFO] Collecting results from {REMOTE_HOST}...")
    # Default to data/ if not specified otherwise
    os.makedirs("data/remote", exist_ok=True)
    subprocess.run([
        "rsync", "-avz",
        "-e", f"ssh -p {REMOTE_PORT} -o StrictHostKeyChecking=no",
        f"{REMOTE_HOST}:{REMOTE_DIR}/data/", "./data/remote/"
    ], check=True)

def run_remote(script_path):
    sync()
    print(f"[INFO] Executing {script_path} on remote...")
    subprocess.run([
        "ssh", "-p", REMOTE_PORT,
        "-o StrictHostKeyChecking=no",
        REMOTE_HOST,
        f"cd {REMOTE_DIR} && python3 {script_path}"
    ], check=True)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        if len(sys.argv) > 2:
            run_remote(sys.argv[2])
        else:
            print("Usage: hoops run <script_path>")
    elif len(sys.argv) > 1 and sys.argv[1] == "colab":
        print("[INFO] Launching Headless Colab Run...")
        run_remote("colab_bootstrap.py")
        print("[INFO] Run finished. Use 'hoops collect' to pull results.")
    elif len(sys.argv) > 1 and sys.argv[1] == "collect":
        collect()
    elif len(sys.argv) > 1 and sys.argv[1] == "sync":
        sync()
    else:
        print("Usage: hoops [sync | run <script> | colab | collect]")


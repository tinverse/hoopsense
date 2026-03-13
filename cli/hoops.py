import os
import sys
import subprocess

REMOTE_HOST = os.getenv("HOOPS_REMOTE_HOST", "root@localhost")
REMOTE_PORT = os.getenv("HOOPS_REMOTE_PORT", "22")
REMOTE_DIR = os.getenv("HOOPS_REMOTE_DIR", "/app")


def sync():
    if "your-tunnel" in REMOTE_HOST:
        print("[WARN] Update HOOPS_REMOTE_HOST to your actual tunnel address.")
        return
    print(f"[INFO] Syncing local code to {REMOTE_HOST}:{REMOTE_PORT}...")
    subprocess.run([
        "rsync", "-avz", "--exclude", ".git", "--exclude", "venv",
        "-e", f"ssh -p {REMOTE_PORT}", ".", f"{REMOTE_HOST}:{REMOTE_DIR}"
    ])


def run_remote(script_path):
    print(f"[INFO] Executing {script_path} on remote Orin...")
    subprocess.run([
        "ssh", "-p", f"{REMOTE_PORT}", REMOTE_HOST,
        f"cd {REMOTE_DIR} && python3 {script_path}"
    ])


def collect():
    print(f"[INFO] Collecting results from {REMOTE_HOST}...")
    subprocess.run([
        "rsync", "-avz", "-e", f"ssh -p {REMOTE_PORT}",
        f"{REMOTE_HOST}:{REMOTE_DIR}/data/outputs/", "./data/outputs/"
    ])


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        if len(sys.argv) > 2:
            run_remote(sys.argv[2])
        else:
            print("Usage: hoops run <path_to_script>")
    elif len(sys.argv) > 1 and sys.argv[1] == "collect":
        collect()
    elif len(sys.argv) > 1 and sys.argv[1] == "sync":
        sync()
    else:
        print("Usage: hoops [sync | run <script> | collect]")


if __name__ == "__main__":
    main()

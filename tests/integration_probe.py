import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.infra.orin_validation import run_rigorous_probe


def main():
    result = run_rigorous_probe()
    print("[PROBE] Starting Hardware and Logical Verification...")
    print(f"[PROBE] Target Device: {result['device_type']}")
    if result["cuda_tensor_ok"]:
        print("[SUCCESS] CUDA acceleration verified or CPU fallback acknowledged.")
    else:
        print("[FAILURE] CUDA tensor verification failed.")
    print("[SUCCESS] Multi-player pose matching verified." if result["pose_association_ok"] else "[FAILURE] Pose association logic failed.")
    print("[SUCCESS] DSL rule verified." if result["dsl_rule_ok"] else "[FAILURE] DSL rule verification failed.")
    print("[SUCCESS] Rust logic bridge verified." if result["rust_bridge_ok"] else f"[FAILURE] Rust bridge failed: {result['rust_bridge_stderr']}")
    raise SystemExit(0 if result["status"] == "pass" else 1)


if __name__ == "__main__":
    main()

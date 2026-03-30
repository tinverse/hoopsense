import platform
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.infra.orin_validation import validate_environment


def validate():
    result = validate_environment()
    print(f"[INFO] Python Version: {result['python_version']}")
    print(f"[INFO] Platform: {result['platform']}")
    print(f"[SUCCESS] NumPy initialized." if result["numpy_ok"] else "[FAILURE] NumPy failed.")
    print(f"[INFO] PyTorch Version: {result['torch_version']}")
    print(f"[INFO] CUDA Available: {result['cuda_available']}")

    if result["cuda_device_name"]:
        print(f"[SUCCESS] GPU Detected: {result['cuda_device_name']}")
    if result["cuda_tensor_ok"]:
        print("[SUCCESS] CUDA tensor allocation verified.")

    if result["status"] != "pass":
        for error in result["errors"]:
            print(f"[FAILURE] {error}")
        sys.exit(1)


if __name__ == "__main__":
    validate()

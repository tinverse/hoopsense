import sys
import numpy as np
import torch


def validate():
    print(f"[INFO] Python Version: {sys.version}")

    # 1. Test NumPy C-Extensions
    try:
        _ = np.array([1, 2, 3])
        print("[SUCCESS] NumPy initialized.")
    except Exception as e:
        print(f"[FAILURE] NumPy error: {e}")
        sys.exit(1)

    # 2. Test PyTorch and CUDA (The Orin Hardware Gate)
    try:
        print(f"[INFO] PyTorch Version: {torch.__version__}")
        has_cuda = torch.cuda.is_available()
        print(f"[INFO] CUDA Available: {has_cuda}")

        if has_cuda:
            device_name = torch.cuda.get_device_name(0)
            print(f"[SUCCESS] GPU Detected: {device_name}")
            _ = torch.randn(1, 3, 224, 224).cuda()
            print("[SUCCESS] CUDA tensor allocation verified.")
        else:
            if sys.platform == "linux" and "aarch64" in sys.version:
                print("[FAILURE] CUDA not detected. Hardware bridge is missing.")
                sys.exit(1)

    except Exception as e:
        print(f"[FAILURE] PyTorch/CUDA check failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    validate()

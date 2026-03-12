import sys
import numpy as np
import torch

def validate():
    print(f"[INFO] Python Version: {sys.version}")
    
    # 1. Test NumPy C-Extensions
    try:
        a = np.array([1, 2, 3])
        print(f"[SUCCESS] NumPy C-Extensions verified (v{np.__version__})")
    except Exception as e:
        print(f"[FAILURE] NumPy C-Extensions failed: {e}")
        sys.exit(1)

    # 2. Test PyTorch & Hardware Bridge
    try:
        print(f"[INFO] PyTorch Version: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"[INFO] CUDA Available: {cuda_available}")
        
        if cuda_available:
            print(f"[INFO] Device Name: {torch.cuda.get_device_name(0)}")
            # Real hardware exercise
            x = torch.randn(5, 5).cuda()
            y = x @ x
            print("[SUCCESS] CUDA Hardware Access verified via PyTorch")
        else:
            # On Orin, if CUDA is not available, the bridge is broken
            print("[FAILURE] CUDA not detected. Hardware bridge is missing.")
            sys.exit(1)
            
    except Exception as e:
        print(f"[FAILURE] PyTorch/CUDA check failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    validate()

#!/bin/bash
# HoopSense Orin Migration Utility
# This script bridges the Guix environment to the Orin's hardware GPU.

echo "[INFO] Starting HoopSense Orin Setup..."

# 1. Detect JetPack CUDA
CUDA_PATH="/usr/local/cuda"
if [ ! -d "$CUDA_PATH" ]; then
    echo "[ERROR] CUDA not found at $CUDA_PATH. Is JetPack installed?"
    exit 1
fi

# 2. Setup the "Hardware Bridge" environment variables
# These allow Guix-installed Python/Rust to see the Orin's GPU and JetPack libraries
export LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu/tegra:$CUDA_PATH/lib64:$LD_LIBRARY_PATH"
export PATH="$CUDA_PATH/bin:$PATH"
export PYTHONPATH="/usr/lib/python3/dist-packages:$PYTHONPATH"

# 3. Create a launch wrapper for the Orin Shell
cat <<EOF > hoops-orin-shell
#!/bin/bash
# HoopSense Orin GPU-Enabled Shell
guix shell -m guix_orin.scm --pure \
  --preserve=LD_LIBRARY_PATH \
  --preserve=PATH \
  --preserve=PYTHONPATH \
  --share=/usr/lib/aarch64-linux-gnu/tegra \
  --share=/usr/local/cuda \
  --share=/usr/lib/python3/dist-packages \
  --share=/etc/nv_tegra_release \
  -- bash -lc "\$@"
EOF

chmod +x hoops-orin-shell

echo "[SUCCESS] Orin Bridge created."
echo "Use './hoops-orin-shell' to run commands with GPU access."
echo "Example: ./hoops-orin-shell python3 tools/training/train_action_brain.py"

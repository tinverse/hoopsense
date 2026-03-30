#!/bin/bash
# HoopSense Orin Migration Utility
# This script bridges the Guix environment to the Orin's hardware GPU.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ORIN_VENV="${REPO_ROOT}/.venv_orin310"
TORCH_WHEEL_URL="https://pypi.jetson-ai-lab.io/jp6/cu126/+f/37d/7e156cfb4a646/torch-2.10.0-cp310-cp310-linux_aarch64.whl#sha256=37d7e156cfb4a646c4d7347597727db1529d184108f703324dfff1842cec094e"
TORCHVISION_VERSION="0.25.0"
ORIN_CV2_SRC="/usr/lib/python3.10/dist-packages/cv2"
ORIN_CV2_DST="${ORIN_VENV}/lib/python3.10/site-packages/cv2"

echo "[INFO] Starting HoopSense Orin Setup..."

# 1. Detect JetPack CUDA
CUDA_PATH="/usr/local/cuda"
if [ ! -d "$CUDA_PATH" ]; then
    echo "[ERROR] CUDA not found at $CUDA_PATH. Is JetPack installed?"
    exit 1
fi

# 2. Setup the "Hardware Bridge" environment variables
# These allow Guix-installed Python/Rust to see the Orin's GPU and JetPack libraries
export LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu/tegra:$CUDA_PATH/lib64:${LD_LIBRARY_PATH:-}"
export PATH="$CUDA_PATH/bin:${PATH:-}"
export PYTHONPATH="/usr/lib/python3/dist-packages:${PYTHONPATH:-}"

# 3. Create a launch wrapper for the Orin Shell
cat <<EOF > hoops-orin-shell
#!/bin/bash
# HoopSense Orin GPU-Enabled Shell
set -e
guix shell -m guix_orin.scm --pure \
  --preserve=LD_LIBRARY_PATH \
  --preserve=PATH \
  --preserve=PYTHONPATH \
  --share=/usr/lib/aarch64-linux-gnu/tegra \
  --share=/usr/local/cuda \
  --share=/usr/lib/python3/dist-packages \
  --share=/etc/nv_tegra_release \
  -- "\$@"
EOF

chmod +x hoops-orin-shell

# 4. Create the reproducible Python 3.10 probe environment.
if [ ! -x "${ORIN_VENV}/bin/python" ]; then
    echo "[INFO] Creating Orin probe venv at ${ORIN_VENV}"
    "${REPO_ROOT}/hoops-orin-shell" python3 -m venv "${ORIN_VENV}"
fi

echo "[INFO] Installing exact Orin probe dependencies into ${ORIN_VENV}"
"${REPO_ROOT}/hoops-orin-shell" "${ORIN_VENV}/bin/python" -m pip install -U pip setuptools wheel numpy
"${REPO_ROOT}/hoops-orin-shell" "${ORIN_VENV}/bin/python" -m pip install "${TORCH_WHEEL_URL}"
"${REPO_ROOT}/hoops-orin-shell" "${ORIN_VENV}/bin/python" -m pip install nvidia-cudss-cu12

echo "[INFO] Installing Layer 1 inference dependencies into ${ORIN_VENV}"
"${REPO_ROOT}/hoops-orin-shell" "${ORIN_VENV}/bin/python" -m pip install \
  pillow \
  pyyaml \
  requests \
  scipy \
  ultralytics-thop \
  matplotlib==3.7.5 \
  psutil \
  polars \
  flask \
  lap
"${REPO_ROOT}/hoops-orin-shell" "${ORIN_VENV}/bin/python" -m pip install --no-deps ultralytics
"${REPO_ROOT}/hoops-orin-shell" "${ORIN_VENV}/bin/python" -m pip install --no-deps "torchvision==${TORCHVISION_VERSION}"

if [ -d "${ORIN_CV2_SRC}" ]; then
    rm -rf "${ORIN_CV2_DST}"
    ln -s "${ORIN_CV2_SRC}" "${ORIN_CV2_DST}"
    echo "[INFO] Linked Jetson system OpenCV into ${ORIN_VENV}"
else
    echo "[WARN] Jetson system OpenCV not found at ${ORIN_CV2_SRC}"
fi

echo "[SUCCESS] Orin Bridge created."
echo "Use './hoops-orin-shell' to run commands with GPU access."
echo "Example: ./hoops-orin-shell python3 tools/training/train_action_brain.py"
echo "Orin probe venv: ${ORIN_VENV}"
echo "Validation: ./scripts/run_orin_cuda_probe.sh"
echo "Labeller: ./scripts/run_layer1_labeller.sh"
echo "Layer 1 artifact generation: ./scripts/generate_layer1_annotations.sh data/raw_clips/youth/<clip>.mp4"

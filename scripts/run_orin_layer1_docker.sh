#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE_TAG="${HOOPSENSE_ORIN_SAM3_IMAGE:-hoopsense-orin:sam3-exp1}"
CONTAINER_NAME="${HOOPSENSE_ORIN_SAM3_CONTAINER:-hoopsense-orin-sam3-$$}"
HF_CACHE_DIR="${HOOPSENSE_HF_CACHE_DIR:-${REPO_ROOT}/.local_models/huggingface}"
TORCH_CACHE_DIR="${HOOPSENSE_TORCH_CACHE_DIR:-${REPO_ROOT}/.local_models/torch}"
YOLO_CACHE_DIR="${HOOPSENSE_YOLO_CACHE_DIR:-${REPO_ROOT}/.local_models/ultralytics}"
EASYOCR_CACHE_DIR="${HOOPSENSE_EASYOCR_CACHE_DIR:-${REPO_ROOT}/.local_models/easyocr}"
CPU_THREADS="${HOOPSENSE_CPU_THREADS:-4}"
EXTRA_DOCKER_ARGS="${HOOPSENSE_ORIN_SAM3_DOCKER_ARGS:-}"

mkdir -p "${HF_CACHE_DIR}" "${TORCH_CACHE_DIR}" "${YOLO_CACHE_DIR}" "${EASYOCR_CACHE_DIR}"

if [ -f "${REPO_ROOT}/.secrets" ]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.secrets"
fi

DOCKER_CMD=(
  docker run --rm --name "${CONTAINER_NAME}"
  --runtime nvidia
  --network host
  --ipc host
  --shm-size=8g
  --ulimit memlock=-1
  --ulimit stack=67108864
  -v "${REPO_ROOT}:/app"
  -v "${HF_CACHE_DIR}:/cache/huggingface"
  -v "${TORCH_CACHE_DIR}:/cache/torch"
  -v "${YOLO_CACHE_DIR}:/cache/ultralytics"
  -v "${EASYOCR_CACHE_DIR}:/root/.EasyOCR"
  -w /app
  -e PYTHONPATH=/app
  -e HF_HOME=/cache/huggingface
  -e TORCH_HOME=/cache/torch
  -e YOLO_CONFIG_DIR=/cache/ultralytics
  -e OMP_NUM_THREADS="${CPU_THREADS}"
  -e OPENBLAS_NUM_THREADS="${CPU_THREADS}"
  -e MKL_NUM_THREADS="${CPU_THREADS}"
  -e NUMEXPR_NUM_THREADS="${CPU_THREADS}"
  -e LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cu12/lib:/usr/lib/aarch64-linux-gnu/tegra:/usr/local/cuda/lib64
  -e NVIDIA_VISIBLE_DEVICES=all
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
)

if [ -n "${HF_TOKEN:-}" ]; then
  DOCKER_CMD+=(-e "HF_TOKEN=${HF_TOKEN}")
fi
if [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  DOCKER_CMD+=(-e "HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}")
fi
if [ -n "${EXTRA_DOCKER_ARGS}" ]; then
  # Intentionally word-split to allow callers to pass normal docker flags.
  # shellcheck disable=SC2206
  EXTRA_ARGS_ARR=(${EXTRA_DOCKER_ARGS})
  DOCKER_CMD+=("${EXTRA_ARGS_ARR[@]}")
fi

DOCKER_CMD+=("${IMAGE_TAG}")

if [ "$#" -eq 0 ]; then
  DOCKER_CMD+=(/bin/bash)
else
  DOCKER_CMD+=("$@")
fi

exec "${DOCKER_CMD[@]}"

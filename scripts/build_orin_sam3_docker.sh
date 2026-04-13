#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE_TAG="${HOOPSENSE_ORIN_SAM3_IMAGE:-hoopsense-orin:sam3-exp1}"
BASE_IMAGE="${HOOPSENSE_ORIN_SAM3_BASE_IMAGE:-nvcr.io/nvidia/l4t-jetpack:r36.4.0}"
DINOV3_REF="${HOOPSENSE_DINOV3_REF:-main}"
SAM3_REF="${HOOPSENSE_SAM3_REF:-main}"

echo "[INFO] Building experimental Orin SAM3 image"
echo "[INFO] tag=${IMAGE_TAG}"
echo "[INFO] base=${BASE_IMAGE}"
echo "[INFO] dinov3_ref=${DINOV3_REF}"
echo "[INFO] sam3_ref=${SAM3_REF}"

docker build \
  -f "${REPO_ROOT}/Dockerfile.orin.sam3" \
  --build-arg "ORIN_SAM3_BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg "DINOV3_REF=${DINOV3_REF}" \
  --build-arg "SAM3_REF=${SAM3_REF}" \
  -t "${IMAGE_TAG}" \
  "${REPO_ROOT}"

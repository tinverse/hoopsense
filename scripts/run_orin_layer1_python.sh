#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_PYTHON="${REPO_ROOT}/.venv_orin310/bin/python"
VENV_CUDSS_LIB="${REPO_ROOT}/.venv_orin310/lib/hoopsense-cudss"

if [ ! -x "${VENV_PYTHON}" ]; then
  echo "[ERROR] Missing ${VENV_PYTHON}. Run ./scripts/setup_orin.sh first."
  exit 1
fi

mkdir -p "${VENV_CUDSS_LIB}"
for f in "${REPO_ROOT}"/.venv_orin310/lib/python3.10/site-packages/nvidia/cu12/lib/libcudss*.so.0; do
  if [ -f "$f" ]; then
    ln -sf "$(realpath "$f")" "${VENV_CUDSS_LIB}/"
  fi
done

export PYTHONNOUSERSITE=1
unset PYTHONPATH
export LD_LIBRARY_PATH="${VENV_CUDSS_LIB}:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu/nvidia:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO_ROOT}"

exec "${VENV_PYTHON}" "$@"

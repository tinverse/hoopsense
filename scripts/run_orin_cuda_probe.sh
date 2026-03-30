#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONNOUSERSITE=1

run_with_explicit_python() {
  local python_bin="$1"
  local lib_dir="${2:-}"
  local label="$3"
  local guix_shell="${4:-}"

  if [ ! -x "$python_bin" ]; then
    return 1
  fi

  echo "[INFO] Using ${label}: ${python_bin}"
  if [ -n "$lib_dir" ]; then
    export LD_LIBRARY_PATH="${lib_dir}:${LD_LIBRARY_PATH:-}"
    echo "[INFO] Added CUDA runtime libs to LD_LIBRARY_PATH: ${lib_dir}"
  fi
  if [ -n "$guix_shell" ] && [ -x "$guix_shell" ]; then
    export HOOPSENSE_RUST_BRIDGE_WRAPPER="$guix_shell"
    echo "[INFO] Using Guix shell for Cargo-backed checks: ${guix_shell}"
  fi
  "$python_bin" "${REPO_ROOT}/scripts/run_orin_cuda_probe.py"
  return 0
}

if [ -n "${HOOPSENSE_ORIN_PYTHON:-}" ]; then
  run_with_explicit_python "${HOOPSENSE_ORIN_PYTHON}" "${HOOPSENSE_ORIN_LIBDIR:-}" "HOOPSENSE_ORIN_PYTHON override" "${REPO_ROOT}/hoops-orin-shell"
  exit $?
fi

VENV_PYTHON="${REPO_ROOT}/.venv_orin310/bin/python"
VENV_CUDSS_LIB="${REPO_ROOT}/.venv_orin310/lib/hoopsense-cudss"
if [ -x "${VENV_PYTHON}" ]; then
  mkdir -p "${VENV_CUDSS_LIB}"
  for f in "${REPO_ROOT}"/.venv_orin310/lib/python3.10/site-packages/nvidia/cu12/lib/libcudss*.so.0; do
    if [ -f "$f" ]; then
      ln -sf "$(realpath "$f")" "${VENV_CUDSS_LIB}/"
    fi
  done
fi
VENV_CUDA_LIB="${VENV_CUDSS_LIB}:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu/nvidia"
if [ -x "${VENV_PYTHON}" ]; then
  run_with_explicit_python "${VENV_PYTHON}" "${VENV_CUDA_LIB}" "repo-local Orin venv" "${REPO_ROOT}/hoops-orin-shell"
  exit $?
fi

if [ -x "${REPO_ROOT}/hoops-orin-shell" ]; then
  echo "[INFO] Using hoops-orin-shell for GPU-aware validation."
  "${REPO_ROOT}/hoops-orin-shell" python3.10 "${REPO_ROOT}/scripts/run_orin_cuda_probe.py"
else
  echo "[WARN] No explicit Orin runtime found."
  echo "[WARN] Create ${REPO_ROOT}/.venv_orin310 with the Jetson cu126 torch wheel, or set HOOPSENSE_ORIN_PYTHON."
  python3 "${REPO_ROOT}/scripts/run_orin_cuda_probe.py"
fi

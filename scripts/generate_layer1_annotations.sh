#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ "$#" -lt 1 ]; then
  echo "usage: $0 <clip_path> [extra args...]"
  exit 1
fi

exec "${REPO_ROOT}/scripts/run_orin_layer1_python.sh" \
  "${REPO_ROOT}/tools/review/labeller/generate_layer1_annotations.py" \
  "$@" \
  --device cuda:0

#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ "$#" -lt 1 ]; then
  echo "usage: $0 <clip_path> [extra args...]"
  exit 1
fi

args=("$@")
has_bootstrap_backend=0
for arg in "${args[@]}"; do
  if [[ "${arg}" == "--bootstrap-foreground-backend" || "${arg}" == --bootstrap-foreground-backend=* ]]; then
    has_bootstrap_backend=1
    break
  fi
done

grounding_args=()
if [[ "${HOOPSENSE_LAYER1_GROUNDING_DINO:-1}" != "0" && "${has_bootstrap_backend}" == "0" ]]; then
  grounding_args=(
    --bootstrap-foreground-backend grounding_dino
    --bootstrap-foreground-model IDEA-Research/grounding-dino-tiny
    --bootstrap-foreground-prompt "basketball court. basketball hoop. basketball backboard. basketball player. basketball referee."
  )
fi

exec "${REPO_ROOT}/scripts/run_orin_layer1_python.sh" \
  "${REPO_ROOT}/tools/review/labeller/generate_layer1_annotations.py" \
  "${args[@]}" \
  "${grounding_args[@]}" \
  --device cuda:0

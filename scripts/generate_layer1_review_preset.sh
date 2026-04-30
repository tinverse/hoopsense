#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ "$#" -lt 1 ]; then
  echo "usage: $0 <clip_path> [--preset grounded_sam3_review] [--dry-run] [-- extra generator args...]"
  exit 1
fi

exec python3 "${REPO_ROOT}/tools/review/labeller/layer1_review_presets.py" "$@"

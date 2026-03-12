#!/bin/bash
# HoopSense Cloud Training Wrapper
# Goal: Handle GCS Data Sync and Model Export for Vertex AI / GCE

set -e

BUCKET_NAME=${GCS_BUCKET_NAME:-"hoopsense-data"}
DATA_PATH="data/training/oracle_dataset_v3.jsonl"
MODEL_PATH="data/models/action_brain.pt"

echo "[CLOUD] Starting HoopSense Training Pipeline..."

# 1. Pull Data from GCS (if not already baked into image)
if [ ! -f "$DATA_PATH" ]; then
    echo "[CLOUD] Fetching dataset from gs://${BUCKET_NAME}/${DATA_PATH}..."
    gsutil cp gs://${BUCKET_NAME}/${DATA_PATH} ${DATA_PATH} || echo "[WARN] Dataset not found in GCS, falling back to local files."
fi

# 2. Run Training
echo "[CLOUD] Triggering Action Brain Training..."
python3 tools/training/train_action_brain.py "$@"

# 3. Export Model to GCS
if [ -f "$MODEL_PATH" ]; then
    echo "[CLOUD] Exporting trained model to gs://${BUCKET_NAME}/models/action_brain_$(date +%Y%m%d_%H%M%S).pt..."
    gsutil cp ${MODEL_PATH} gs://${BUCKET_NAME}/models/action_brain_$(date +%Y%m%d_%H%M%S).pt
    gsutil cp ${MODEL_PATH} gs://${BUCKET_NAME}/models/action_brain_latest.pt
    echo "[SUCCESS] Model synced to GCS."
else
    echo "[ERROR] Model file not found at ${MODEL_PATH}. Training may have failed."
    exit 1
fi

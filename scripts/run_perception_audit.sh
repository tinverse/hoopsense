#!/bin/bash
# scripts/run_perception_audit.sh
# Validation script for Layer 1 Quality Gate.

set +e # Allow tests to fail so we can capture status

echo "--- HoopSense Perception Audit ---"

# 1. Unit Tests
echo "[STEP 1/3] Running Layer 1 quality gate tests..."
python3 -m unittest tests/test_perception_gate.py
UNIT_TEST_STATUS=$?

# 2. Smoke Test
echo "[STEP 2/3] Running inference pipeline smoke test..."
python3 pipelines/inference.py --smoke-test
SMOKE_TEST_STATUS=$?

# 3. Report Generation
echo "[STEP 3/3] Generating PERCEPTION_READINESS.md..."

REPORT_FILE="PERCEPTION_READINESS.md"
echo "# Perception Readiness Audit" > $REPORT_FILE
echo "" >> $REPORT_FILE
echo "## Quality Gate Status (Layer 1)" >> $REPORT_FILE

if [ $UNIT_TEST_STATUS -eq 0 ]; then
    echo "- **Unit Tests**: PASS" >> $REPORT_FILE
else
    echo "- **Unit Tests**: FAIL (Exit Code: $UNIT_TEST_STATUS)" >> $REPORT_FILE
fi

if [ $SMOKE_TEST_STATUS -eq 0 ]; then
    echo "- **Smoke Test**: PASS" >> $REPORT_FILE
else
    echo "- **Smoke Test**: FAIL (Exit Code: $SMOKE_TEST_STATUS)" >> $REPORT_FILE
fi

echo "" >> $REPORT_FILE
echo "## Technical Summary" >> $REPORT_FILE
echo "- **Model Architecture**: Unified YOLOv8-Pose" >> $REPORT_FILE
echo "- **Audit Date**: $(date)" >> $REPORT_FILE

if [ $UNIT_TEST_STATUS -ne 0 ] || [ $SMOKE_TEST_STATUS -ne 0 ]; then
    echo "--- Audit FAILED. See $REPORT_FILE ---"
    exit 1
else
    echo "--- Audit PASSED. See $REPORT_FILE ---"
    exit 0
fi

#!/usr/bin/env bash
# Test stub for $TRAIN_PYTHON_BIN in train_mac.sh -- captures the argv it was
# invoked with and the env vars train_mac.sh is expected to set, instead of
# actually running training (no real vendor/starcop venv needed).
set -euo pipefail

: "${STUB_CAPTURE_DIR:?STUB_CAPTURE_DIR must be set}"
mkdir -p "$STUB_CAPTURE_DIR"

printf '%s\n' "$@" > "$STUB_CAPTURE_DIR/argv"
{
  echo "PYTORCH_ENABLE_MPS_FALLBACK=${PYTORCH_ENABLE_MPS_FALLBACK:-}"
  echo "MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-}"
  echo "MLFLOW_TRACKING_USERNAME=${MLFLOW_TRACKING_USERNAME:-}"
} > "$STUB_CAPTURE_DIR/env"

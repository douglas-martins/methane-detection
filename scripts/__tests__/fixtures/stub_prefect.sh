#!/usr/bin/env bash
# Test stub for $PREFECT_BIN in prefect_worker_mac.sh -- captures the argv
# it was invoked with and the env vars the script is expected to set,
# instead of actually starting a worker (no real Prefect server needed).
set -euo pipefail

: "${STUB_CAPTURE_DIR:?STUB_CAPTURE_DIR must be set}"
mkdir -p "$STUB_CAPTURE_DIR"

printf '%s\n' "$@" > "$STUB_CAPTURE_DIR/argv"
{
  echo "PREFECT_API_URL=${PREFECT_API_URL:-}"
  echo "PREFECT_API_AUTH_STRING=${PREFECT_API_AUTH_STRING:-}"
} > "$STUB_CAPTURE_DIR/env"

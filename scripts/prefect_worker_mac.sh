#!/usr/bin/env bash
# Starts a native Prefect worker on the M4 Pro, polling the Coolify-hosted
# Prefect server's public API outbound-only -- implements TASK-7.1's D-10
# decision (worker-per-machine, no SSH from the VPS). See
# mlops-methane-detection-plan.md TASK-7.1 and deploy/prefect/README.md for
# the full design and the server-side auth this pairs with.
#
# Usage: ./scripts/prefect_worker_mac.sh [work_pool_name]
#   work_pool_name defaults to mac-mps.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

WORK_POOL_NAME="${1:-mac-mps}"

# Overridable for testing: PREFECT_BIN lets tests stub the CLI without a
# real Environment B venv or real server credentials. ENV_FILE is the
# credentials file to source. Neither is set in normal use.
PREFECT_BIN="${PREFECT_BIN:-.venv/bin/prefect}"
ENV_FILE="${ENV_FILE:-.env.prefect}"

# PREFECT_API_AUTH_STRING (client-side setting, pairs with the server's
# PREFECT_SERVER_API_AUTH_STRING -- see deploy/prefect/README.md's Auth
# section) -- must use `set -a` since .env.prefect has plain VAR=value
# lines with no `export`; a bare `source` would set shell-local vars that
# never reach the `prefect` subprocess.
set -a
# shellcheck disable=SC1091
source "$ENV_FILE"
set +a

# Public, non-secret -- hardcoded here rather than left to .env.prefect so
# this script doesn't silently depend on a value living in a git-ignored
# file that could omit or drift from it (same reasoning as
# scripts/train_mac.sh's hardcoded MLFLOW_TRACKING_URI).
export PREFECT_API_URL="https://methane-detection-prefect.ghostface.tech/api"

if [ -z "${PREFECT_API_AUTH_STRING:-}" ]; then
  echo "prefect_worker_mac.sh: required env var PREFECT_API_AUTH_STRING is not set (check .env.prefect / deploy/prefect/README.md)." >&2
  exit 1
fi

exec "$PREFECT_BIN" worker start \
  --pool "$WORK_POOL_NAME" \
  --type process \
  --create-pool-if-not-found

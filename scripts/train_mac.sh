#!/usr/bin/env bash
# Launches an MLflow-tracked STARCOP training run on the M4 Pro (Apple MPS),
# reproducing TASK-3.2's proven run (71e388fabefd40e892483f552a97efbb) via
# src/training/train.py. See mlops-methane-detection-plan.md TASK-3.3a and
# training-runbook.md for the underlying command this transcribes.
#
# Usage: ./scripts/train_mac.sh [dataset_name] [extra hydra overrides...]
#   dataset_name defaults to starcop_mini.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATASET_NAME="${1:-starcop_mini}"
if [ "$#" -gt 0 ]; then
  shift
fi

# Overridable for testing (scripts/__tests__/train_mac.bats stubs these to
# avoid needing a real vendor/starcop venv or real credentials): PYTHON_BIN
# runs the pure-stdlib launch_profiles calls below (any python3 works, no
# vendor deps); TRAIN_PYTHON_BIN launches the actual training subprocess and
# defaults to the same interpreter for real use. ENV_FILE is the credentials
# file to source. None of these are set in normal use.
PYTHON_BIN="${PYTHON_BIN:-vendor/starcop/.venv/bin/python}"
TRAIN_PYTHON_BIN="${TRAIN_PYTHON_BIN:-$PYTHON_BIN}"
ENV_FILE="${ENV_FILE:-.env.mlflow}"

# Credentials (tracking auth + B2 artifact upload) -- must use `set -a` since
# .env.mlflow has plain VAR=value lines with no `export`; a bare `source`
# would set shell-local vars that never reach the Python subprocess.
set -a
# shellcheck disable=SC1091
source "$ENV_FILE"
set +a

# Public, non-secret -- hardcoded here rather than left to .env.mlflow so
# this script doesn't silently depend on a value living in a git-ignored
# file that could omit or drift from it.
export MLFLOW_TRACKING_URI="https://mlflow.ghostface.tech"

# torch.unique has no MPS kernel in torch 1.13.1 (TASK-3.2 attempt 2) --
# needed on every run, not just once per venv.
export PYTORCH_ENABLE_MPS_FALLBACK=1

REQUIRED_VARS=$("$PYTHON_BIN" -c "
import sys
sys.path.insert(0, 'src/training')
import launch_profiles
print('\n'.join(launch_profiles.required_env_vars('macbook')))
")
while IFS= read -r var; do
  if [ -z "${!var:-}" ]; then
    echo "train_mac.sh: required env var $var is not set (check .env.mlflow / training-runbook.md)." >&2
    exit 1
  fi
done <<< "$REQUIRED_VARS"

LAUNCH_ARGS=()
while IFS= read -r arg; do
  LAUNCH_ARGS+=("$arg")
done < <("$PYTHON_BIN" -c "
import sys
sys.path.insert(0, 'src/training')
import launch_profiles
for arg in launch_profiles.build_launch_args('macbook', sys.argv[1]):
    print(arg)
" "$DATASET_NAME")

exec "$TRAIN_PYTHON_BIN" src/training/train.py "${LAUNCH_ARGS[@]}" "$@"

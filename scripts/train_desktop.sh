#!/usr/bin/env bash
# Launches an MLflow-tracked STARCOP training run on the Desktop's RTX 5070
# via src/training/train.py. See mlops-methane-detection-plan.md TASK-3.1/
# TASK-3.3b: Environment A's stock torch==1.13.1 silently corrupts compute
# on this GPU (Blackwell/sm_120 has no compiled kernels in that build, and
# it does not raise -- see TASK-3.1's 2026-08-23 spike). Environment B
# (root .venv, torch>=2.5) was verified end-to-end on this exact machine
# instead: real forward+backward pass through the actual model-construction
# path, correct (non-corrupted) output, Trainer resolving to CUDAAccelerator
# on cuda:0. This script therefore runs under Environment B, unlike
# train_mac.sh's Environment A -- that is a deliberate, documented deviation,
# not an oversight.
#
# Usage: ./scripts/train_desktop.sh [dataset_name] [extra hydra overrides...]
#   dataset_name defaults to starcop_mini.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATASET_NAME="${1:-starcop_mini}"
if [ "$#" -gt 0 ]; then
  shift
fi

# Overridable for testing (scripts/__tests__/train_desktop.bats stubs these
# to avoid needing a real GPU or real credentials): PYTHON_BIN runs the pure-
# stdlib launch_profiles calls below (any python3 works, no torch/lightning
# deps); TRAIN_PYTHON_BIN launches the actual training subprocess and
# defaults to the same interpreter for real use. ENV_FILE is the credentials
# file to source. None of these are set in normal use.
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
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
export MLFLOW_TRACKING_URI="https://methane-detection-mlflow.ghostface.tech"

# D-09: WandbLogger tries an interactive `wandb.login()` prompt when no
# WANDB_API_KEY is configured, which hangs/fails outright with no TTY (e.g.
# an unattended Prefect flow run) -- `wandb.errors.UsageError: api_key not
# configured (no-tty)`. Default to disabled in that case; an explicit
# WANDB_MODE (e.g. offline) or a real WANDB_API_KEY both take precedence.
# MLflow logging is unaffected either way.
if [ -z "${WANDB_API_KEY:-}" ]; then
  export WANDB_MODE="${WANDB_MODE:-disabled}"
fi

REQUIRED_VARS=$("$PYTHON_BIN" -c "
import sys
sys.path.insert(0, 'src/training')
import launch_profiles
print('\n'.join(launch_profiles.required_env_vars('desktop')))
")
while IFS= read -r var; do
  if [ -z "${!var:-}" ]; then
    echo "train_desktop.sh: required env var $var is not set (check .env.mlflow / internal-docs/runbooks/training.md)." >&2
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
for arg in launch_profiles.build_launch_args('desktop', sys.argv[1]):
    print(arg)
" "$DATASET_NAME")

exec "$TRAIN_PYTHON_BIN" src/training/train.py "${LAUNCH_ARGS[@]}" "$@"

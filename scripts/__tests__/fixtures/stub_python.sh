#!/usr/bin/env bash
# Test stub for $PYTHON_BIN in train_mac.sh -- returns canned output shaped
# like the real launch_profiles.py calls, so train_mac.bats can run in a
# bats-only container with no Python installed. This only tests train_mac.sh's
# own consumption of that output (looping over required vars, building the
# LAUNCH_ARGS array); the real output shape/correctness of
# launch_profiles.required_env_vars/build_launch_args is covered separately
# by src/training/__tests__/test_launch_profiles.py under pytest.
set -euo pipefail

if [ "$#" -eq 2 ]; then
  # required_env_vars('macbook'): "$PYTHON_BIN" -c "<script>"
  cat <<'EOF'
MLFLOW_TRACKING_URI
MLFLOW_TRACKING_USERNAME
MLFLOW_TRACKING_PASSWORD
MLFLOW_S3_ENDPOINT_URL
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
EOF
elif [ "$#" -eq 3 ]; then
  # build_launch_args('macbook', dataset_name): "$PYTHON_BIN" -c "<script>" "$dataset_name"
  dataset_name="$3"
  cat <<EOF
+machine=macbook
+dataset_name=${dataset_name}
training.accelerator=mps
training.devices=1
EOF
else
  echo "stub_python.sh: unexpected argc=$#" >&2
  exit 1
fi

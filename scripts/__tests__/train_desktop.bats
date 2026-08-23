#!/usr/bin/env bats
# Tests for scripts/train_desktop.sh -- the shell orchestration layer only
# (env sourcing, pre-flight credential check, arg plumbing, delegating to
# the training subprocess). The two `python -c` calls it makes are stubbed
# with fixtures/stub_python_desktop.sh (canned output, no python needed in
# this bats-only container) so these tests don't depend on and don't
# re-validate launch_profiles.py's own logic -- that's covered by
# src/training/__tests__/test_launch_profiles.py under pytest. The actual
# training invocation is stubbed with fixtures/stub_train_python.sh, which
# captures argv/env instead of running real training.
#
# Run via: docker run --rm -v "$PWD":/code -w /code bats/bats:latest \
#   scripts/__tests__/train_desktop.bats

load '/usr/lib/bats/bats-support/load'
load '/usr/lib/bats/bats-assert/load'

setup() {
  REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
  SCRIPT="${REPO_ROOT}/scripts/train_desktop.sh"
  FIXTURES="${BATS_TEST_DIRNAME}/fixtures"

  export PYTHON_BIN="${FIXTURES}/stub_python_desktop.sh"
  export TRAIN_PYTHON_BIN="${FIXTURES}/stub_train_python.sh"
  export STUB_CAPTURE_DIR="${BATS_TEST_TMPDIR}/capture"
  export ENV_FILE="${FIXTURES}/valid.env.mlflow"
}

@test "fails with a clear message when a required credential is missing" {
  export ENV_FILE="${FIXTURES}/missing_aws_secret.env.mlflow"

  run "$SCRIPT"

  assert_failure
  assert_output --partial "AWS_SECRET_ACCESS_KEY"
  assert_output --partial "not set"
}

@test "does not invoke the training subprocess when a required credential is missing" {
  export ENV_FILE="${FIXTURES}/missing_aws_secret.env.mlflow"

  run "$SCRIPT"

  assert [ ! -f "${STUB_CAPTURE_DIR}/argv" ]
}

@test "defaults the dataset name to starcop_mini when none is given" {
  run "$SCRIPT"

  assert_success
  run cat "${STUB_CAPTURE_DIR}/argv"
  assert_output --partial "+dataset_name=starcop_mini"
}

@test "passes a custom dataset name through" {
  run "$SCRIPT" starcop_raw

  assert_success
  run cat "${STUB_CAPTURE_DIR}/argv"
  assert_output --partial "+dataset_name=starcop_raw"
}

@test "passes extra hydra overrides through after the dataset name" {
  run "$SCRIPT" starcop_mini training.max_epochs=1

  assert_success
  run cat "${STUB_CAPTURE_DIR}/argv"
  assert_output --partial "training.max_epochs=1"
}

@test "requests the gpu accelerator for the desktop machine profile" {
  run "$SCRIPT"

  assert_success
  run cat "${STUB_CAPTURE_DIR}/argv"
  assert_output --partial "training.accelerator=gpu"
}

@test "hardcodes MLFLOW_TRACKING_URI regardless of what the credentials file sets" {
  run "$SCRIPT"

  assert_success
  run cat "${STUB_CAPTURE_DIR}/env"
  assert_output --partial "MLFLOW_TRACKING_URI=https://methane-detection-mlflow.ghostface.tech"
  refute_output --partial "wrong-url.example.com"
}

@test "sourced credentials reach the training subprocess environment" {
  run "$SCRIPT"

  assert_success
  run cat "${STUB_CAPTURE_DIR}/env"
  assert_output --partial "MLFLOW_TRACKING_USERNAME=test-user"
}

@test "invokes the training subprocess with train.py as the first argument" {
  run "$SCRIPT"

  assert_success
  run head -n 1 "${STUB_CAPTURE_DIR}/argv"
  assert_output "src/training/train.py"
}

@test "defaults WANDB_MODE=disabled when WANDB_API_KEY is not set" {
  unset WANDB_API_KEY WANDB_MODE || true

  run "$SCRIPT"

  assert_success
  run cat "${STUB_CAPTURE_DIR}/env"
  assert_output --partial "WANDB_MODE=disabled"
}

@test "does not force WANDB_MODE when WANDB_API_KEY is set" {
  unset WANDB_MODE || true
  export WANDB_API_KEY="fake-key"

  run "$SCRIPT"

  assert_success
  run cat "${STUB_CAPTURE_DIR}/env"
  assert_output --partial "WANDB_MODE="
  refute_output --partial "WANDB_MODE=disabled"
}

@test "respects an explicitly set WANDB_MODE even without WANDB_API_KEY" {
  unset WANDB_API_KEY || true
  export WANDB_MODE="offline"

  run "$SCRIPT"

  assert_success
  run cat "${STUB_CAPTURE_DIR}/env"
  assert_output --partial "WANDB_MODE=offline"
}

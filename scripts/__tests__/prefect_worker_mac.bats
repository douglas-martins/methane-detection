#!/usr/bin/env bats
# Tests for scripts/prefect_worker_mac.sh -- the shell orchestration layer
# only (env sourcing, pre-flight auth check, arg plumbing to `prefect
# worker start`). The `prefect` CLI itself is stubbed with
# fixtures/stub_prefect.sh (captures argv/env instead of starting a real
# worker against a real server) so these tests don't need a live Prefect
# server or Environment B venv.
#
# Run via: docker run --rm -v "$PWD":/code -w /code bats/bats:latest \
#   scripts/__tests__/prefect_worker_mac.bats

load '/usr/lib/bats/bats-support/load'
load '/usr/lib/bats/bats-assert/load'

setup() {
  # Guard against a PREFECT_API_AUTH_STRING already exported in the
  # invoking shell (e.g. a developer who sourced .env.prefect for real
  # manual use, per deploy/prefect/README.md, running bats in that same
  # session) -- the script's `source "$ENV_FILE"` only sets vars the file
  # itself defines, it can't unset an inherited one, so the missing-auth
  # tests below would silently see a non-empty value and falsely pass
  # through the pre-flight check.
  unset PREFECT_API_AUTH_STRING

  REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
  SCRIPT="${REPO_ROOT}/scripts/prefect_worker_mac.sh"
  FIXTURES="${BATS_TEST_DIRNAME}/fixtures"

  export PREFECT_BIN="${FIXTURES}/stub_prefect.sh"
  export STUB_CAPTURE_DIR="${BATS_TEST_TMPDIR}/capture"
  export ENV_FILE="${FIXTURES}/valid.env.prefect"
}

@test "fails with a clear message when PREFECT_API_AUTH_STRING is missing" {
  export ENV_FILE="${FIXTURES}/missing_auth.env.prefect"

  run "$SCRIPT"

  assert_failure
  assert_output --partial "PREFECT_API_AUTH_STRING"
  assert_output --partial "not set"
}

@test "does not invoke the prefect CLI when auth is missing" {
  export ENV_FILE="${FIXTURES}/missing_auth.env.prefect"

  run "$SCRIPT"

  assert [ ! -f "${STUB_CAPTURE_DIR}/argv" ]
}

@test "defaults the work pool name to mac-mps when none is given" {
  run "$SCRIPT"

  assert_success
  run cat "${STUB_CAPTURE_DIR}/argv"
  assert_output --partial "mac-mps"
}

@test "passes a custom work pool name through" {
  run "$SCRIPT" desktop-rtx5070

  assert_success
  run cat "${STUB_CAPTURE_DIR}/argv"
  assert_output --partial "desktop-rtx5070"
  refute_output --partial "mac-mps"
}

@test "starts a process-type worker" {
  run "$SCRIPT"

  assert_success
  run cat "${STUB_CAPTURE_DIR}/argv"
  assert_output --partial "process"
}

@test "hardcodes PREFECT_API_URL regardless of what the credentials file sets" {
  run "$SCRIPT"

  assert_success
  run cat "${STUB_CAPTURE_DIR}/env"
  assert_output --partial "PREFECT_API_URL=https://methane-detection-prefect.ghostface.tech/api"
}

@test "sourced auth string reaches the prefect subprocess environment" {
  run "$SCRIPT"

  assert_success
  run cat "${STUB_CAPTURE_DIR}/env"
  assert_output --partial "PREFECT_API_AUTH_STRING=test-user:test-pass"
}

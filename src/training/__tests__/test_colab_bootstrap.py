"""Tests for src/training/colab_bootstrap.py -- pure helpers for
notebooks/train_colab.ipynb (Test Size: Small, no mocking, no `google.colab`
import): building the DVC service-account setup commands, the Colab Secret
names the notebook needs, and D-09's WANDB_MODE default -- see
mlops-methane-detection-plan.md TASK-3.3c.
"""

import colab_bootstrap
import pytest


class TestDvcServiceAccountSetupCommands:
    def test_sets_expected_flags_for_given_key_path(self):
        commands = colab_bootstrap.dvc_service_account_setup_commands(
            "/content/gdrive-service-account.json"
        )

        assert [
            "dvc",
            "remote",
            "modify",
            "--local",
            "gdrive",
            "gdrive_use_service_account",
            "true",
        ] in commands
        assert [
            "dvc",
            "remote",
            "modify",
            "--local",
            "gdrive",
            "gdrive_service_account_json_file_path",
            "/content/gdrive-service-account.json",
        ] in commands

    def test_raises_value_error_on_empty_path(self):
        with pytest.raises(ValueError, match="service_account_json_path"):
            colab_bootstrap.dvc_service_account_setup_commands("")


class TestRequiredColabSecrets:
    def test_includes_launch_profiles_vars_plus_dvc_service_account_json(self):
        secrets = colab_bootstrap.required_colab_secrets()

        assert "MLFLOW_TRACKING_USERNAME" in secrets
        assert "MLFLOW_TRACKING_PASSWORD" in secrets
        assert "MLFLOW_S3_ENDPOINT_URL" in secrets
        assert "AWS_ACCESS_KEY_ID" in secrets
        assert "AWS_SECRET_ACCESS_KEY" in secrets
        assert "DVC_GDRIVE_SERVICE_ACCOUNT_JSON" in secrets

    def test_excludes_mlflow_tracking_uri(self):
        # The notebook hardcodes MLFLOW_TRACKING_URI unconditionally (same
        # reasoning as train_mac.sh/train_desktop.sh) -- requiring the user to
        # also supply it as a Secret/env var would demand a value that's
        # discarded regardless, and raise a confusing error if they don't.
        secrets = colab_bootstrap.required_colab_secrets()

        assert "MLFLOW_TRACKING_URI" not in secrets


class TestReadSecret:
    def test_prefers_userdata_when_it_returns_a_value(self):
        value = colab_bootstrap.read_secret(
            "MLFLOW_TRACKING_URI",
            userdata_get=lambda name: "from-userdata",
            environ={"MLFLOW_TRACKING_URI": "from-environ"},
        )

        assert value == "from-userdata"

    def test_falls_back_to_environ_when_userdata_get_is_none(self):
        # Simulates `google.colab` not being importable at all -- e.g. a
        # local/non-Colab context, not just a VS Code-attached kernel.
        value = colab_bootstrap.read_secret(
            "MLFLOW_TRACKING_URI",
            userdata_get=None,
            environ={"MLFLOW_TRACKING_URI": "from-environ"},
        )

        assert value == "from-environ"

    def test_falls_back_to_environ_when_userdata_get_raises(self):
        # Simulates a VS Code-attached Colab kernel with no browser frontend
        # to service the Secrets RPC -- google.colab.userdata.get() raises
        # instead of returning a value.
        def raising_userdata_get(name):
            raise RuntimeError("no frontend attached")

        value = colab_bootstrap.read_secret(
            "MLFLOW_TRACKING_URI",
            userdata_get=raising_userdata_get,
            environ={"MLFLOW_TRACKING_URI": "from-environ"},
        )

        assert value == "from-environ"

    def test_falls_back_to_environ_when_userdata_returns_empty_string(self):
        value = colab_bootstrap.read_secret(
            "MLFLOW_TRACKING_URI",
            userdata_get=lambda name: "",
            environ={"MLFLOW_TRACKING_URI": "from-environ"},
        )

        assert value == "from-environ"

    def test_returns_none_when_neither_source_has_the_value(self):
        value = colab_bootstrap.read_secret("MISSING", userdata_get=None, environ={})

        assert value is None


class TestResolveWandbMode:
    def test_defaults_to_disabled_when_no_api_key(self):
        mode = colab_bootstrap.resolve_wandb_mode(wandb_api_key=None, requested_mode=None)

        assert mode == "disabled"

    def test_respects_explicit_override_even_without_api_key(self):
        mode = colab_bootstrap.resolve_wandb_mode(wandb_api_key=None, requested_mode="offline")

        assert mode == "offline"

    def test_does_not_override_when_api_key_is_present(self):
        mode = colab_bootstrap.resolve_wandb_mode(wandb_api_key="real-key", requested_mode=None)

        assert mode is None

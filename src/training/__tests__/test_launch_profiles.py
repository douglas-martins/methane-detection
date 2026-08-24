"""Tests for src/training/launch_profiles.py -- pure arg-building logic
(Test Size: Small, no mocking): builds the Hydra CLI arg list and required
credential env vars for a per-machine training launch script, so
scripts/train_mac.sh (and later train_desktop.sh/train_colab.ipynb) can stay
thin, untested glue instead of duplicating this decision logic in shell --
see mlops-methane-detection-plan.md TASK-3.3a.
"""

import launch_profiles
import pytest


class TestBuildLaunchArgs:
    def test_build_launch_args_for_macbook_includes_mps_accelerator_and_devices(self):
        args = launch_profiles.build_launch_args("macbook", "starcop_mini")

        assert "+machine=macbook" in args
        assert "+dataset_name=starcop_mini" in args
        assert "training.accelerator=mps" in args
        assert "training.devices=1" in args

    def test_build_launch_args_overrides_replace_machine_defaults_without_duplicating_flags(self):
        args = launch_profiles.build_launch_args(
            "macbook", "starcop_mini", overrides={"training.accelerator": "cpu"}
        )

        assert "training.accelerator=cpu" in args
        assert "training.accelerator=mps" not in args
        assert sum(arg.startswith("training.accelerator=") for arg in args) == 1

    def test_build_launch_args_rejects_unknown_machine(self):
        with pytest.raises(ValueError, match="raspberry-pi"):
            launch_profiles.build_launch_args("raspberry-pi", "starcop_mini")

    def test_build_launch_args_for_desktop_includes_gpu_accelerator(self):
        args = launch_profiles.build_launch_args("desktop", "starcop_mini")

        assert "+machine=desktop" in args
        assert "+dataset_name=starcop_mini" in args
        assert "training.accelerator=gpu" in args
        assert "training.devices=1" in args

    def test_build_launch_args_for_colab_includes_gpu_accelerator(self):
        args = launch_profiles.build_launch_args("colab", "starcop_mini")

        assert "+machine=colab" in args
        assert "+dataset_name=starcop_mini" in args
        assert "training.accelerator=gpu" in args
        assert "training.devices=1" in args


class TestRequiredEnvVars:
    def test_required_env_vars_includes_tracking_and_artifact_credentials(self):
        env_vars = launch_profiles.required_env_vars("macbook")

        assert "MLFLOW_TRACKING_URI" in env_vars
        assert "MLFLOW_TRACKING_USERNAME" in env_vars
        assert "MLFLOW_TRACKING_PASSWORD" in env_vars
        assert "MLFLOW_S3_ENDPOINT_URL" in env_vars
        assert "AWS_ACCESS_KEY_ID" in env_vars
        assert "AWS_SECRET_ACCESS_KEY" in env_vars

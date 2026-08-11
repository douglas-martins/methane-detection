"""Tests for src/training/mlflow_utils.py -- pure functions, no MLflow SDK
calls, no live server needed (Test Size: Small)."""

import pytest
from omegaconf import OmegaConf

import mlflow_utils


class TestRequireMlflowTrackingEnv:
    def test_passes_when_all_vars_set(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.com")
        monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "user")
        monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "pass")

        mlflow_utils.require_mlflow_tracking_env()

    def test_raises_when_uri_missing(self, monkeypatch):
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "user")
        monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "pass")

        with pytest.raises(RuntimeError, match="MLFLOW_TRACKING_URI"):
            mlflow_utils.require_mlflow_tracking_env()

    def test_raises_listing_all_missing_vars(self, monkeypatch):
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        monkeypatch.delenv("MLFLOW_TRACKING_USERNAME", raising=False)
        monkeypatch.delenv("MLFLOW_TRACKING_PASSWORD", raising=False)

        with pytest.raises(RuntimeError) as exc_info:
            mlflow_utils.require_mlflow_tracking_env()

        assert "MLFLOW_TRACKING_URI" in str(exc_info.value)
        assert "MLFLOW_TRACKING_USERNAME" in str(exc_info.value)
        assert "MLFLOW_TRACKING_PASSWORD" in str(exc_info.value)


class TestBuildRunTags:
    def test_returns_expected_keys_and_values(self):
        tags = mlflow_utils.build_run_tags(
            dataset_version="abc123.dir",
            dataset_dirty=False,
            machine="desktop",
            sensor="AVIRIS-NG",
        )

        assert tags == {
            "dataset_version": "abc123.dir",
            "dataset_dirty": "False",
            "machine": "desktop",
            "sensor": "AVIRIS-NG",
        }

    def test_stringifies_dataset_dirty_true(self):
        tags = mlflow_utils.build_run_tags(
            dataset_version="abc123.dir",
            dataset_dirty=True,
            machine="macbook",
            sensor="AVIRIS-NG",
        )

        assert tags["dataset_dirty"] == "True"


class TestFlattenHydraParams:
    def test_produces_dot_separated_keys_for_nested_config(self):
        settings = OmegaConf.create({"model": {"lr": 0.0001, "optimizer": "adam"}})

        flat = mlflow_utils.flatten_hydra_params(settings)

        assert flat["model.lr"] == "0.0001"
        assert flat["model.optimizer"] == "adam"

    def test_stringifies_list_valued_params(self):
        settings = OmegaConf.create(
            {"dataset": {"input_products": ["mag1c", "TOA_AVIRIS_640nm"]}}
        )

        flat = mlflow_utils.flatten_hydra_params(settings)

        assert flat["dataset.input_products"] == "['mag1c', 'TOA_AVIRIS_640nm']"

    def test_flat_top_level_key_has_no_leading_dot(self):
        settings = OmegaConf.create({"seed": 42})

        flat = mlflow_utils.flatten_hydra_params(settings)

        assert flat == {"seed": "42"}

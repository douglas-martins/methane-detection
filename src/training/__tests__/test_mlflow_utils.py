"""Tests for src/training/mlflow_utils.py -- pure functions, no MLflow SDK
calls, no live server needed (Test Size: Small)."""

from omegaconf import OmegaConf

import mlflow_utils


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

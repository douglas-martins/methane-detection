"""Tests for src/training/settings_overlay.py -- pure OmegaConf logic (Test
Size: Small, real objects, no mocking): merges configs/training/overlay.yaml
on top of STARCOP's Hydra-derived settings and validates the required
`machine` field, without ever touching vendor/starcop/scripts/configs/config.yaml.
"""

import pytest
import settings_overlay
from omegaconf import OmegaConf

VALID_MACHINES = ("desktop", "macbook", "colab")


class TestMergeOverlay:
    def test_overlay_defaults_survive_when_not_overridden(self, tmp_path):
        overlay_path = tmp_path / "overlay.yaml"
        overlay_path.write_text("machine: ???\ndataset:\n  augmentations:\n    rotation_p: 0.5\n")
        hydra_settings = OmegaConf.create(
            {"machine": "desktop", "dataset": {"input_products": ["mag1c"]}}
        )

        merged = settings_overlay.merge_overlay(hydra_settings, overlay_path)

        assert merged.dataset.augmentations.rotation_p == 0.5
        assert merged.dataset.input_products == ["mag1c"]

    def test_cli_provided_machine_wins_over_overlay_placeholder(self, tmp_path):
        overlay_path = tmp_path / "overlay.yaml"
        overlay_path.write_text("machine: ???\n")
        hydra_settings = OmegaConf.create({"machine": "macbook"})

        merged = settings_overlay.merge_overlay(hydra_settings, overlay_path)

        assert merged.machine == "macbook"


class TestValidateMachine:
    @pytest.mark.parametrize("machine", VALID_MACHINES)
    def test_accepts_valid_machine_values(self, machine):
        settings = OmegaConf.create({"machine": machine})

        settings_overlay.validate_machine(settings)  # should not raise

    def test_raises_when_machine_is_still_the_missing_sentinel(self):
        settings = OmegaConf.create({"machine": "???"})

        with pytest.raises(ValueError, match="machine"):
            settings_overlay.validate_machine(settings)

    def test_raises_when_machine_is_not_one_of_the_allowed_values(self):
        settings = OmegaConf.create({"machine": "raspberry-pi"})

        with pytest.raises(ValueError, match="raspberry-pi"):
            settings_overlay.validate_machine(settings)


class TestValidateDatasetName:
    @pytest.mark.parametrize("dataset_name", ("starcop_mini", "starcop_raw"))
    def test_accepts_valid_dataset_names(self, dataset_name):
        settings = OmegaConf.create({"dataset_name": dataset_name})

        settings_overlay.validate_dataset_name(settings)  # should not raise

    def test_raises_when_dataset_name_is_still_the_missing_sentinel(self):
        settings = OmegaConf.create({"dataset_name": "???"})

        with pytest.raises(ValueError, match="dataset_name"):
            settings_overlay.validate_dataset_name(settings)

    def test_raises_when_dataset_name_is_not_recognized(self):
        settings = OmegaConf.create({"dataset_name": "starcop_huge"})

        with pytest.raises(ValueError, match="starcop_huge"):
            settings_overlay.validate_dataset_name(settings)

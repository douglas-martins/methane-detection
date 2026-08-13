"""Tests for src/registry/hf_baseline_import.py -- pure functions only (no
network, no MLflow SDK, no live server). Test Size: Small, except the
model-class dispatch tests, which import the real (unmodified)
starcop.models classes via the _vendor_starcop_baseline seam -- Test Size:
Medium (real import, no I/O).
"""

import hf_baseline_import
import pytest
from _vendor_starcop_baseline import ModelModule, ModelModuleRegression


class TestVariantSubfolder:
    def test_mag1c_only_maps_to_its_hf_subfolder(self):
        assert hf_baseline_import.variant_subfolder("mag1c_only") == "hyperstarcop_mag1c_only"

    def test_mag1c_rgb_maps_to_its_hf_subfolder(self):
        assert hf_baseline_import.variant_subfolder("mag1c_rgb") == "hyperstarcop_mag1c_rgb"

    def test_unknown_variant_raises_value_error(self):
        with pytest.raises(ValueError, match="unknown_variant"):
            hf_baseline_import.variant_subfolder("unknown_variant")


class TestRegistryModelName:
    def test_mag1c_only_becomes_kebab_case_registry_name(self):
        assert hf_baseline_import.registry_model_name("mag1c_only") == "starcop-baseline-mag1c-only"

    def test_mag1c_rgb_becomes_kebab_case_registry_name(self):
        assert hf_baseline_import.registry_model_name("mag1c_rgb") == "starcop-baseline-mag1c-rgb"


class TestModelClassForMode:
    def test_segmentation_output_resolves_to_model_module(self):
        assert hf_baseline_import.model_class_for_mode("segmentation_output") is ModelModule

    def test_regression_output_resolves_to_model_module_regression(self):
        assert hf_baseline_import.model_class_for_mode("regression_output") is ModelModuleRegression

    def test_unknown_mode_raises_value_error(self):
        with pytest.raises(ValueError, match="unknown_mode"):
            hf_baseline_import.model_class_for_mode("unknown_mode")

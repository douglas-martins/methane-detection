"""Tests for src/training/validation_metrics.py -- pure functions, no MLflow
SDK calls, no live server needed (Test Size: Small)."""

import numpy as np
import torch

import validation_metrics


class TestExtractScalarMetrics:
    def test_prefixes_scalar_float_values(self):
        metrics = {"accuracy": 0.9, "f1score": 0.75}

        result = validation_metrics.extract_scalar_metrics(metrics, prefix="test")

        assert result == {"test_accuracy": 0.9, "test_f1score": 0.75}

    def test_includes_numpy_scalar_values(self):
        metrics = {"frac_total_easy": np.float64(0.42)}

        result = validation_metrics.extract_scalar_metrics(metrics, prefix="test")

        assert result == {"test_frac_total_easy": 0.42}

    def test_excludes_tensor_values(self):
        metrics = {
            "accuracy": 0.9,
            "confusion_matrix": torch.tensor([[1, 2], [3, 4]]),
            "classification_confusion_matrix": torch.tensor([[1, 0], [0, 1]]),
        }

        result = validation_metrics.extract_scalar_metrics(metrics, prefix="test")

        assert result == {"test_accuracy": 0.9}

    def test_excludes_list_and_dict_values(self):
        metrics = {
            "accuracy": 0.9,
            "thresholded": [{"threshold": 0.5, "precision": 0.8}],
        }

        result = validation_metrics.extract_scalar_metrics(metrics, prefix="test")

        assert result == {"test_accuracy": 0.9}

    def test_returns_empty_dict_for_no_scalar_metrics(self):
        metrics = {"confusion_matrix": torch.tensor([[1, 2], [3, 4]])}

        result = validation_metrics.extract_scalar_metrics(metrics, prefix="test")

        assert result == {}

    def test_result_values_are_plain_python_floats(self):
        metrics = {"accuracy": np.float64(0.9)}

        result = validation_metrics.extract_scalar_metrics(metrics, prefix="test")

        assert type(result["test_accuracy"]) is float

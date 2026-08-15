"""Tests for src/serving/inference.py -- pure tensor-assembly and inference
logic, no model loading, no MLflow, no network (Test Size: Small).

The mask/probability recipe mirrors vendor/starcop's own
ModelModule.batch_with_preds (model_module.py:191-208): sigmoid, then
threshold at 0.5. run_inference is exercised against a real (not mocked)
minimal torch.nn.Module whose forward() returns fixed, known logits, so the
sigmoid/threshold arithmetic is verified against real tensor operations.
"""

import inference
import numpy as np
import pytest
import torch
from torch import nn


class TestAssembleInputTensor:
    def test_transposes_a_channel_last_array_to_channel_first(self):
        array = np.zeros((8, 8, 4), dtype=np.float32)

        tensor = inference.assemble_input_tensor(array, expected_channels=4)

        assert tensor.shape == (1, 4, 8, 8)

    def test_passes_through_a_channel_first_array_unchanged_in_shape(self):
        array = np.zeros((4, 8, 8), dtype=np.float32)

        tensor = inference.assemble_input_tensor(array, expected_channels=4)

        assert tensor.shape == (1, 4, 8, 8)

    def test_preserves_values_when_transposing_channel_last_input(self):
        array = np.arange(2 * 3 * 4, dtype=np.float32).reshape(3, 4, 2)

        tensor = inference.assemble_input_tensor(array, expected_channels=2)

        assert tensor[0, 0, 0, 0].item() == array[0, 0, 0]
        assert tensor[0, 1, 2, 3].item() == array[2, 3, 1]

    def test_converts_integer_input_to_float32(self):
        array = np.ones((8, 8, 4), dtype=np.int16)

        tensor = inference.assemble_input_tensor(array, expected_channels=4)

        assert tensor.dtype == torch.float32

    def test_raises_when_neither_axis_matches_expected_channels(self):
        array = np.zeros((8, 8, 3), dtype=np.float32)

        with pytest.raises(ValueError, match="4"):
            inference.assemble_input_tensor(array, expected_channels=4)

    def test_raises_on_a_non_3d_array(self):
        array = np.zeros((8, 8), dtype=np.float32)

        with pytest.raises(ValueError, match="3D|shape"):
            inference.assemble_input_tensor(array, expected_channels=4)


class _FixedLogitsModel(nn.Module):
    """A real (not mocked) minimal module returning a pre-set logits tensor,
    regardless of input -- lets run_inference's sigmoid/threshold arithmetic
    be checked against known values without needing a real trained model."""

    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self._logits = logits

    def forward(self, x):
        return self._logits


class TestRunInference:
    def test_thresholds_positive_logits_as_plume_present(self):
        # sigmoid(5.0) ~ 0.993, well above the 0.5 threshold
        logits = torch.full((1, 1, 2, 2), 5.0)
        model = _FixedLogitsModel(logits)
        x = torch.zeros((1, 1, 2, 2))

        mask, probs = inference.run_inference(model, x)

        assert mask.shape == (2, 2)
        assert (mask == 1).all()
        assert (probs > 0.5).all()

    def test_thresholds_negative_logits_as_no_plume(self):
        # sigmoid(-5.0) ~ 0.007, well below the 0.5 threshold
        logits = torch.full((1, 1, 2, 2), -5.0)
        model = _FixedLogitsModel(logits)
        x = torch.zeros((1, 1, 2, 2))

        mask, probs = inference.run_inference(model, x)

        assert (mask == 0).all()
        assert (probs < 0.5).all()

    def test_probs_matches_manual_sigmoid_of_the_logits(self):
        logits = torch.tensor([[[[0.0, 2.0], [-2.0, 4.0]]]])
        model = _FixedLogitsModel(logits)
        x = torch.zeros((1, 1, 2, 2))

        _, probs = inference.run_inference(model, x)

        expected = torch.sigmoid(logits).squeeze(0).squeeze(0).numpy()
        assert np.allclose(probs, expected)

    def test_returns_plain_numpy_arrays_not_tensors(self):
        logits = torch.zeros((1, 1, 2, 2))
        model = _FixedLogitsModel(logits)
        x = torch.zeros((1, 1, 2, 2))

        mask, probs = inference.run_inference(model, x)

        assert isinstance(mask, np.ndarray)
        assert isinstance(probs, np.ndarray)


class TestPredictResponse:
    """predict_response is the full assemble -> infer -> shape-as-JSON
    pipeline behind POST /predict's response body -- pulled out of
    service.py (BentoML SDK glue, not unit tested) so this logic is
    unit tested directly instead of only via a live `bentoml serve` + curl
    run. Same real-fixed-logits-model approach as TestRunInference, no
    mocking.
    """

    def test_returns_json_ready_mask_and_confidence_lists(self):
        logits = torch.full((1, 1, 2, 2), 5.0)
        model = _FixedLogitsModel(logits)
        array = np.zeros((2, 2, 1), dtype=np.float32)

        response = inference.predict_response(model, array, expected_channels=1)

        assert response["mask"] == [[1, 1], [1, 1]]
        assert np.allclose(response["confidence"], [[0.9933071, 0.9933071]] * 2, atol=1e-6)

    def test_response_values_are_plain_python_lists_not_numpy_or_tensors(self):
        logits = torch.zeros((1, 1, 2, 2))
        model = _FixedLogitsModel(logits)
        array = np.zeros((2, 2, 1), dtype=np.float32)

        response = inference.predict_response(model, array, expected_channels=1)

        assert isinstance(response["mask"], list)
        assert isinstance(response["mask"][0], list)
        assert isinstance(response["mask"][0][0], int)
        assert isinstance(response["confidence"][0][0], float)

    def test_propagates_the_channel_mismatch_error_from_assemble_input_tensor(self):
        model = _FixedLogitsModel(torch.zeros((1, 1, 2, 2)))
        array = np.zeros((2, 2, 3), dtype=np.float32)

        with pytest.raises(ValueError, match="4"):
            inference.predict_response(model, array, expected_channels=4)

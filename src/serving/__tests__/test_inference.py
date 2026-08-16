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

    def test_raises_on_string_dtype_instead_of_letting_torch_raise_typeerror(self):
        # torch.from_numpy raises a raw TypeError (not ValueError) for
        # non-numeric dtypes -- service.py only catches ValueError to
        # translate into an HTTP 400, so an uncaught TypeError here would
        # surface as an opaque 500 instead of a client error.
        array = np.full((8, 8, 4), "x", dtype="<U1")

        with pytest.raises(ValueError, match="dtype"):
            inference.assemble_input_tensor(array, expected_channels=4)

    def test_raises_on_structured_dtype(self):
        array = np.zeros((8, 8, 4), dtype=[("a", "f4")])

        with pytest.raises(ValueError, match="dtype"):
            inference.assemble_input_tensor(array, expected_channels=4)

    def test_converts_longdouble_to_float32_instead_of_letting_torch_raise(self):
        # np.longdouble passes np.issubdtype(..., np.number) but
        # torch.from_numpy still rejects it directly -- a numpy-side
        # .astype(float32) before tensor creation sidesteps this, rather
        # than a naive "is it numeric" dtype check that would still crash.
        array = np.ones((8, 8, 4), dtype=np.longdouble)

        tensor = inference.assemble_input_tensor(array, expected_channels=4)

        assert tensor.dtype == torch.float32

    def test_raises_on_complex_dtype_instead_of_silently_discarding_the_imaginary_part(self):
        # complex64/complex128 pass np.issubdtype(..., np.number) (complex
        # IS numeric) but .astype(float32) on a complex array doesn't raise
        # -- it silently drops the imaginary component (only a
        # ComplexWarning, not an exception), which would corrupt input data
        # without any visible error.
        array = np.full((8, 8, 4), 1 + 2j, dtype=np.complex64)

        with pytest.raises(ValueError, match="complex"):
            inference.assemble_input_tensor(array, expected_channels=4)

    def test_raises_when_both_first_and_last_axis_match_expected_channels(self):
        # (4, 8, 4) with expected_channels=4: shape[0] == shape[-1] == 4, so
        # channel-last vs. channel-first can't be distinguished -- silently
        # picking one (as the old if/elif did) risks transposing H and W.
        array = np.zeros((4, 8, 4), dtype=np.float32)

        with pytest.raises(ValueError, match="ambiguous"):
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


class TestPerBandMeans:
    """per_band_means backs TASK-6.2's rolling drift-detection window
    (src/serving/service.py) -- each request contributes one mean-per-band
    value, later aggregated across the last N requests by
    drift.update_rolling_stats. Reuses assemble_input_tensor's
    channel-axis detection, so values are on the same raw (non-normalized)
    scale docs/dataset_report.md's baseline stats report.
    """

    def test_returns_one_mean_per_channel_in_channel_order(self):
        array = np.zeros((3, 3, 2), dtype=np.float32)
        array[:, :, 0] = 10.0
        array[:, :, 1] = 20.0

        means = inference.per_band_means(array, expected_channels=2)

        assert means == [10.0, 20.0]

    def test_averages_over_the_spatial_dimensions_only(self):
        array = np.array([[[1.0], [3.0]], [[5.0], [7.0]]], dtype=np.float32)

        means = inference.per_band_means(array, expected_channels=1)

        assert means == [4.0]

    def test_works_on_a_channel_first_array_too(self):
        array = np.zeros((2, 3, 3), dtype=np.float32)
        array[0, :, :] = 10.0
        array[1, :, :] = 20.0

        means = inference.per_band_means(array, expected_channels=2)

        assert means == [10.0, 20.0]

    def test_propagates_the_channel_mismatch_error_from_assemble_input_tensor(self):
        array = np.zeros((2, 2, 3), dtype=np.float32)

        with pytest.raises(ValueError, match="4"):
            inference.per_band_means(array, expected_channels=4)

    def test_returns_plain_python_floats_not_numpy_or_tensor_scalars(self):
        array = np.zeros((2, 2, 1), dtype=np.float32)

        means = inference.per_band_means(array, expected_channels=1)

        assert isinstance(means[0], float)


class TestHasPlume:
    """has_plume backs TASK-6.1's methane_prediction_total Prometheus
    counter (src/serving/service.py) -- BentoML's own built-in metrics have
    no visibility into what a response actually predicted, only request
    count/latency/status code, so this is the signal that turns a /predict
    response into a "plume_detected" vs "no_plume" label.
    """

    def test_true_when_any_pixel_is_positive(self):
        mask = np.array([[0, 0], [0, 1]])

        assert inference.has_plume(mask) is True

    def test_false_when_all_pixels_are_zero(self):
        mask = np.zeros((4, 4), dtype=int)

        assert inference.has_plume(mask) is False

    def test_accepts_a_plain_nested_list_not_just_a_numpy_array(self):
        # service.py calls this on predict_response's JSON-ready "mask"
        # field (a plain list of lists), not the raw numpy array -- must not
        # require np.ndarray specifically.
        assert inference.has_plume([[0, 0], [1, 0]]) is True
        assert inference.has_plume([[0, 0], [0, 0]]) is False

    def test_returns_a_plain_python_bool_not_a_numpy_bool(self):
        mask = np.array([[1, 1]])

        result = inference.has_plume(mask)

        assert type(result) is bool

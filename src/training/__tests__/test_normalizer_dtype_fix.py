"""Tests for src/training/normalizer_dtype_fix.py -- pure runtime attribute
override (Test Size: Small, real torch objects, no mocking; see TASK-3.2 in
mlops-methane-detection-plan.md).

STARCOP's DataNormalizer (vendor/starcop/starcop/data/normalizer_module.py)
builds offsets_input/factors_input/clip_min_input/clip_max_input Parameters
via `torch.from_numpy(np.array(python_ints_or_floats))` -- when every value
for the active input_products happens to be a plain int (e.g. clip=(0, 2)
for AVIRIS bands and mag1c), numpy infers int64, and the resulting Parameter
is int64. `torch.clamp` on CPU implicitly promotes an int64 bound against a
float32 input; MPS's clamp kernel cannot broadcast a dtype-mismatched pair
and aborts the process (not a catchable Python exception). This module casts
those Parameters to float32 in place from outside DataNormalizer -- a
runtime attribute override, not an edit to vendor/starcop/ (composition-only
rule, decision 0 in TASK-2.2).
"""

import torch

import normalizer_dtype_fix


class _FakeNormalizer(torch.nn.Module):
    """Stands in for STARCOP's real DataNormalizer -- same attribute names,
    real torch.nn.Parameter values, no vendor import needed for this pure
    dtype-casting logic.
    """

    def __init__(self, **params):
        super().__init__()
        for name, value in params.items():
            if value is None:
                setattr(self, name, None)
            else:
                setattr(self, name, torch.nn.Parameter(value, requires_grad=False))


class TestCastNormalizerParamsToFloat32:
    def test_casts_int64_input_bound_parameters_to_float32(self):
        normalizer = _FakeNormalizer(
            offsets_input=torch.tensor([0, 0, 0], dtype=torch.int64),
            factors_input=torch.tensor([1, 60, 1750], dtype=torch.int64),
            clip_min_input=torch.tensor([0, 0, 0], dtype=torch.int64),
            clip_max_input=torch.tensor([2, 2, 2], dtype=torch.int64),
        )

        normalizer_dtype_fix.cast_normalizer_params_to_float32(normalizer)

        assert normalizer.offsets_input.dtype == torch.float32
        assert normalizer.factors_input.dtype == torch.float32
        assert normalizer.clip_min_input.dtype == torch.float32
        assert normalizer.clip_max_input.dtype == torch.float32

    def test_preserves_numeric_values_after_casting(self):
        normalizer = _FakeNormalizer(factors_input=torch.tensor([60], dtype=torch.int64))

        normalizer_dtype_fix.cast_normalizer_params_to_float32(normalizer)

        assert torch.equal(normalizer.factors_input, torch.tensor([60.0]))

    def test_is_a_noop_when_a_parameter_is_already_float32(self):
        normalizer = _FakeNormalizer(clip_max_input=torch.tensor([2.0], dtype=torch.float32))
        original = normalizer.clip_max_input

        normalizer_dtype_fix.cast_normalizer_params_to_float32(normalizer)

        assert normalizer.clip_max_input is original

    def test_skips_output_parameters_that_are_none_without_raising(self):
        normalizer = _FakeNormalizer(
            offsets_input=torch.tensor([0], dtype=torch.int64),
            offsets_output=None,
            factors_output=None,
        )

        normalizer_dtype_fix.cast_normalizer_params_to_float32(normalizer)  # should not raise

    def test_skips_attributes_that_are_entirely_absent_without_raising(self):
        # Mirrors STARCOP's real DataNormalizer: when factors_output is
        # empty, clip_min_output/clip_max_output are never set at all --
        # not even to None (see normalizer_module.py:120-131).
        normalizer = _FakeNormalizer(offsets_input=torch.tensor([0], dtype=torch.int64))

        normalizer_dtype_fix.cast_normalizer_params_to_float32(normalizer)  # should not raise

"""Casts STARCOP's DataNormalizer clip/offset/factor Parameters to float32
-- see TASK-3.2 in mlops-methane-detection-plan.md.

vendor/starcop/starcop/data/normalizer_module.py builds these Parameters via
`torch.from_numpy(np.array(python_ints_or_floats))`; when every value for
the active input_products is a plain int (e.g. clip=(0, 2)), numpy infers
int64. torch.clamp on CPU implicitly promotes an int64 bound against a
float32 input, so this goes unnoticed there -- but MPS's clamp kernel cannot
broadcast a dtype-mismatched pair and aborts the process. Casting to float32
is numerically a no-op (same values, wider dtype), so this is applied
unconditionally rather than gated on accelerator -- a runtime attribute
override, never an edit to vendor/starcop/ (composition-only rule).
"""

import torch

_PARAM_NAMES = (
    "offsets_input",
    "factors_input",
    "clip_min_input",
    "clip_max_input",
    "offsets_output",
    "factors_output",
    "clip_min_output",
    "clip_max_output",
)


def cast_normalizer_params_to_float32(normalizer: torch.nn.Module) -> None:
    """Casts every present, non-None DataNormalizer Parameter to float32 in
    place. Some output-side Parameters may be None (no output-band
    normalization configured) or entirely absent as an attribute
    (STARCOP's own DataNormalizer.__init__ only sets clip_min_output/
    clip_max_output when factors_output is non-empty) -- both are skipped.
    """
    for name in _PARAM_NAMES:
        param = getattr(normalizer, name, None)
        if param is None or param.dtype == torch.float32:
            continue
        setattr(
            normalizer,
            name,
            torch.nn.Parameter(param.data.float(), requires_grad=param.requires_grad),
        )

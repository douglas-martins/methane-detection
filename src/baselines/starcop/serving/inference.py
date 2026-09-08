"""Pure tensor-assembly and inference logic for the BentoML serving layer
(TASK-5.1). No MLflow, no model loading, no I/O -- see model_loader.py for
that.

The mask/confidence recipe mirrors vendor/starcop's own
ModelModule.batch_with_preds (vendor/starcop/starcop/models/model_module.py:
191-208): sigmoid over the raw logits, then threshold at 0.5.
ModelModule.forward already applies input normalization internally
(self.network(self.normalizer.normalize_x(x))), so no separate
normalization step is needed here -- only correct tensor assembly.
"""

import numpy as np
import torch


def assemble_input_tensor(array: np.ndarray, expected_channels: int) -> torch.Tensor:
    """Builds a (1, expected_channels, H, W) float32 tensor from a raw
    uploaded scene.

    Accepts either channel-last (H, W, C) or already channel-first
    (C, H, W) arrays -- whichever axis matches `expected_channels` is
    treated as the channel axis. Raises ValueError if the array isn't 3D,
    if neither axis matches, or if both the first and last axis match
    (ambiguous -- e.g. a square-ish patch where H or W happens to equal the
    channel count -- silently guessing risks transposing H and W).
    """
    if array.ndim != 3:
        raise ValueError(f"expected a 3D array (H, W, C) or (C, H, W), got shape {array.shape}")

    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"array has non-numeric dtype {array.dtype!r}; expected a numeric array")

    if np.issubdtype(array.dtype, np.complexfloating):
        # complex is a subtype of np.number, so the check above alone lets
        # it through -- but .astype(float32) below doesn't raise for a
        # complex array, it silently discards the imaginary part (only a
        # ComplexWarning), which would corrupt input data without error.
        raise ValueError(f"array has complex dtype {array.dtype!r}; expected a real-valued array")

    first_matches = array.shape[0] == expected_channels
    last_matches = array.shape[-1] == expected_channels

    if first_matches and last_matches:
        raise ValueError(
            f"array shape {array.shape} is ambiguous: both the first and last axis "
            f"match the model's expected {expected_channels} channels, so channel-last "
            "vs. channel-first can't be determined"
        )
    elif last_matches:
        chw = np.transpose(array, (2, 0, 1))
    elif first_matches:
        chw = array
    else:
        raise ValueError(
            f"array shape {array.shape} has no axis matching the model's "
            f"expected {expected_channels} channels"
        )

    # .astype(float32) happens on the numpy side, before torch.from_numpy --
    # numpy can downcast any numeric dtype (including e.g. np.longdouble,
    # which torch.from_numpy rejects directly despite being numeric) to a
    # dtype torch.from_numpy is guaranteed to accept.
    return torch.from_numpy(np.ascontiguousarray(chw).astype(np.float32)).unsqueeze(0)


def run_inference(model, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Runs `model(x)` and returns (mask, probs) as (H, W) numpy arrays.

    probs is sigmoid(logits) (confidence scores); mask is probs > 0.5 as
    0/1 integers -- the same recipe as ModelModule.batch_with_preds.
    """
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)
        mask = (probs > 0.5).long()

    return (
        mask.squeeze(0).squeeze(0).numpy(),
        probs.squeeze(0).squeeze(0).numpy(),
    )


def per_band_means(array: np.ndarray, expected_channels: int) -> list:
    """Per-channel spatial mean of a raw uploaded scene, in the model's own
    channel order. Backs TASK-6.2's rolling drift-detection window: each
    request contributes one mean-per-band value here.

    Reuses assemble_input_tensor's channel-axis detection/validation, so
    this raises the same ValueError it does on a channel-count mismatch,
    and returns values on the exact same raw (non-normalized) scale
    run_inference/predict_response use -- assemble_input_tensor only
    transposes/casts dtype, normalization happens inside model.forward.
    """
    x = assemble_input_tensor(array, expected_channels)
    return x.squeeze(0).mean(dim=(1, 2)).tolist()


def has_plume(mask) -> bool:
    """True if any pixel in a binary segmentation mask is a positive (plume)
    prediction. Accepts either a numpy array or a plain nested list (e.g.
    predict_response's JSON-ready "mask" field) -- backs TASK-6.1's
    methane_prediction_total Prometheus counter in service.py, which labels
    each /predict response by predicted class.
    """
    return bool(np.any(np.asarray(mask)))


def predict_response(model, array: np.ndarray, expected_channels: int) -> dict:
    """Full assemble -> infer -> shape-as-JSON pipeline behind POST
    /predict's response body. Pulled out of service.py (BentoML SDK glue,
    not unit tested -- see that module's docstring) so this logic has real
    unit test coverage instead of only a live `bentoml serve` + curl run.

    Raises ValueError (propagated from assemble_input_tensor) on a
    channel-count mismatch or a non-3D array -- service.py is responsible
    for translating that into a client-facing 400, not this function.
    """
    x = assemble_input_tensor(array, expected_channels)
    mask, probs = run_inference(model, x)
    return {"mask": mask.tolist(), "confidence": probs.tolist()}

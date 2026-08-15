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
    treated as the channel axis. Raises ValueError if the array isn't 3D or
    neither axis matches.
    """
    if array.ndim != 3:
        raise ValueError(f"expected a 3D array (H, W, C) or (C, H, W), got shape {array.shape}")

    if array.shape[-1] == expected_channels:
        chw = np.transpose(array, (2, 0, 1))
    elif array.shape[0] == expected_channels:
        chw = array
    else:
        raise ValueError(
            f"array shape {array.shape} has no axis matching the model's "
            f"expected {expected_channels} channels"
        )

    return torch.from_numpy(np.ascontiguousarray(chw)).float().unsqueeze(0)


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

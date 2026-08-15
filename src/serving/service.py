"""BentoML inference service for the STARCOP methane-plume segmentation
model (TASK-5.1).

Loads a model from the MLflow registry at startup (MODEL_NAME/MODEL_STAGE
env vars, defaulting to the real starcop-baseline-mag1c-rgb Staging model --
see mlops-methane-detection-plan.md TASK-5.1's 2026-08-15 readiness audit)
and exposes POST /predict, GET /health. GET /metrics is BentoML's own
built-in Prometheus endpoint, not hand-rolled here.

This class itself is thin BentoML SDK glue -- framework decorators, env var
reads, exception-to-HTTP-status translation -- exercised by a real
`bentoml serve` run + curl rather than unit tests, matching this repo's
established Test Size: Large boundary for framework wiring (see
src/training/train.py, src/registry/hf_baseline_import.py). The actual
predict logic (assemble -> infer -> shape response) is NOT here: it lives in
inference.py::predict_response, which IS unit tested directly (real fixed-
logits model, no mocking) -- the same "thin glue vs. tested logic" split
already used elsewhere in this repo (e.g. TASK-3.3a's launch_profiles.py,
TASK-2.3's promotion_criteria.py).
"""

import os
import sys
import time
from pathlib import Path

import bentoml
import numpy as np
from bentoml.exceptions import BadInput

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import inference  # noqa: E402
import model_loader  # noqa: E402

DEFAULT_MODEL_NAME = "starcop-baseline-mag1c-rgb"
DEFAULT_MODEL_STAGE = "Staging"


@bentoml.service(resources={"cpu": "2"}, traffic={"timeout": 30})
class MethaneDetectionService:
    def __init__(self) -> None:
        tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
        self.model_name = os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME)
        self.model_stage = os.environ.get("MODEL_STAGE", DEFAULT_MODEL_STAGE)

        self.model, self.model_version = model_loader.load_model(
            tracking_uri, self.model_name, self.model_stage
        )
        self.num_channels = self.model.num_channels
        self._start_time = time.time()

    @bentoml.api(route="/predict")
    def predict(self, file: Path) -> dict:
        """Accepts an uploaded .npy hyperspectral patch ((H, W, C) or
        (C, H, W), channel count matching the loaded model) and returns the
        binary segmentation mask plus per-pixel confidence scores.
        """
        try:
            array = np.load(file)
        except (OSError, ValueError) as exc:
            raise BadInput(f"could not read {file.name!r} as a .npy array: {exc}") from exc

        if not isinstance(array, np.ndarray):
            # np.load doesn't raise for a .npz archive (multiple named
            # arrays, uploaded under a misleading .npy-looking request) --
            # it returns an NpzFile with no .ndim/.shape/.dtype, which would
            # otherwise reach predict_response and fail with an uncaught
            # AttributeError (opaque 500) instead of a client-facing 400.
            if isinstance(array, np.lib.npyio.NpzFile):
                array.close()
            raise BadInput(
                f"{file.name!r} did not load as a single .npy array "
                f"(got {type(array).__name__}); .npz archives are not supported"
            )

        try:
            return inference.predict_response(self.model, array, self.num_channels)
        except ValueError as exc:
            raise BadInput(str(exc)) from exc

    @bentoml.api(route="/health")
    def health(self) -> dict:
        return {
            "status": "ok",
            "model_name": self.model_name,
            "model_version": self.model_version.version,
            "model_stage": self.model_stage,
            "uptime_seconds": time.time() - self._start_time,
            "device": str(next(self.model.parameters()).device),
        }

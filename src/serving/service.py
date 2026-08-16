"""BentoML inference service for the STARCOP methane-plume segmentation
model (TASK-5.1).

Loads a model from the MLflow registry at startup (MODEL_NAME/MODEL_STAGE
env vars, defaulting to the real starcop-baseline-mag1c-rgb Staging model --
see mlops-methane-detection-plan.md TASK-5.1's 2026-08-15 readiness audit)
and exposes POST /predict, GET /health. GET /metrics is BentoML's own
built-in Prometheus endpoint, not hand-rolled here -- it already covers
request count, latency, and error rate (TASK-6.1's readiness audit traced
this directly against bentoml's own instrumentation code). The one metric
it can't provide -- which class /predict actually returned -- is added
below as a small custom counter, for TASK-6.1's "prediction class
distribution over time" dashboard panel and its detection-rate alert.

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
import threading
import time
from collections import deque
from pathlib import Path

import bentoml
import numpy as np
from bentoml.exceptions import BadInput

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import band_baseline  # noqa: E402
import drift  # noqa: E402
import inference  # noqa: E402
import model_loader  # noqa: E402

_DRIFT_WINDOW_SIZE = 100

DEFAULT_MODEL_NAME = "starcop-baseline-mag1c-rgb"
DEFAULT_MODEL_STAGE = "Staging"

_prediction_counter = None


def _get_prediction_counter():
    """Lazily creates the methane_prediction_total Counter on first use.

    Deliberately NOT created at module import time, and deliberately using
    prometheus_client directly rather than the deprecated bentoml.metrics
    shim: bentoml.metrics's own docstring warns that BentoML's worker
    processes set PROMETHEUS_MULTIPROC_DIR *after* this module is first
    imported, so a Counter() constructed eagerly at import time can end up
    registered against a single-process registry that the /metrics
    MultiProcessCollector never reads -- silently invisible on the real
    endpoint despite working in-process. Deferring construction to first
    call (from inside predict(), i.e. after BentoML's own startup) avoids
    that ordering hazard the same way bentoml.metrics's lazy __getattr__
    does, without triggering its deprecation warning.
    """
    global _prediction_counter
    if _prediction_counter is None:
        import prometheus_client

        _prediction_counter = prometheus_client.Counter(
            "methane_prediction_total",
            "Count of /predict responses, labeled by predicted class",
            labelnames=["result"],
        )
    return _prediction_counter


_band_drift_gauge = None


def _get_band_drift_gauge():
    """Lazily creates the methane_band_kl_divergence Gauge on first use --
    same deferred-construction reasoning as _get_prediction_counter above.
    """
    global _band_drift_gauge
    if _band_drift_gauge is None:
        import prometheus_client

        _band_drift_gauge = prometheus_client.Gauge(
            "methane_band_kl_divergence",
            "Gaussian KL divergence of each input band's rolling per-request "
            "mean from its training baseline",
            labelnames=["band"],
        )
    return _band_drift_gauge


@bentoml.service(resources={"cpu": "2"}, traffic={"timeout": 30})
class MethaneDetectionService:
    """BentoML service exposing the STARCOP segmentation model over HTTP.

    See module docstring for the full design (routes, model-loading, and
    the thin-glue-vs-tested-logic split with inference.py).
    """

    def __init__(self) -> None:
        """Loads the configured MLflow model/stage once at service startup."""
        tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
        self.model_name = os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME)
        self.model_stage = os.environ.get("MODEL_STAGE", DEFAULT_MODEL_STAGE)

        self.model, self.model_version = model_loader.load_model(
            tracking_uri, self.model_name, self.model_stage
        )
        self.num_channels = self.model.num_channels
        self._start_time = time.time()

        # TASK-6.2 input-drift rolling state: one deque per band, in-memory
        # and per-process (resets on redeploy, would fragment across
        # replicas if this service is ever horizontally scaled -- not a
        # concern today, single instance, no autoscaling configured).
        # BentoML dispatches this class's sync @bentoml.api methods via
        # Starlette's run_in_threadpool (confirmed in bentoml's own
        # http_app.py), so concurrent /predict requests genuinely run on
        # separate threads against this same instance -- _band_lock
        # serializes access to _band_windows so a concurrent deque append
        # can't interleave with another thread's iteration over it (which
        # raises "deque mutated during iteration", uncaught by predict()'s
        # existing except clause) and so each band's rolling mean/std is
        # read as a consistent snapshot, not a torn one.
        self.band_names = band_baseline.band_names_for_model(self.model_name, self.num_channels)
        self._band_windows = {name: deque(maxlen=_DRIFT_WINDOW_SIZE) for name in self.band_names}
        self._band_lock = threading.Lock()

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
            result = inference.predict_response(self.model, array, self.num_channels)
        except ValueError as exc:
            raise BadInput(str(exc)) from exc

        label = "plume_detected" if inference.has_plume(result["mask"]) else "no_plume"
        _get_prediction_counter().labels(result=label).inc()

        # TASK-6.2: per-band input drift. Reuses the same array/channel
        # count already validated above by predict_response -- cannot fail
        # differently here.
        means = inference.per_band_means(array, self.num_channels)
        for band_name, value in zip(self.band_names, means):
            baseline = band_baseline.baseline_for_band(band_name)
            if baseline is None:
                continue
            with self._band_lock:
                window = self._band_windows[band_name]
                rolling = drift.update_rolling_stats(window, value)
                divergence = drift.kl_divergence_gaussian(
                    rolling.mean, rolling.std, baseline.mean, baseline.std
                )
                has_enough_samples = len(window) >= _DRIFT_WINDOW_SIZE
            if has_enough_samples:
                # Below the full window size, sigma_p is estimated from too
                # few samples to be trustworthy -- with 1-2 samples it's
                # often clamped to _MIN_SIGMA entirely (drift.py), producing
                # a large divergence that reflects sample scarcity, not
                # real drift. Leaving the gauge unset until the window
                # fills (rather than publishing a misleading early value)
                # means the series simply doesn't exist on /metrics yet,
                # which the alert rule's noDataState: NoData already
                # handles as "nothing to evaluate", not a false positive.
                # Trade-off: the alert is blind for the first
                # _DRIFT_WINDOW_SIZE requests after every redeploy, since
                # the rolling windows reset then too (see __init__'s
                # comment).
                _get_band_drift_gauge().labels(band=band_name).set(divergence)

        return result

    @bentoml.api(route="/health")
    def health(self) -> dict:
        """Returns loaded-model identity, process uptime, and inference device."""
        return {
            "status": "ok",
            "model_name": self.model_name,
            "model_version": self.model_version.version,
            "model_stage": self.model_stage,
            "uptime_seconds": time.time() - self._start_time,
            "device": str(next(self.model.parameters()).device),
        }

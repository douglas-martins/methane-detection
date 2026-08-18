"""Pure helpers for MLflow run tagging/param-logging.

Kept free of any `mlflow` SDK calls so they're testable without a live
tracking server -- src/training/train.py does the thin SDK glue
(mlflow.set_tags(build_run_tags(...)), mlflow.log_params(flatten_hydra_params(...)))
around these.
"""

import os
from typing import Dict

from omegaconf import DictConfig, OmegaConf

RUN_ID_MARKER_FILENAME = "mlflow_run_id.txt"

RUN_ID_MARKER_FILENAME = "mlflow_run_id.txt"

REQUIRED_TRACKING_ENV_VARS = (
    "MLFLOW_TRACKING_URI",
    "MLFLOW_TRACKING_USERNAME",
    "MLFLOW_TRACKING_PASSWORD",
)


def require_mlflow_tracking_env() -> None:
    """Raises RuntimeError if any of REQUIRED_TRACKING_ENV_VARS is unset.

    mlflow.start_run() silently falls back to a local file/sqlite tracking
    store when MLFLOW_TRACKING_URI is unset -- it does not raise -- so a
    missing var would otherwise train without ever reaching the intended
    server instead of failing fast. Call this before the first mlflow call.
    """
    missing = [name for name in REQUIRED_TRACKING_ENV_VARS if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            "Missing required MLflow tracking environment variable(s): "
            f"{', '.join(missing)} (see docs/environment_notes.md)"
        )


def build_run_tags(
    dataset_version: str, dataset_dirty: bool, machine: str, sensor: str
) -> Dict[str, str]:
    """Builds the MLflow run tags identifying what data/machine a run used."""
    return {
        "dataset_version": dataset_version,
        "dataset_dirty": str(dataset_dirty),
        "machine": machine,
        "sensor": sensor,
    }


def write_run_id_marker(experiment_path: str, run_id: str) -> str:
    """Writes `run_id` to a file under `experiment_path` and returns a
    stdout-sentinel line an external subprocess caller can scan for.

    train.py's own stdout/train.log is dominated by third-party INFO spam
    (botocore, Lightning progress bars), so callers should look for this
    exact "MLFLOW_RUN_ID=..." line rather than parsing logging output.
    """
    marker_path = os.path.join(experiment_path, RUN_ID_MARKER_FILENAME)
    with open(marker_path, "w") as f:
        f.write(run_id)
    return f"MLFLOW_RUN_ID={run_id}"


def flatten_hydra_params(settings: DictConfig) -> Dict[str, str]:
    """Flattens a (possibly nested) Hydra config into MLflow-safe params.

    MLflow's log_params rejects nested/non-string values, so nested keys
    become dot-separated (dataset.input_products) and non-scalar values
    (e.g. a list like input_products) are stringified.
    """
    container = OmegaConf.to_container(settings, resolve=True)

    flat: Dict[str, str] = {}

    def _walk(prefix: str, obj) -> None:
        """Recursively flattens `obj` into `flat`, dot-joining nested keys."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                child_prefix = f"{prefix}.{key}" if prefix else str(key)
                _walk(child_prefix, value)
        else:
            flat[prefix] = str(obj)

    _walk("", container)
    return flat

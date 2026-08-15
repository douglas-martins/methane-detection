"""MLflow SDK glue for loading a model from the registry into the BentoML
serving layer (TASK-5.1).

Importing this module puts vendor/starcop on sys.path via
_vendor_starcop_serving (see that module's docstring) -- required before
mlflow.pytorch.load_model() can unpickle a real STARCOP checkpoint.

load_model is SDK glue, unit tested against a real local sqlite tracking
store with a tiny real nn.Module standing in for the model (Test Size:
Medium, mirrors src/registry/__tests__/test_mlflow_registry.py's own
precedent); loading a real STARCOP checkpoint end to end is validated by an
actual run instead, same Test Size: Large boundary hf_baseline_import.py's
own load_model is at.
"""

import sys
from pathlib import Path
from typing import Tuple

import _vendor_starcop_serving  # noqa: F401 -- sys.path side effect, see its docstring
import mlflow.pytorch
from mlflow.entities.model_registry import ModelVersion
from mlflow.tracking import MlflowClient


def load_model(tracking_uri: str, model_name: str, stage: str) -> Tuple[object, ModelVersion]:
    """Loads the model currently at `stage` for `model_name` from the MLflow
    registry at `tracking_uri`. Returns (model, ModelVersion) with the model
    already switched to eval() mode. Raises ValueError if no version of
    `model_name` is currently at `stage`.
    """
    _registry_dir = str(Path(__file__).resolve().parents[1] / "registry")
    if _registry_dir not in sys.path:
        sys.path.insert(0, _registry_dir)
    import mlflow_registry  # noqa: E402

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    version = mlflow_registry.resolve_stage_version(client, model_name, stage)
    # version.source is the exact runs:/<run_id>/<artifact_path> URI recorded
    # at registration time (mlflow_registry.register_and_promote sets it from
    # its own artifact_path param, not hardcoded to "model") -- using it
    # directly, rather than reconstructing a URI that assumes artifact_path
    # is always "model", loads correctly regardless of what artifact_path a
    # given registration actually used. Still deliberately a runs:/ URI, not
    # models:/<name>/<version>: verified directly against the live server
    # (2026-08-15) that the registry-scheme resolver raises `MlflowException:
    # No such artifact: 'MLmodel'` against this project's S3-compatible
    # (Backblaze B2) artifact store, while the equivalent runs:/ URI for the
    # exact same underlying artifact loads correctly.
    model = mlflow.pytorch.load_model(version.source)
    model.eval()
    return model, version

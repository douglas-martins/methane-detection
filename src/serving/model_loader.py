"""MLflow SDK glue for loading a model from the registry into the BentoML
serving layer (TASK-5.1).

Importing this module puts vendor/starcop on sys.path via
_vendor_starcop_serving (see that module's docstring) -- required before
mlflow.pytorch.load_model() can unpickle a real STARCOP checkpoint.

resolve_model_uri is pure, unit tested directly. load_model is SDK glue,
unit tested against a real local sqlite tracking store with a tiny real
nn.Module standing in for the model (Test Size: Medium, mirrors
src/registry/__tests__/test_mlflow_registry.py's own precedent); loading a
real STARCOP checkpoint end to end is validated by an actual run instead,
same Test Size: Large boundary hf_baseline_import.py's own load_model is at.
"""

import sys
from pathlib import Path
from typing import Tuple

import _vendor_starcop_serving  # noqa: F401 -- sys.path side effect, see its docstring
import mlflow.pytorch
from mlflow.entities.model_registry import ModelVersion
from mlflow.tracking import MlflowClient


def resolve_model_uri(run_id: str, artifact_path: str = "model") -> str:
    """Returns a run-scoped MLflow model URI (``runs:/<run_id>/<artifact_path>``).

    Deliberately *not* a `models:/<name>/<version>` registry-scheme URI:
    verified directly against the live server (2026-08-15) that the
    registry-scheme resolver raises `MlflowException: No such artifact:
    'MLmodel'` against this project's S3-compatible (Backblaze B2) artifact
    store, while the equivalent `runs:/<run_id>/model` URI for the exact
    same underlying artifact loads correctly. `ModelVersion.run_id` gives
    the run to build this from, so the registry (`resolve_stage_version`) is
    still what decides *which* run is current at a given stage -- only the
    final load URI shape changes.
    """
    return f"runs:/{run_id}/{artifact_path}"


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
    model = mlflow.pytorch.load_model(resolve_model_uri(version.run_id))
    model.eval()
    return model, version

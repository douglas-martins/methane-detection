"""MLflow SDK glue for the model registry promotion workflow (TASK-2.3).

Thin wrappers around MlflowClient, kept separate from promotion_criteria.py
(pure decision logic) so the decision logic is testable without any tracking
store at all, while this module is tested against a real local sqlite
tracking store (Test Size: Medium) -- never mocked.

Uses MlflowClient's registry methods directly (create_registered_model /
create_model_version / transition_model_version_stage) rather than the
mlflow.register_model() convenience function, so callers pass an explicit,
already-configured client instead of relying on global
mlflow.set_tracking_uri() state.
"""

from typing import Dict, List

from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient


def resolve_run_id(client: MlflowClient, run_id: str | None, experiment_id: str = "0") -> str:
    """Returns `run_id` unchanged if given, else the most recently started
    run in `experiment_id`. Raises ValueError if that experiment has no runs.
    """
    if run_id is not None:
        return run_id

    runs = client.search_runs(
        experiment_ids=[experiment_id], order_by=["attributes.start_time DESC"], max_results=1
    )
    if not runs:
        raise ValueError(f"experiment {experiment_id!r} has no runs to resolve a run_id from")
    return runs[0].info.run_id


def fetch_run_metrics(client: MlflowClient, run_id: str) -> Dict[str, float]:
    """Returns the final logged value of every metric on `run_id`."""
    return dict(client.get_run(run_id).data.metrics)


def fetch_metric_history(client: MlflowClient, run_id: str, key: str) -> List[float]:
    """Returns every logged value of metric `key` on `run_id`, ordered by step."""
    history = client.get_metric_history(run_id, key)
    return [metric.value for metric in sorted(history, key=lambda metric: metric.step)]


def register_and_promote(
    client: MlflowClient,
    run_id: str,
    model_name: str,
    stage: str,
    artifact_path: str = "model",
) -> ModelVersion:
    """Registers `run_id`'s `artifact_path` artifact as a new version of
    `model_name` (creating the registered model if it doesn't exist yet) and
    transitions that version to `stage`. Idempotent per run_id: a repeat call
    for a run already registered under `model_name` reuses that version
    instead of creating a duplicate (e.g. a retried CI promotion step).
    """
    try:
        client.create_registered_model(model_name)
    except MlflowException as exc:
        if exc.error_code != "RESOURCE_ALREADY_EXISTS":
            raise

    existing = [
        version
        for version in client.search_model_versions(f"name='{model_name}'")
        if version.run_id == run_id
    ]
    if len(existing) > 1:
        raise ValueError(
            f"run {run_id!r} matches {len(existing)} existing versions of "
            f"{model_name!r} (versions {[v.version for v in existing]}); expected at most 1"
        )

    model_version = (
        existing[0]
        if existing
        else client.create_model_version(
            name=model_name, source=f"runs:/{run_id}/{artifact_path}", run_id=run_id
        )
    )
    return client.transition_model_version_stage(
        name=model_name, version=model_version.version, stage=stage
    )

"""Tests for src/registry/promote_model.py's decide_and_promote -- the
orchestration function the CLI wraps. Real fixtures: a real MlflowClient
against a fresh local sqlite tracking store (Test Size: Medium), exercising
the actual experiment -> staging -> production ladder end to end.
"""

from types import SimpleNamespace

import promote_model
import pytest
from mlflow.tracking import MlflowClient

SMOOTH_LOSS_HISTORY = [0.9, 0.7, 0.6, 0.55, 0.5, 0.48, 0.47]
SPIKING_LOSS_HISTORY = [0.5, 0.9, 0.05, 0.85, 0.02]


@pytest.fixture
def client(tmp_path):
    return MlflowClient(tracking_uri=f"sqlite:///{tmp_path}/mlflow.db")


def _create_run(client, tmp_path, metrics, val_loss_history, with_model_artifact=True):
    run = client.create_run("0", start_time=1000)
    for key, value in metrics.items():
        client.log_metric(run.info.run_id, key, value, step=0)
    for step, value in enumerate(val_loss_history):
        client.log_metric(run.info.run_id, "val_loss", value, step=step)
    if with_model_artifact:
        artifact_file = tmp_path / "dummy_model_file.txt"
        artifact_file.write_text("not a real model, just exercising the registry API")
        client.log_artifact(run.info.run_id, str(artifact_file), artifact_path="model")
    return run


class _RecordingRegistry:
    def __init__(self, metrics):
        self.metrics = metrics
        self.resolve_args = None
        self.fetch_metrics_args = None
        self.fetch_history_args = None
        self.register_args = None

    def resolve_run_id(self, client, run_id, experiment_id):
        self.resolve_args = (client, run_id, experiment_id)
        return "resolved-run"

    def fetch_run_metrics(self, client, run_id):
        self.fetch_metrics_args = (client, run_id)
        return self.metrics

    def fetch_metric_history(self, client, run_id, key):
        self.fetch_history_args = (client, run_id, key)
        return SMOOTH_LOSS_HISTORY

    def register_and_promote(self, client, run_id, model_name, stage):
        self.register_args = (client, run_id, model_name, stage)
        return SimpleNamespace(name=model_name, version="7", current_stage=stage)


class TestDecideAndPromote:
    def test_promotes_to_production_when_staging_and_production_criteria_both_pass(
        self, client, tmp_path
    ):
        run = _create_run(
            client,
            tmp_path,
            metrics={
                "val_accuracy": 0.95,
                "val_f1score": 0.9,
                "test_accuracy": 0.95,
                "test_f1score": 0.9,
            },
            val_loss_history=SMOOTH_LOSS_HISTORY,
        )

        outcome = promote_model.decide_and_promote(
            client, run_id=run.info.run_id, model_name="methane-cnn-starcop"
        )

        assert outcome.stage == "Production"
        assert outcome.model_version is not None
        assert outcome.model_version.current_stage == "Production"

    def test_promotes_to_staging_when_only_staging_criteria_pass(self, client, tmp_path):
        run = _create_run(
            client,
            tmp_path,
            metrics={
                "val_accuracy": 0.87,
                "val_f1score": 0.75,
                "test_accuracy": 0.5,
                "test_f1score": 0.5,
            },
            val_loss_history=SMOOTH_LOSS_HISTORY,
        )

        outcome = promote_model.decide_and_promote(
            client, run_id=run.info.run_id, model_name="methane-cnn-starcop"
        )

        assert outcome.stage == "Staging"
        assert outcome.model_version is not None
        assert outcome.model_version.current_stage == "Staging"

    def test_rejects_when_staging_metrics_fail(self, client, tmp_path):
        run = _create_run(
            client,
            tmp_path,
            metrics={"val_accuracy": 0.5, "val_f1score": 0.4},
            val_loss_history=SMOOTH_LOSS_HISTORY,
            with_model_artifact=False,
        )

        outcome = promote_model.decide_and_promote(
            client, run_id=run.info.run_id, model_name="methane-cnn-starcop"
        )

        assert outcome.stage is None
        assert outcome.model_version is None
        assert outcome.run_id == run.info.run_id
        assert any("val_accuracy" in reason for reason in outcome.decision.reasons)

    def test_forwards_production_promotion_arguments_and_preserves_outcome(self, monkeypatch):
        client = object()
        registry = _RecordingRegistry(
            metrics={
                "val_accuracy": 0.95,
                "val_f1score": 0.9,
                "test_accuracy": 0.95,
                "test_f1score": 0.9,
            }
        )
        monkeypatch.setattr(promote_model, "mlflow_registry", registry)

        outcome = promote_model.decide_and_promote(
            client,
            run_id="input-run",
            model_name="custom-model",
            experiment_id="custom-experiment",
        )

        assert registry.resolve_args == (client, "input-run", "custom-experiment")
        assert registry.fetch_metrics_args == (client, "resolved-run")
        assert registry.fetch_history_args == (client, "resolved-run", "val_loss")
        assert registry.register_args == (client, "resolved-run", "custom-model", "Production")
        assert outcome.run_id == "resolved-run"
        assert outcome.stage == "Production"
        assert outcome.model_version.current_stage == "Production"
        assert outcome.decision.promote is True

    def test_forwards_staging_promotion_arguments_and_preserves_outcome(self, monkeypatch):
        client = object()
        registry = _RecordingRegistry(
            metrics={
                "val_accuracy": 0.87,
                "val_f1score": 0.75,
                "test_accuracy": 0.5,
                "test_f1score": 0.5,
            }
        )
        monkeypatch.setattr(promote_model, "mlflow_registry", registry)

        outcome = promote_model.decide_and_promote(
            client,
            run_id="input-run",
            model_name="custom-model",
            experiment_id="custom-experiment",
        )

        assert registry.resolve_args == (client, "input-run", "custom-experiment")
        assert registry.fetch_metrics_args == (client, "resolved-run")
        assert registry.fetch_history_args == (client, "resolved-run", "val_loss")
        assert registry.register_args == (client, "resolved-run", "custom-model", "Staging")
        assert outcome.run_id == "resolved-run"
        assert outcome.stage == "Staging"
        assert outcome.model_version.current_stage == "Staging"
        assert outcome.decision.promote is True

    def test_rejects_when_loss_history_is_unstable_even_if_metrics_pass(self, client, tmp_path):
        run = _create_run(
            client,
            tmp_path,
            metrics={"val_accuracy": 0.95, "val_f1score": 0.9},
            val_loss_history=SPIKING_LOSS_HISTORY,
            with_model_artifact=False,
        )

        outcome = promote_model.decide_and_promote(
            client, run_id=run.info.run_id, model_name="methane-cnn-starcop"
        )

        assert outcome.stage is None
        assert any("loss" in reason.lower() for reason in outcome.decision.reasons)

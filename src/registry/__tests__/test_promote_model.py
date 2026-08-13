"""Tests for src/registry/promote_model.py's decide_and_promote -- the
orchestration function the CLI wraps. Real fixtures: a real MlflowClient
against a fresh local sqlite tracking store (Test Size: Medium), exercising
the actual experiment -> staging -> production ladder end to end.
"""

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
        assert any("val_accuracy" in reason for reason in outcome.decision.reasons)

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

"""Tests for src/registry/mlflow_registry.py -- MLflow SDK glue.

Real fixtures, not mocks: each test spins up a real MlflowClient against a
fresh local sqlite tracking store (Test Size: Medium -- local sqlite file,
no network, no live server). The plain file-store backend is hard-deprecated
in the installed MLflow version for anything beyond reading existing data
(raises MlflowException on write unless MLFLOW_ALLOW_FILE_STORE=true) and
historically never supported the model registry at all, so sqlite is both
the only realistic option and what the real tracking server itself uses.
"""

import mlflow_registry
import pytest
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient


@pytest.fixture
def client(tmp_path):
    return MlflowClient(tracking_uri=f"sqlite:///{tmp_path}/mlflow.db")


def _create_run_with_metric(client, experiment_id, start_time, key=None, values=None):
    """Creates a real run, optionally logging `key` at steps 0..len(values)-1."""
    run = client.create_run(experiment_id, start_time=start_time)
    if key is not None:
        for step, value in enumerate(values):
            client.log_metric(run.info.run_id, key, value, step=step)
    return run


class TestResolveRunId:
    def test_returns_explicit_run_id_unchanged(self, client):
        result = mlflow_registry.resolve_run_id(client, run_id="explicit-run-id")

        assert result == "explicit-run-id"

    def test_falls_back_to_the_latest_run_in_the_experiment_by_start_time(self, client):
        older = _create_run_with_metric(client, "0", start_time=1000)
        newer = _create_run_with_metric(client, "0", start_time=2000)

        result = mlflow_registry.resolve_run_id(client, run_id=None, experiment_id="0")

        assert result == newer.info.run_id
        assert result != older.info.run_id

    def test_raises_when_the_experiment_has_no_runs(self, client):
        empty_experiment_id = client.create_experiment("empty-experiment")

        with pytest.raises(ValueError, match="empty-experiment|no runs"):
            mlflow_registry.resolve_run_id(client, run_id=None, experiment_id=empty_experiment_id)


class TestFetchRunMetrics:
    def test_returns_the_final_logged_value_per_key(self, client):
        run = _create_run_with_metric(
            client, "0", start_time=1000, key="val_accuracy", values=[0.5, 0.7, 0.9]
        )

        metrics = mlflow_registry.fetch_run_metrics(client, run.info.run_id)

        assert metrics["val_accuracy"] == 0.9


class TestFetchMetricHistory:
    def test_returns_values_in_step_order(self, client):
        run = _create_run_with_metric(
            client, "0", start_time=1000, key="val_loss", values=[0.9, 0.5, 0.3]
        )

        history = mlflow_registry.fetch_metric_history(client, run.info.run_id, "val_loss")

        assert history == [0.9, 0.5, 0.3]


class TestResolveStageVersion:
    def _run_with_model_artifact(self, client, tmp_path, name="dummy.txt"):
        run = client.create_run("0", start_time=1000)
        artifact_file = tmp_path / name
        artifact_file.write_text("not a real model, just exercising the registry API")
        client.log_artifact(run.info.run_id, str(artifact_file), artifact_path="model")
        return run

    def test_returns_the_version_currently_at_the_given_stage(self, client, tmp_path):
        run = self._run_with_model_artifact(client, tmp_path)
        registered = mlflow_registry.register_and_promote(
            client, run_id=run.info.run_id, model_name="starcop-baseline-mag1c-rgb", stage="Staging"
        )

        result = mlflow_registry.resolve_stage_version(
            client, "starcop-baseline-mag1c-rgb", "Staging"
        )

        assert result.version == registered.version
        assert result.current_stage == "Staging"

    def test_raises_when_no_version_is_at_the_given_stage(self, client, tmp_path):
        run = self._run_with_model_artifact(client, tmp_path)
        mlflow_registry.register_and_promote(
            client, run_id=run.info.run_id, model_name="starcop-baseline-mag1c-rgb", stage="Staging"
        )

        with pytest.raises(ValueError, match="starcop-baseline-mag1c-rgb|Production"):
            mlflow_registry.resolve_stage_version(
                client, "starcop-baseline-mag1c-rgb", "Production"
            )

    def test_raises_when_the_registered_model_does_not_exist_at_all(self, client):
        with pytest.raises(ValueError, match="does-not-exist"):
            mlflow_registry.resolve_stage_version(client, "does-not-exist", "Staging")

    def test_propagates_other_mlflow_exceptions_unchanged(self, client, tmp_path):
        # The model exists, but the stage name is invalid -- a real MLflow
        # error distinct from "model doesn't exist" (error_code
        # INVALID_PARAMETER_VALUE, not RESOURCE_DOES_NOT_EXIST). Confirms
        # the bare `raise` re-raises anything that isn't specifically a
        # missing-model error, rather than swallowing or misclassifying it.
        run = self._run_with_model_artifact(client, tmp_path)
        mlflow_registry.register_and_promote(
            client, run_id=run.info.run_id, model_name="starcop-baseline-mag1c-rgb", stage="Staging"
        )

        with pytest.raises(MlflowException) as exc_info:
            mlflow_registry.resolve_stage_version(
                client, "starcop-baseline-mag1c-rgb", "NotARealStage"
            )
        assert exc_info.value.error_code == "INVALID_PARAMETER_VALUE"

    def test_returns_the_latest_version_when_a_stage_has_more_than_one(self, client, tmp_path):
        run_a = self._run_with_model_artifact(client, tmp_path, name="a.txt")
        run_b = self._run_with_model_artifact(client, tmp_path, name="b.txt")
        mlflow_registry.register_and_promote(
            client,
            run_id=run_a.info.run_id,
            model_name="starcop-baseline-mag1c-rgb",
            stage="Staging",
        )
        second = mlflow_registry.register_and_promote(
            client,
            run_id=run_b.info.run_id,
            model_name="starcop-baseline-mag1c-rgb",
            stage="Staging",
        )

        result = mlflow_registry.resolve_stage_version(
            client, "starcop-baseline-mag1c-rgb", "Staging"
        )

        assert result.version == second.version


class TestRegisterAndPromote:
    def _run_with_model_artifact(self, client, tmp_path):
        run = client.create_run("0", start_time=1000)
        artifact_file = tmp_path / "dummy_model_file.txt"
        artifact_file.write_text("not a real model, just exercising the registry API")
        client.log_artifact(run.info.run_id, str(artifact_file), artifact_path="model")
        return run

    def test_creates_a_new_registered_model_and_version_at_the_given_stage(self, client, tmp_path):
        run = self._run_with_model_artifact(client, tmp_path)

        model_version = mlflow_registry.register_and_promote(
            client, run_id=run.info.run_id, model_name="methane-cnn-starcop", stage="Staging"
        )

        assert model_version.current_stage == "Staging"
        fetched = client.get_model_version("methane-cnn-starcop", model_version.version)
        assert fetched.run_id == run.info.run_id

    def test_reuses_an_existing_registered_model_for_a_second_promotion(self, client, tmp_path):
        run_a = self._run_with_model_artifact(client, tmp_path)
        run_b = self._run_with_model_artifact(client, tmp_path)

        first = mlflow_registry.register_and_promote(
            client, run_id=run_a.info.run_id, model_name="methane-cnn-starcop", stage="Staging"
        )
        second = mlflow_registry.register_and_promote(
            client, run_id=run_b.info.run_id, model_name="methane-cnn-starcop", stage="Production"
        )

        assert int(second.version) == int(first.version) + 1
        assert second.current_stage == "Production"

    def test_calling_it_twice_for_the_same_run_reuses_the_version_instead_of_duplicating(
        self, client, tmp_path
    ):
        run = self._run_with_model_artifact(client, tmp_path)

        first = mlflow_registry.register_and_promote(
            client, run_id=run.info.run_id, model_name="methane-cnn-starcop", stage="Staging"
        )
        second = mlflow_registry.register_and_promote(
            client, run_id=run.info.run_id, model_name="methane-cnn-starcop", stage="Production"
        )

        assert second.version == first.version
        assert second.current_stage == "Production"
        assert len(client.search_model_versions("name='methane-cnn-starcop'")) == 1

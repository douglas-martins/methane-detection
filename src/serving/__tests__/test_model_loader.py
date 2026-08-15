"""Tests for src/serving/model_loader.py.

resolve_model_uri is pure -- unit tested directly (Test Size: Small).

load_model is MLflow SDK glue, tested against a real local sqlite tracking
store (Test Size: Medium, mirrors src/registry/__tests__/test_mlflow_registry.py's
own precedent) with a tiny real torch.nn.Module standing in for the model --
no Mock(). This proves the real wiring end to end: resolve_stage_version ->
models:/<name>/<version> URI -> mlflow.pytorch.load_model, all under
Environment B. Loading an actual STARCOP checkpoint (which additionally
needs vendor/starcop importable at unpickle time, handled by this module's
_vendor_starcop_serving import) is SDK/network glue validated by an actual
run against the live server instead, same Test Size: Large boundary
src/registry/hf_baseline_import.py's own load_model/import_variant are
validated at.
"""

import mlflow.pytorch
import model_loader
import pytest
import torch
from mlflow.tracking import MlflowClient
from torch import nn


class TestResolveModelUri:
    def test_builds_a_runs_scheme_uri_from_run_id_and_default_artifact_path(self):
        assert model_loader.resolve_model_uri("abc123") == "runs:/abc123/model"

    def test_accepts_a_custom_artifact_path(self):
        assert model_loader.resolve_model_uri("abc123", artifact_path="checkpoint") == (
            "runs:/abc123/checkpoint"
        )


@pytest.fixture
def tracking_uri(tmp_path):
    return f"sqlite:///{tmp_path}/mlflow.db"


class TestLoadModel:
    def _log_and_promote(self, tracking_uri, model_name, stage):
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri=tracking_uri)
        real_model = nn.Linear(4, 1)

        with mlflow.start_run() as run:
            # serialization_format="pickle": mlflow's default for torch>=2.4
            # is "pt2" (torch.export tracing), which requires an
            # input_example -- and wouldn't work for STARCOP's real
            # LightningModule anyway (data-dependent control flow isn't
            # export-traceable). Matches how the real registered models
            # were actually saved (verified against the live server).
            mlflow.pytorch.log_model(
                real_model, artifact_path="model", serialization_format="pickle"
            )

        client.create_registered_model(model_name)
        version = client.create_model_version(
            name=model_name, source=f"runs:/{run.info.run_id}/model", run_id=run.info.run_id
        )
        client.transition_model_version_stage(name=model_name, version=version.version, stage=stage)
        return real_model

    def test_loads_the_version_currently_at_the_given_stage(self, tracking_uri):
        self._log_and_promote(tracking_uri, "test-model", "Staging")

        model, version = model_loader.load_model(tracking_uri, "test-model", "Staging")

        assert isinstance(model, nn.Linear)
        assert version.current_stage == "Staging"

    def test_loaded_model_is_in_eval_mode(self, tracking_uri):
        self._log_and_promote(tracking_uri, "test-model", "Staging")

        model, _ = model_loader.load_model(tracking_uri, "test-model", "Staging")

        assert model.training is False

    def test_loaded_model_produces_the_same_output_as_the_original(self, tracking_uri):
        original = self._log_and_promote(tracking_uri, "test-model", "Staging")
        original.eval()
        x = torch.rand(1, 4)

        model, _ = model_loader.load_model(tracking_uri, "test-model", "Staging")

        with torch.no_grad():
            assert torch.equal(model(x), original(x))

    def test_raises_when_no_version_is_at_the_requested_stage(self, tracking_uri):
        self._log_and_promote(tracking_uri, "test-model", "Staging")

        with pytest.raises(ValueError, match="test-model|Production"):
            model_loader.load_model(tracking_uri, "test-model", "Production")

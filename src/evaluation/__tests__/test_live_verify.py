"""Tests for src/evaluation/live_verify.py -- Phase 5's live-vs-offline
prediction diff (Test Size: Small for pure comparison/parsing logic,
Medium for resolve_paper_eval_run against a real sqlite MlflowClient, same
convention as test_paper_eval_mlflow.py::TestCheckRegistryVersionMatches).
verify_variant itself (the HTTP round-trip against a real `bentoml serve`
process) is Large-boundary, real-run validated, not unit tested -- same
pattern as evaluate_variant()/import_variant(). See
track-a-paper-benchmark-reproduction-plan.md Phase 5.
"""

import json

import live_verify
import numpy as np
import pytest
from mlflow.tracking import MlflowClient


class TestMaskSha256:
    def test_matches_run_baseline_evals_own_digest_recipe(self):
        # Phase 5's own resolved-details note: live_verify.py's digest must
        # be bit-for-bit comparable against run_baseline_eval.py's
        # _mask_digest -- same recipe, independently re-derived here rather
        # than imported, since live_verify.py compares a JSON-round-tripped
        # (nested list) mask, not a numpy array straight out of the model.
        import hashlib

        mask = [[0, 1], [1, 0]]
        expected = hashlib.sha256(
            np.ascontiguousarray(np.array(mask, dtype=np.int64)).tobytes()
        ).hexdigest()

        assert live_verify.mask_sha256(mask) == expected

    def test_accepts_a_plain_nested_list_not_just_a_numpy_array(self):
        as_list = live_verify.mask_sha256([[0, 1], [1, 1]])
        as_array = live_verify.mask_sha256(np.array([[0, 1], [1, 1]], dtype=np.int64))

        assert as_list == as_array

    def test_differs_for_different_masks(self):
        assert live_verify.mask_sha256([[0, 1]]) != live_verify.mask_sha256([[1, 1]])


class TestComparePrediction:
    def _offline(self, scene_id="scene_a", mask_sha256="deadbeef", confidence=None):
        return {
            "scene_id": scene_id,
            "mask_sha256": mask_sha256,
            "confidence": confidence if confidence is not None else [[0.1, 0.9]],
        }

    def test_passes_when_mask_digest_and_confidence_both_match(self):
        offline = self._offline(
            mask_sha256=live_verify.mask_sha256([[0, 1]]), confidence=[[0.1, 0.9]]
        )
        live = {"mask": [[0, 1]], "confidence": [[0.1, 0.9]]}

        result = live_verify.compare_prediction(offline, live)

        assert result == {
            "scene_id": "scene_a",
            "mask_match": True,
            "confidence_match": True,
            "passed": True,
            "live_mask_sha256": offline["mask_sha256"],
            "offline_mask_sha256": offline["mask_sha256"],
        }

    def test_fails_when_mask_digest_differs_even_if_confidence_matches(self):
        offline = self._offline(mask_sha256="not-the-real-digest", confidence=[[0.1, 0.9]])
        live = {"mask": [[0, 1]], "confidence": [[0.1, 0.9]]}

        result = live_verify.compare_prediction(offline, live)

        assert result["mask_match"] is False
        assert result["passed"] is False

    def test_fails_when_confidence_differs_beyond_tolerance_even_if_mask_matches(self):
        offline = self._offline(
            mask_sha256=live_verify.mask_sha256([[0, 1]]), confidence=[[0.1, 0.9]]
        )
        live = {"mask": [[0, 1]], "confidence": [[0.5, 0.9]]}

        result = live_verify.compare_prediction(offline, live)

        assert result["confidence_match"] is False
        assert result["passed"] is False

    def test_passes_when_confidence_differs_within_tolerance(self):
        offline = self._offline(
            mask_sha256=live_verify.mask_sha256([[0, 1]]), confidence=[[0.1, 0.9]]
        )
        live = {"mask": [[0, 1]], "confidence": [[0.1 + 1e-7, 0.9]]}

        result = live_verify.compare_prediction(offline, live)

        assert result["confidence_match"] is True
        assert result["passed"] is True

    def test_respects_a_custom_atol(self):
        offline = self._offline(
            mask_sha256=live_verify.mask_sha256([[0, 1]]), confidence=[[0.1, 0.9]]
        )
        live = {"mask": [[0, 1]], "confidence": [[0.11, 0.9]]}

        assert live_verify.compare_prediction(offline, live, atol=1e-5)["confidence_match"] is False
        assert live_verify.compare_prediction(offline, live, atol=0.02)["confidence_match"] is True


class TestAssertModelIdentity:
    def _health(self, model_name="starcop-baseline-mag1c-only", model_version="2"):
        return {"model_name": model_name, "model_version": model_version}

    def test_passes_silently_when_name_and_version_both_match(self):
        live_verify.assert_model_identity(
            self._health(),
            expected_model_name="starcop-baseline-mag1c-only",
            expected_registry_version="2",
        )

    def test_passes_when_version_types_differ_but_values_are_equal(self):
        live_verify.assert_model_identity(
            self._health(model_version=2),
            expected_model_name="starcop-baseline-mag1c-only",
            expected_registry_version="2",
        )

    def test_raises_when_versions_differ_even_if_the_name_matches(self):
        with pytest.raises(live_verify.ModelIdentityMismatch, match="mag1c-only.*3.*mag1c-only.*2"):
            live_verify.assert_model_identity(
                self._health(model_version="3"),
                expected_model_name="starcop-baseline-mag1c-only",
                expected_registry_version="2",
            )

    def test_raises_when_the_name_differs_even_if_the_version_number_matches(self):
        # The real bug this guards: mag1c_only and mag1c_rgb can independently
        # sit at the identical registry version number (each model name has
        # its own version counter) -- comparing the version alone would
        # silently pass while verifying against the wrong served model.
        with pytest.raises(live_verify.ModelIdentityMismatch, match="mag1c-only.*mag1c-rgb"):
            live_verify.assert_model_identity(
                self._health(model_name="starcop-baseline-mag1c-only", model_version="2"),
                expected_model_name="starcop-baseline-mag1c-rgb",
                expected_registry_version="2",
            )

    def test_raises_when_both_name_and_version_differ(self):
        with pytest.raises(live_verify.ModelIdentityMismatch):
            live_verify.assert_model_identity(
                self._health(model_name="starcop-baseline-mag1c-only", model_version="1"),
                expected_model_name="starcop-baseline-mag1c-rgb",
                expected_registry_version="2",
            )


class TestRejectOutOfScopeVariant:
    def test_passes_silently_for_mag1c_only(self):
        live_verify.assert_variant_is_servable("mag1c_only")

    def test_passes_silently_for_mag1c_rgb(self):
        live_verify.assert_variant_is_servable("mag1c_rgb")

    def test_raises_for_varon(self):
        with pytest.raises(ValueError, match="varon"):
            live_verify.assert_variant_is_servable("varon")


class TestParseOfflinePredictionsDir:
    def test_returns_records_keyed_by_scene_id(self, tmp_path):
        (tmp_path / "scene_a.json").write_text(
            json.dumps({"scene_id": "scene_a", "mask_sha256": "aaa"})
        )
        (tmp_path / "scene_b.json").write_text(
            json.dumps({"scene_id": "scene_b", "mask_sha256": "bbb"})
        )

        result = live_verify.parse_offline_predictions_dir(tmp_path)

        assert set(result.keys()) == {"scene_a", "scene_b"}
        assert result["scene_a"]["mask_sha256"] == "aaa"

    def test_returns_empty_dict_for_an_empty_directory(self, tmp_path):
        assert live_verify.parse_offline_predictions_dir(tmp_path) == {}

    def test_ignores_non_json_files(self, tmp_path):
        (tmp_path / "scene_a.json").write_text(
            json.dumps({"scene_id": "scene_a", "mask_sha256": "aaa"})
        )
        (tmp_path / "notes.txt").write_text("not json")

        result = live_verify.parse_offline_predictions_dir(tmp_path)

        assert list(result.keys()) == ["scene_a"]


class TestArrayToNpyBytes:
    def test_round_trips_through_np_load(self):
        import io

        array = np.arange(12, dtype=np.float32).reshape(3, 4)

        npy_bytes = live_verify.array_to_npy_bytes(array)
        reloaded = np.load(io.BytesIO(npy_bytes))

        assert np.array_equal(reloaded, array)
        assert reloaded.dtype == array.dtype


class TestResolvePaperEvalRun:
    """Real sqlite MlflowClient, same convention as
    test_paper_eval_mlflow.py::TestCheckRegistryVersionMatches."""

    @pytest.fixture
    def client(self, tmp_path):
        return MlflowClient(tracking_uri=f"sqlite:///{tmp_path}/mlflow.db")

    def _log_run(self, client, experiment_name, variant, registry_version="2", start_time=None):
        experiment = client.get_experiment_by_name(experiment_name)
        experiment_id = (
            experiment.experiment_id if experiment else client.create_experiment(experiment_name)
        )
        run = client.create_run(experiment_id, start_time=start_time)
        client.set_tag(run.info.run_id, "variant", variant)
        client.set_tag(run.info.run_id, "registry_version", registry_version)
        return run

    def test_returns_the_single_run_for_a_variant(self, client):
        run = self._log_run(client, "starcop-paper-eval", "mag1c_only")
        self._log_run(client, "starcop-paper-eval", "mag1c_rgb")

        result = live_verify.resolve_paper_eval_run(client, "mag1c_only")

        assert result.info.run_id == run.info.run_id

    def test_raises_when_no_run_exists_for_the_variant(self, client):
        self._log_run(client, "starcop-paper-eval", "mag1c_rgb")

        with pytest.raises(ValueError, match="mag1c_only"):
            live_verify.resolve_paper_eval_run(client, "mag1c_only")

    def test_raises_when_the_experiment_does_not_exist(self, client):
        with pytest.raises(ValueError, match="starcop-paper-eval"):
            live_verify.resolve_paper_eval_run(client, "mag1c_only")

    def test_picks_the_most_recent_run_when_several_exist_for_the_same_variant(self, client):
        # Phase 4's flow is explicitly repeatable -- runs accumulate over
        # time in starcop-paper-eval as the permanent audit trail Phase 3
        # was built for (docs, dataset snapshot, or eval code changing all
        # trigger a fresh flow run). Requiring exactly one run to ever
        # exist (this function's original design) broke on every re-run
        # after the first -- caught live 2026-08-22 when a real flow
        # execution's own evaluate step created new runs alongside earlier
        # ones from prior attempts, and the live-check step silently
        # degraded to "not_run" instead of erroring loudly. Old runs are
        # not stale data to clean up; picking the newest is correct.
        older = self._log_run(client, "starcop-paper-eval", "mag1c_only", start_time=1000)
        newer = self._log_run(client, "starcop-paper-eval", "mag1c_only", start_time=2000)

        result = live_verify.resolve_paper_eval_run(client, "mag1c_only")

        assert result.info.run_id == newer.info.run_id
        assert result.info.run_id != older.info.run_id

    def test_still_finds_the_only_run_when_just_one_exists(self, client):
        run = self._log_run(client, "starcop-paper-eval", "mag1c_only")

        result = live_verify.resolve_paper_eval_run(client, "mag1c_only")

        assert result.info.run_id == run.info.run_id

    def test_ignores_soft_deleted_runs(self, client):
        stale = self._log_run(client, "starcop-paper-eval", "mag1c_only")
        current = self._log_run(client, "starcop-paper-eval", "mag1c_only")
        client.delete_run(stale.info.run_id)

        result = live_verify.resolve_paper_eval_run(client, "mag1c_only")

        assert result.info.run_id == current.info.run_id

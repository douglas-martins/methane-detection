"""Tests for src/evaluation/run_baseline_eval.py's pure/injectable-boundary
logic (Test Size: Small). The checkpoint-loading + real dataloader +
run_validation orchestration itself is thin glue, exercised by the real
Phase 0/1 run rather than a unit test -- same pattern as
starcop_datamodule.py's own untested glue.
"""

import json

import numpy as np
import pandas as pd
import pytest
import run_baseline_eval as rbe
import torch


def _test_df(rows: dict) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "id"
    return df


class TestLoadAndPlaceModel:
    def test_moves_the_loaded_model_to_the_requested_device(self):
        fake_model = torch.nn.Linear(1, 1)

        def fake_load_model(checkpoint_path):
            return fake_model, {"settings": "sentinel"}

        model, settings = rbe.load_and_place_model(
            "unused/path.ckpt", device=torch.device("cpu"), load_model_fn=fake_load_model
        )

        assert next(model.parameters()).device == torch.device("cpu")
        assert settings == {"settings": "sentinel"}

    def test_returns_the_settings_from_load_model_fn_unchanged(self):
        def fake_load_model(checkpoint_path):
            return torch.nn.Linear(1, 1), {"model_mode": "segmentation_output"}

        _, settings = rbe.load_and_place_model(
            "unused/path.ckpt", device=torch.device("cpu"), load_model_fn=fake_load_model
        )

        assert settings == {"model_mode": "segmentation_output"}


class TestRunValidationSafely:
    def test_returns_run_validation_fns_result_on_success(self):
        def fake_run_validation(model, dataloader, **kwargs):
            return "out_data_sentinel", {"metrics": "sentinel"}

        result = rbe.run_validation_safely(
            model=None, dataloader=None, run_validation_fn=fake_run_validation
        )

        assert result == ("out_data_sentinel", {"metrics": "sentinel"})

    def test_raises_known_difficulty_bucket_gap_error_on_keyerror(self):
        def fake_run_validation(model, dataloader, **kwargs):
            raise KeyError((False, "hard"))

        with pytest.raises(rbe.KnownDifficultyBucketGapError, match="difficulty"):
            rbe.run_validation_safely(
                model=None, dataloader=None, run_validation_fn=fake_run_validation
            )

    def test_does_not_swallow_other_exception_types(self):
        def fake_run_validation(model, dataloader, **kwargs):
            raise RuntimeError("a real device/environment failure")

        with pytest.raises(RuntimeError, match="a real device/environment failure"):
            rbe.run_validation_safely(
                model=None, dataloader=None, run_validation_fn=fake_run_validation
            )

    def test_forwards_kwargs_to_run_validation_fn(self):
        received = {}

        def fake_run_validation(model, dataloader, **kwargs):
            received.update(kwargs)
            return None, {}

        rbe.run_validation_safely(
            model=None,
            dataloader=None,
            run_validation_fn=fake_run_validation,
            products_plot=["mag1c"],
        )

        assert received == {"products_plot": ["mag1c"]}


class TestSelectLimitSceneIds:
    def test_includes_at_least_one_no_plume_scene_when_available(self):
        test_df = _test_df(
            {
                "plume1": {"has_plume": True, "qplume": 1500.0},
                "noplume1": {"has_plume": False, "qplume": 0.0},
            }
        )

        selected = rbe.select_limit_scene_ids(test_df, limit=1)

        assert selected == ["noplume1"]

    def test_includes_the_highest_qplume_plume_scene(self):
        test_df = _test_df(
            {
                "noplume1": {"has_plume": False, "qplume": 0.0},
                "plume_low": {"has_plume": True, "qplume": 200.0},
                "plume_high": {"has_plume": True, "qplume": 3000.0},
            }
        )

        selected = rbe.select_limit_scene_ids(test_df, limit=2)

        assert "plume_high" in selected

    def test_includes_the_lowest_qplume_plume_scene_when_limit_allows(self):
        test_df = _test_df(
            {
                "noplume1": {"has_plume": False, "qplume": 0.0},
                "plume_low": {"has_plume": True, "qplume": 200.0},
                "plume_mid": {"has_plume": True, "qplume": 1200.0},
                "plume_high": {"has_plume": True, "qplume": 3000.0},
            }
        )

        selected = rbe.select_limit_scene_ids(test_df, limit=3)

        assert "plume_low" in selected
        assert "plume_high" in selected

    def test_never_returns_more_than_limit_ids(self):
        test_df = _test_df(
            {f"scene{i}": {"has_plume": i % 2 == 0, "qplume": float(i * 100)} for i in range(10)}
        )

        selected = rbe.select_limit_scene_ids(test_df, limit=3)

        assert len(selected) == 3

    def test_returns_empty_list_for_limit_zero(self):
        test_df = _test_df({"plume1": {"has_plume": True, "qplume": 1500.0}})

        assert rbe.select_limit_scene_ids(test_df, limit=0) == []

    def test_returns_fewer_than_limit_when_pool_is_smaller(self):
        test_df = _test_df({"plume1": {"has_plume": True, "qplume": 1500.0}})

        selected = rbe.select_limit_scene_ids(test_df, limit=5)

        assert selected == ["plume1"]

    def test_selected_ids_are_unique(self):
        test_df = _test_df(
            {f"scene{i}": {"has_plume": i % 2 == 0, "qplume": float(i * 100)} for i in range(6)}
        )

        selected = rbe.select_limit_scene_ids(test_df, limit=6)

        assert len(selected) == len(set(selected))


class TestAssertKnownSceneCounts:
    def test_passes_silently_when_counts_match(self):
        rows = {}
        for i in range(57):
            rows[f"strong{i}"] = {"has_plume": True, "qplume": 1000.0}
        for i in range(109):
            rows[f"weak{i}"] = {"has_plume": True, "qplume": 999.0}
        for i in range(176):
            rows[f"noplume{i}"] = {"has_plume": False, "qplume": 0.0}
        test_df = _test_df(rows)

        rbe.assert_known_scene_counts(test_df)  # must not raise

    def test_raises_when_total_count_is_wrong(self):
        test_df = _test_df({"plume1": {"has_plume": True, "qplume": 1500.0}})

        with pytest.raises(AssertionError, match="scene counts"):
            rbe.assert_known_scene_counts(test_df)

    def test_raises_when_strong_weak_split_is_wrong(self):
        rows = {f"scene{i}": {"has_plume": True, "qplume": 1500.0} for i in range(342)}
        test_df = _test_df(rows)

        with pytest.raises(AssertionError, match="scene counts"):
            rbe.assert_known_scene_counts(test_df)


class TestDeriveFeaturesExtract:
    """Detects which of a checkpoint's own input_products are computed
    features (e.g. MultiSTARCOP/varon's Varon ratio bands) rather than raw
    satellite bands, keyed off vendor/starcop's own real
    feature_extration.FEATURES table -- so a variant needing on-the-fly
    feature extraction is found automatically from its own settings, not a
    hand-maintained per-variant list. See
    track-a-paper-benchmark-reproduction-plan.md Phase 2."""

    def test_returns_empty_list_when_no_input_product_is_a_computed_feature(self):
        # HyperSTARCOP's raw mag1c/RGB band names aren't in FEATURES.
        assert rbe.derive_features_extract(["mag1c", "TOA_AVIRIS_640nm"]) == []

    def test_returns_the_computed_feature_subset_for_varons_input_products(self):
        varon_products = [
            "ratio_wv3_B7_B5_varon21_sum_c_out",
            "ratio_wv3_B8_B5_varon21_sum_c_out",
            "ratio_wv3_B7_B6_varon21_sum_c_out",
        ]

        assert rbe.derive_features_extract(varon_products) == varon_products

    def test_excludes_non_feature_products_from_a_mixed_list(self):
        result = rbe.derive_features_extract(["mag1c", "ratio_wv3_B7_B5_varon21_sum_c_out"])

        assert result == ["ratio_wv3_B7_B5_varon21_sum_c_out"]


class TestValidateCliArgs:
    def test_passes_silently_when_only_limit_is_set(self):
        rbe.validate_cli_args(limit=5, emit_docs_assets=None)

    def test_passes_silently_when_only_emit_docs_assets_is_set(self):
        rbe.validate_cli_args(limit=None, emit_docs_assets="/tmp/staging")

    def test_passes_silently_when_neither_is_set(self):
        rbe.validate_cli_args(limit=None, emit_docs_assets=None)

    def test_raises_when_both_are_set(self):
        with pytest.raises(ValueError, match="--limit"):
            rbe.validate_cli_args(limit=5, emit_docs_assets="/tmp/staging")


class TestMaskDigest:
    def test_is_deterministic_for_the_same_mask(self):
        mask = np.array([[0, 1], [1, 0]], dtype=np.int64)

        assert rbe._mask_digest(mask) == rbe._mask_digest(mask.copy())

    def test_differs_for_different_masks(self):
        mask_a = np.array([[0, 1], [1, 0]], dtype=np.int64)
        mask_b = np.array([[1, 1], [1, 0]], dtype=np.int64)

        assert rbe._mask_digest(mask_a) != rbe._mask_digest(mask_b)

    def test_is_stable_across_non_contiguous_views(self):
        # A transposed array shares the same underlying buffer in a
        # different memory layout -- np.ascontiguousarray inside
        # _mask_digest must normalize this before hashing, or a live/offline
        # comparison could spuriously fail on layout alone, not content.
        base = np.array([[0, 1], [1, 0]], dtype=np.int64)
        transposed_view = base.T

        assert rbe._mask_digest(transposed_view) == rbe._mask_digest(
            np.ascontiguousarray(transposed_view)
        )


class TestPersistOfflinePredictions:
    def test_writes_one_json_file_per_curated_scene_prefixed_by_variant(self, tmp_path):
        # Real bug, caught live 2026-08-22: all three variants share one
        # staging_dir within a single eval_baseline flow run
        # (run_eval_baseline_cycle's single tempfile.TemporaryDirectory()
        # wraps the whole evaluate loop). Without a variant prefix, a later
        # variant's evaluate_variant() call uploads *every* file sitting in
        # that shared offline_predictions/ folder as its own MLflow
        # artifact -- including an earlier variant's leftover scene JSONs.
        # Confirmed for real: mag1c_only's uploaded offline_predictions/
        # had 8 files (4 its own + 4 leftover from varon, which ran
        # first), mag1c_rgb's had 12 (its own 4 + varon's 4 + mag1c_only's
        # 4). Phase 5's live_verify.py then compared the wrong variant's
        # stored predictions against the live server for those scene ids
        # and correctly reported a mismatch -- a real false failure, not a
        # flaky test.
        curated = {
            "scene_a": {
                "mask": np.array([[0, 1]], dtype=np.int64),
                "confidence": np.array([[0.1, 0.9]]),
            },
            "scene_b": {
                "mask": np.array([[1, 0]], dtype=np.int64),
                "confidence": np.array([[0.8, 0.2]]),
            },
        }

        rbe.persist_offline_predictions(curated, tmp_path, variant="mag1c_only")

        assert sorted(p.name for p in tmp_path.glob("*.json")) == [
            "mag1c_only_scene_a.json",
            "mag1c_only_scene_b.json",
        ]

    def test_each_record_has_scene_id_mask_digest_and_confidence(self, tmp_path):
        mask = np.array([[0, 1], [1, 1]], dtype=np.int64)
        confidence = np.array([[0.1, 0.9], [0.8, 0.7]])
        curated = {"scene_a": {"mask": mask, "confidence": confidence}}

        rbe.persist_offline_predictions(curated, tmp_path, variant="varon")

        record = json.loads((tmp_path / "varon_scene_a.json").read_text())
        assert record == {
            "scene_id": "scene_a",
            "mask_sha256": rbe._mask_digest(mask),
            "confidence": confidence.tolist(),
        }

    def test_creates_the_output_directory_if_missing(self, tmp_path):
        output_dir = tmp_path / "nested" / "offline_predictions"
        curated = {"scene_a": {"mask": np.zeros((1, 1)), "confidence": np.zeros((1, 1))}}

        rbe.persist_offline_predictions(curated, output_dir, variant="mag1c_rgb")

        assert (output_dir / "mag1c_rgb_scene_a.json").exists()

    def test_different_variants_writing_to_the_same_directory_never_collide(self, tmp_path):
        # The exact scenario that broke live: two variants happening to
        # pick the same curated scene_id, sharing one directory.
        shared_scene = {
            "ang20191018t141549_r11008_c0_w512_h512": {
                "mask": np.array([[0, 1]], dtype=np.int64),
                "confidence": np.array([[0.1, 0.9]]),
            }
        }

        rbe.persist_offline_predictions(shared_scene, tmp_path, variant="varon")
        rbe.persist_offline_predictions(shared_scene, tmp_path, variant="mag1c_only")

        assert sorted(p.name for p in tmp_path.glob("*.json")) == [
            "mag1c_only_ang20191018t141549_r11008_c0_w512_h512.json",
            "varon_ang20191018t141549_r11008_c0_w512_h512.json",
        ]

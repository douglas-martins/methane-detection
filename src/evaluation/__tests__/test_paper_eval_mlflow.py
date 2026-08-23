"""Tests for src/evaluation/paper_eval_mlflow.py -- Phase 3's tag/metric/
comparison builders (Test Size: Small, pure logic) plus
check_registry_version_matches (Test Size: Medium, real sqlite MlflowClient,
same convention as test_mlflow_registry.py -- never mocked). log_paper_eval_run
itself is thin SDK glue, Large-boundary, validated by a real run instead of a
unit test -- same pattern as hf_baseline_import.import_variant(). See
track-a-paper-benchmark-reproduction-plan.md Phase 3.
"""

import mlflow_registry
import paper_eval_mlflow
import pytest
from mlflow.tracking import MlflowClient


class TestIsGitDirty:
    def test_false_when_git_status_output_is_empty(self):
        assert (
            paper_eval_mlflow.is_git_dirty("unused/root", run_git_status_fn=lambda root: "")
            is False
        )

    def test_true_when_git_status_has_output(self):
        result = paper_eval_mlflow.is_git_dirty(
            "unused/root", run_git_status_fn=lambda root: " M some/file.py\n"
        )

        assert result is True

    def test_passes_repo_root_through_to_the_status_fn(self):
        received = []

        def fake_status(root):
            received.append(root)
            return ""

        paper_eval_mlflow.is_git_dirty("the/repo/root", run_git_status_fn=fake_status)

        assert received == ["the/repo/root"]


class TestGitSubmoduleSha:
    def test_returns_the_rev_parse_fns_result(self):
        result = paper_eval_mlflow.git_submodule_sha(
            "vendor/starcop", run_git_rev_parse_fn=lambda path: "abc123deadbeef"
        )

        assert result == "abc123deadbeef"

    def test_passes_submodule_dir_through(self):
        received = []

        paper_eval_mlflow.git_submodule_sha("vendor/starcop", run_git_rev_parse_fn=received.append)

        assert received == ["vendor/starcop"]


class TestDvcTrackedDirHash:
    def test_reads_the_md5_from_a_dvc_pointer_file(self, tmp_path):
        dvc_file = tmp_path / "starcop_raw.dvc"
        dvc_file.write_text(
            "outs:\n"
            "- md5: e11b16a61ddf235613701fbece9b59d6.dir\n"
            "  size: 63296760557\n"
            "  nfiles: 75353\n"
            "  hash: md5\n"
            "  path: starcop_raw\n"
        )

        result = paper_eval_mlflow.dvc_tracked_dir_hash(dvc_file)

        assert result == "e11b16a61ddf235613701fbece9b59d6.dir"

    def test_raises_when_the_dvc_file_has_no_outs(self, tmp_path):
        dvc_file = tmp_path / "empty.dvc"
        dvc_file.write_text("outs: []\n")

        with pytest.raises(ValueError, match="no outs|No outs"):
            paper_eval_mlflow.dvc_tracked_dir_hash(dvc_file)


class TestDependencyManifest:
    def test_returns_the_freeze_fns_result(self):
        result = paper_eval_mlflow.dependency_manifest(
            "unused/python", run_freeze_fn=lambda python: "numpy==1.2.3\n"
        )

        assert result == "numpy==1.2.3\n"

    def test_passes_python_executable_through(self):
        received = []

        paper_eval_mlflow.dependency_manifest("path/to/python", run_freeze_fn=received.append)

        assert received == ["path/to/python"]


class TestResolveUvBinary:
    """Real bug, caught live 2026-08-22: the eval-baseline Prefect flow's
    launchd-supervised worker process gets launchd's own minimal default
    PATH (/usr/bin:/bin:/usr/sbin:/sbin, confirmed via `launchctl print` --
    no ~/.local/bin), so a bare "uv" PATH lookup raised FileNotFoundError
    even though uv is installed and used successfully everywhere else in
    this project (interactive shells always have the fuller PATH)."""

    def test_returns_the_which_result_when_uv_is_on_path(self):
        result = paper_eval_mlflow.resolve_uv_binary(which_fn=lambda name: "/usr/local/bin/uv")

        assert result == "/usr/local/bin/uv"

    def test_falls_back_to_the_home_local_bin_install_when_which_fails(self, tmp_path, monkeypatch):
        fallback = tmp_path / ".local" / "bin" / "uv"
        fallback.parent.mkdir(parents=True)
        fallback.touch()
        monkeypatch.setattr(paper_eval_mlflow.Path, "home", lambda: tmp_path)

        result = paper_eval_mlflow.resolve_uv_binary(which_fn=lambda name: None)

        assert result == str(fallback)

    def test_raises_a_clear_error_when_neither_resolves(self, tmp_path, monkeypatch):
        monkeypatch.setattr(paper_eval_mlflow.Path, "home", lambda: tmp_path)

        with pytest.raises(RuntimeError, match="uv"):
            paper_eval_mlflow.resolve_uv_binary(which_fn=lambda name: None)


class TestResolveGitBinary:
    """Same class of bug as resolve_uv_binary, fixed preemptively for
    consistency: git happens to already be safe under launchd's restricted
    worker PATH (ships at /usr/bin/git on every Mac, which the default
    PATH does include), but that safety was coincidental, not designed --
    see deploy/prefect/README.md's "always resolve explicitly" rule."""

    def test_returns_the_which_result_when_git_is_on_path(self):
        result = paper_eval_mlflow.resolve_git_binary(which_fn=lambda name: "/opt/homebrew/bin/git")

        assert result == "/opt/homebrew/bin/git"

    def test_falls_back_to_usr_bin_git_when_which_fails(self):
        result = paper_eval_mlflow.resolve_git_binary(which_fn=lambda name: None)

        assert result == "/usr/bin/git"


class TestPaperEvalRunName:
    def test_builds_the_expected_run_name(self):
        result = paper_eval_mlflow.paper_eval_run_name("mag1c_rgb", "2026-08-21")

        assert result == "starcop-baseline-mag1c-rgb-paper-eval-2026-08-21"

    def test_varon_kebab_cases_correctly(self):
        result = paper_eval_mlflow.paper_eval_run_name("varon", "2026-08-21")

        assert result == "starcop-baseline-varon-paper-eval-2026-08-21"


class TestBuildPaperEvalTags:
    def test_builds_the_expected_tag_set(self):
        tags = paper_eval_mlflow.build_paper_eval_tags(
            variant="varon",
            registry_version=1,
            checkpoint_sha256="deadbeef",
            dvc_dataset_version="e11b16a6.dir",
            n_test_scenes=342,
            resolved_device="cpu",
            eval_code_dirty=False,
            vendor_starcop_sha="c4789268",
        )

        assert tags == {
            "variant": "varon",
            "registry_model_name": "starcop-baseline-varon",
            "registry_version": "1",
            "checkpoint_sha256": "deadbeef",
            "dvc_dataset_version": "e11b16a6.dir",
            "n_test_scenes": "342",
            "paper_reference": "true",
            "resolved_device": "cpu",
            "eval_code_dirty": "False",
            "vendor_starcop_sha": "c4789268",
        }


class TestPaperEvalMetrics:
    def test_merges_corrected_headline_metrics_with_the_prefixed_full_aggregate(self):
        corrected = {"strong_f1score": 0.289, "auprc": 0.112}
        run_validation_metrics = {"TP": 10, "FP": 2, "thresholded": [{"a": 1}]}

        result = paper_eval_mlflow.paper_eval_metrics(corrected, run_validation_metrics)

        assert result["strong_f1score"] == 0.289
        assert result["auprc"] == 0.112
        assert result["raw_TP"] == 10.0
        assert result["raw_FP"] == 2.0
        assert "raw_thresholded" not in result

    def test_corrected_keys_are_never_shadowed_by_the_raw_prefix(self):
        corrected = {"strong_f1score": 0.5}
        run_validation_metrics = {"strong_f1score": 0.999}

        result = paper_eval_mlflow.paper_eval_metrics(corrected, run_validation_metrics)

        assert result["strong_f1score"] == 0.5
        assert result["raw_strong_f1score"] == 0.999


class TestLoadPaperReferenceMetrics:
    def test_parses_the_fenced_yaml_block(self, tmp_path):
        md_path = tmp_path / "paper_reference_metrics.md"
        md_path.write_text(
            "# doc\n\nSome prose table here.\n\n"
            "```yaml\n"
            "varon:\n"
            '  citation: "Table 1, page 9"\n'
            "  strong_f1score: {mean: 0.3072, std: 0.0287}\n"
            "```\n"
        )

        result = paper_eval_mlflow.load_paper_reference_metrics(md_path)

        assert result["varon"]["citation"] == "Table 1, page 9"
        assert result["varon"]["strong_f1score"] == {"mean": 0.3072, "std": 0.0287}

    def test_raises_when_no_yaml_block_is_present(self, tmp_path):
        md_path = tmp_path / "no_yaml.md"
        md_path.write_text("# doc\n\nJust prose, no fenced block.\n")

        with pytest.raises(ValueError, match="yaml"):
            paper_eval_mlflow.load_paper_reference_metrics(md_path)


class TestRenderPaperComparison:
    def test_renders_a_comparison_table_with_the_citation(self):
        reference = {
            "varon": {
                "citation": "Table 1, page 9 -- row 'Our (Varon)'",
                "strong_f1score": {"mean": 0.3072, "std": 0.0287},
                "weak_f1score": {"mean": 0.1035, "std": 0.0152},
                "no_plume_FPR": {"mean": 0.8789, "std": 0.0467},
                "auprc": {"mean": 0.1192, "std": 0.0135},
            }
        }
        this_run_metrics = {
            "strong_f1score": 0.2894,
            "weak_f1score": 0.1701,
            "no_plume_FPR": 0.8457,
            "auprc": 0.1127,
        }

        rendered = paper_eval_mlflow.render_paper_comparison("varon", this_run_metrics, reference)

        assert "Table 1, page 9" in rendered
        assert "30.72" in rendered
        assert "28.94" in rendered


class TestCollectDocsAssetArtifacts:
    """Pure filesystem-selection logic behind Phase 5's dependency: which
    curated-scene files a docs-assets run wrote that must be uploaded to
    MLflow (log_paper_eval_run itself stays Large-boundary/untested, same as
    the rest of this module)."""

    def test_finds_only_this_variants_sample_mask_pngs(self, tmp_path):
        (tmp_path / "mag1c_rgb_scene_a.png").touch()
        (tmp_path / "mag1c_rgb_scene_b.png").touch()
        (tmp_path / "mag1c_only_scene_a.png").touch()

        result = paper_eval_mlflow.collect_docs_asset_artifacts(tmp_path, "mag1c_rgb")

        assert sorted(p.name for p in result["sample_masks"]) == [
            "mag1c_rgb_scene_a.png",
            "mag1c_rgb_scene_b.png",
        ]

    def test_finds_only_this_variants_offline_prediction_json_files(self, tmp_path):
        # Real bug, caught live 2026-08-22: all three variants share one
        # staging_dir within a single eval_baseline flow run, and
        # persist_offline_predictions now prefixes filenames by variant to
        # avoid collisions -- this function must filter the same way, or a
        # later variant's MLflow upload still picks up an earlier
        # variant's leftover scene JSONs sitting in the same shared
        # offline_predictions/ folder (confirmed live: mag1c_only's
        # uploaded artifact had 8 files, 4 its own + 4 leftover from
        # varon, which ran first in the loop).
        offline_dir = tmp_path / "offline_predictions"
        offline_dir.mkdir()
        (offline_dir / "mag1c_rgb_scene_a.json").touch()
        (offline_dir / "mag1c_rgb_scene_b.json").touch()
        (offline_dir / "varon_scene_c.json").touch()
        (offline_dir / "mag1c_only_scene_d.json").touch()

        result = paper_eval_mlflow.collect_docs_asset_artifacts(tmp_path, "mag1c_rgb")

        assert sorted(p.name for p in result["offline_predictions"]) == [
            "mag1c_rgb_scene_a.json",
            "mag1c_rgb_scene_b.json",
        ]

    def test_returns_empty_lists_when_nothing_was_emitted(self, tmp_path):
        result = paper_eval_mlflow.collect_docs_asset_artifacts(tmp_path, "mag1c_rgb")

        assert result == {"sample_masks": [], "offline_predictions": []}

    def test_ignores_non_png_files_at_the_top_level(self, tmp_path):
        (tmp_path / "mag1c_rgb_scene_a.png").touch()
        (tmp_path / "mag1c_rgb_notes.txt").touch()

        result = paper_eval_mlflow.collect_docs_asset_artifacts(tmp_path, "mag1c_rgb")

        assert [p.name for p in result["sample_masks"]] == ["mag1c_rgb_scene_a.png"]


class TestCheckRegistryVersionMatches:
    """Real sqlite MlflowClient, same convention as test_mlflow_registry.py."""

    @pytest.fixture
    def client(self, tmp_path):
        return MlflowClient(tracking_uri=f"sqlite:///{tmp_path}/mlflow.db")

    def _register_with_sha(self, client, model_name, sha256, stage="Staging"):
        experiment_id = client.create_experiment(f"exp-for-{model_name}")
        run = client.create_run(experiment_id)
        client.set_tag(run.info.run_id, "checkpoint_sha256", sha256)
        return mlflow_registry.register_and_promote(client, run.info.run_id, model_name, stage)

    def test_returns_the_version_when_sha256_matches(self, client):
        version = self._register_with_sha(client, "starcop-baseline-varon", "deadbeef")

        result = paper_eval_mlflow.check_registry_version_matches(
            client, "starcop-baseline-varon", "deadbeef"
        )

        assert result.version == version.version

    def test_raises_on_sha256_mismatch(self, client):
        self._register_with_sha(client, "starcop-baseline-varon", "deadbeef")

        with pytest.raises(ValueError, match="drift"):
            paper_eval_mlflow.check_registry_version_matches(
                client, "starcop-baseline-varon", "different-sha"
            )

    def test_propagates_resolve_stage_version_error_when_nothing_is_registered(self, client):
        with pytest.raises(ValueError, match="no registered model"):
            paper_eval_mlflow.check_registry_version_matches(
                client, "starcop-baseline-does-not-exist", "deadbeef"
            )

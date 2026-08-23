"""Tests for src/registry/hf_baseline_import.py -- pure functions only (no
network, no MLflow SDK, no live server). Test Size: Small, except the
model-class dispatch tests, which import the real (unmodified)
starcop.models classes via the _vendor_starcop_baseline seam -- Test Size:
Medium (real import, no I/O).
"""

from pathlib import Path

import hf_baseline_import
import pytest
from _vendor_starcop_baseline import ModelModule, ModelModuleRegression


class TestVariantSubfolder:
    def test_mag1c_only_maps_to_its_hf_subfolder(self):
        assert hf_baseline_import.variant_subfolder("mag1c_only") == "hyperstarcop_mag1c_only"

    def test_mag1c_rgb_maps_to_its_hf_subfolder(self):
        assert hf_baseline_import.variant_subfolder("mag1c_rgb") == "hyperstarcop_mag1c_rgb"

    def test_unknown_variant_raises_value_error(self):
        with pytest.raises(ValueError, match="unknown_variant"):
            hf_baseline_import.variant_subfolder("unknown_variant")


class TestRegistryModelName:
    def test_mag1c_only_becomes_kebab_case_registry_name(self):
        assert hf_baseline_import.registry_model_name("mag1c_only") == "starcop-baseline-mag1c-only"

    def test_mag1c_rgb_becomes_kebab_case_registry_name(self):
        assert hf_baseline_import.registry_model_name("mag1c_rgb") == "starcop-baseline-mag1c-rgb"


class TestModelClassForMode:
    def test_segmentation_output_resolves_to_model_module(self):
        assert hf_baseline_import.model_class_for_mode("segmentation_output") is ModelModule

    def test_regression_output_resolves_to_model_module_regression(self):
        assert hf_baseline_import.model_class_for_mode("regression_output") is ModelModuleRegression

    def test_unknown_mode_raises_value_error(self):
        with pytest.raises(ValueError, match="unknown_mode"):
            hf_baseline_import.model_class_for_mode("unknown_mode")


class TestVerifyCheckpointDigest:
    def test_passes_silently_when_digest_matches_the_pinned_value(self, monkeypatch, tmp_path):
        import hashlib

        checkpoint_path = tmp_path / "final_checkpoint_model.ckpt"
        checkpoint_path.write_bytes(b"real checkpoint bytes")
        digest = hashlib.sha256(b"real checkpoint bytes").hexdigest()
        monkeypatch.setitem(hf_baseline_import._EXPECTED_CHECKPOINT_SHA256, "mag1c_only", digest)

        hf_baseline_import.verify_checkpoint_digest("mag1c_only", checkpoint_path)

    def test_raises_value_error_when_digest_does_not_match_the_pinned_value(
        self, monkeypatch, tmp_path
    ):
        checkpoint_path = tmp_path / "final_checkpoint_model.ckpt"
        checkpoint_path.write_bytes(b"tampered or corrupted bytes")
        monkeypatch.setitem(hf_baseline_import._EXPECTED_CHECKPOINT_SHA256, "mag1c_only", "0" * 64)

        with pytest.raises(ValueError, match="digest mismatch"):
            hf_baseline_import.verify_checkpoint_digest("mag1c_only", checkpoint_path)

    def test_computes_the_correct_digest_without_hashlib_file_digest(self, monkeypatch, tmp_path):
        """hashlib.file_digest was only added in Python 3.11 -- Environment A
        (vendor/starcop/.venv) is Python 3.10, so this function must not
        depend on it. Multi-chunk content (bigger than one read buffer)
        catches a chunk-boundary bug a naive single-read fallback wouldn't."""
        import hashlib

        content = b"x" * (2 * 1024 * 1024 + 137)  # spans multiple read chunks
        checkpoint_path = tmp_path / "final_checkpoint_model.ckpt"
        checkpoint_path.write_bytes(content)
        expected_digest = hashlib.sha256(content).hexdigest()
        monkeypatch.setitem(hf_baseline_import._EXPECTED_CHECKPOINT_SHA256, "mag1c_only", expected_digest)
        monkeypatch.delattr(hashlib, "file_digest", raising=False)

        hf_baseline_import.verify_checkpoint_digest("mag1c_only", checkpoint_path)  # must not raise


class TestDownloadCheckpoint:
    """Mocked at the huggingface_hub boundary (network, slow, non-deterministic
    upstream state) -- proves download_checkpoint's own orchestration (pass
    the pinned revision to both downloads, verify the checkpoint's digest)
    rather than huggingface_hub itself."""

    def test_checkpoint_and_config_are_downloaded_at_the_pinned_revision(
        self, monkeypatch, tmp_path
    ):
        import hashlib

        import huggingface_hub

        checkpoint_bytes = b"fake checkpoint content"
        monkeypatch.setitem(
            hf_baseline_import._EXPECTED_CHECKPOINT_SHA256,
            "mag1c_only",
            hashlib.sha256(checkpoint_bytes).hexdigest(),
        )

        calls = []

        def fake_hf_hub_download(repo_id, filename, revision=None, local_dir=None):
            calls.append({"filename": filename, "revision": revision})
            path = Path(local_dir) / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(checkpoint_bytes if filename.endswith(".ckpt") else b"config: true")
            return str(path)

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)

        checkpoint_path, config_path, revision = hf_baseline_import.download_checkpoint(
            "mag1c_only", tmp_path
        )

        assert revision == hf_baseline_import._PINNED_REVISION
        assert len(calls) == 2
        assert all(call["revision"] == hf_baseline_import._PINNED_REVISION for call in calls)


class TestLocalCheckpointDir:
    def test_varon_maps_to_its_local_dvc_tracked_directory(self):
        assert hf_baseline_import.local_checkpoint_dir("varon") == Path(
            "models/starcop_baseline/multistarcop_varon"
        )

    def test_unknown_variant_raises_value_error(self):
        with pytest.raises(ValueError, match="unknown_local_variant"):
            hf_baseline_import.local_checkpoint_dir("mag1c_only")


class TestResolveCheckpoint:
    """resolve_checkpoint is the single entry point both import_variant
    (registry import) and run_baseline_eval.py::evaluate_variant (the real
    eval run) call instead of download_checkpoint directly -- so a variant's
    source (HuggingFace vs. local/DVC-tracked) is decided in exactly one
    place. See track-a-paper-benchmark-reproduction-plan.md Phase 2."""

    def test_hf_variant_dispatches_to_download_checkpoint_fn_and_tags_source(self, tmp_path):
        calls = []

        def fake_download_checkpoint(variant, dest_dir):
            calls.append((variant, dest_dir))
            return Path("fake/checkpoint.ckpt"), Path("fake/config.yaml"), "abc123"

        checkpoint_path, config_path, tags = hf_baseline_import.resolve_checkpoint(
            "mag1c_only", tmp_path, download_checkpoint_fn=fake_download_checkpoint
        )

        assert calls == [("mag1c_only", tmp_path)]
        assert checkpoint_path == Path("fake/checkpoint.ckpt")
        assert config_path == Path("fake/config.yaml")
        assert tags == {
            "source": "huggingface",
            "hf_repo": hf_baseline_import._HF_REPO,
            "hf_revision": "abc123",
            "hf_subfolder": "hyperstarcop_mag1c_only",
            "checkpoint_sha256": hf_baseline_import._EXPECTED_CHECKPOINT_SHA256["mag1c_only"],
        }

    def test_local_variant_reads_from_disk_without_calling_download_checkpoint_fn(
        self, monkeypatch, tmp_path
    ):
        import hashlib

        checkpoint_bytes = b"fake local checkpoint content"
        local_dir = tmp_path / "multistarcop_varon"
        local_dir.mkdir()
        (local_dir / "final_checkpoint_model.ckpt").write_bytes(checkpoint_bytes)
        (local_dir / "config.yaml").write_text("model: {}")
        monkeypatch.setitem(hf_baseline_import._LOCAL_CHECKPOINT_PATHS, "varon", local_dir)
        monkeypatch.setitem(
            hf_baseline_import._EXPECTED_CHECKPOINT_SHA256,
            "varon",
            hashlib.sha256(checkpoint_bytes).hexdigest(),
        )

        def unreachable(variant, dest_dir):
            raise AssertionError("download_checkpoint_fn must not be called for a local variant")

        checkpoint_path, config_path, tags = hf_baseline_import.resolve_checkpoint(
            "varon", tmp_path, download_checkpoint_fn=unreachable
        )

        assert checkpoint_path == local_dir / "final_checkpoint_model.ckpt"
        assert config_path == local_dir / "config.yaml"
        assert tags == {
            "source": "local",
            "local_path": str(checkpoint_path),
            "checkpoint_sha256": hf_baseline_import._EXPECTED_CHECKPOINT_SHA256["varon"],
        }

    def test_local_variant_digest_mismatch_raises(self, monkeypatch, tmp_path):
        local_dir = tmp_path / "multistarcop_varon"
        local_dir.mkdir()
        (local_dir / "final_checkpoint_model.ckpt").write_bytes(b"tampered bytes")
        (local_dir / "config.yaml").write_text("model: {}")
        monkeypatch.setitem(hf_baseline_import._LOCAL_CHECKPOINT_PATHS, "varon", local_dir)
        monkeypatch.setitem(hf_baseline_import._EXPECTED_CHECKPOINT_SHA256, "varon", "0" * 64)

        with pytest.raises(ValueError, match="digest mismatch"):
            hf_baseline_import.resolve_checkpoint("varon", tmp_path)

    def test_unknown_variant_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError, match="unknown_variant"):
            hf_baseline_import.resolve_checkpoint("nonexistent", tmp_path)

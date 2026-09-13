"""Tests for src/training/dvc_dataset_version.py.

get_dataset_version reads a real dvc.lock file from disk (Test Size: Small --
single process, local file only, no network/subprocess). is_dataset_dirty
shells out to a real dvc binary against a real, minimal DVC repo built fresh
in tmp_path for each test (Test Size: Medium -- real subprocess, local-disk
only) -- see mlops-methane-detection-plan.md TASK-2.2 step 5: real fixtures
over mocks throughout.
"""

import subprocess
import sys
from pathlib import Path

import dvc_dataset_version as dvcv
import pytest
import yaml

REAL_DVC_BINARY = Path(sys.executable).with_name("dvc")


def _write_lock(tmp_path: Path, stages: dict) -> Path:
    lock_path = tmp_path / "dvc.lock"
    lock_path.write_text(yaml.safe_dump({"schema": "2.0", "stages": stages}))
    return lock_path


class TestGetDatasetVersion:
    def test_returns_md5_for_known_stage(self, tmp_path):
        lock_path = _write_lock(
            tmp_path,
            {
                "patch_extract@starcop_mini": {
                    "cmd": "python patch_extract.py",
                    "outs": [{"path": "data/processed/starcop_mini/patches", "md5": "abc123.dir"}],
                }
            },
        )

        assert dvcv.get_dataset_version("starcop_mini", lock_path) == "abc123.dir"

    def test_raises_value_error_when_stage_missing(self, tmp_path):
        lock_path = _write_lock(
            tmp_path,
            {
                "patch_extract@starcop_raw": {
                    "outs": [{"path": "x", "md5": "def456"}],
                }
            },
        )

        with pytest.raises(ValueError, match="patch_extract@starcop_mini"):
            dvcv.get_dataset_version("starcop_mini", lock_path)

    def test_raises_value_error_when_lock_has_no_stages_mapping(self, tmp_path):
        lock_path = tmp_path / "dvc.lock"
        lock_path.write_text("schema: '2.0'\n")

        with pytest.raises(ValueError, match="Available stages: \\[\\]"):
            dvcv.get_dataset_version("starcop_mini", lock_path)

    def test_raises_exact_error_when_stage_has_no_outputs(self, tmp_path):
        lock_path = _write_lock(tmp_path, {"patch_extract@starcop_mini": {}})

        with pytest.raises(ValueError) as exc_info:
            dvcv.get_dataset_version("starcop_mini", lock_path)

        assert str(exc_info.value) == (
            f"Stage 'patch_extract@starcop_mini' in {lock_path} has no outputs"
        )

    def test_raises_clear_error_on_malformed_yaml(self, tmp_path):
        lock_path = tmp_path / "dvc.lock"
        lock_path.write_text("stages: [this is not: valid: yaml: at all")

        with pytest.raises(ValueError, match="Malformed"):
            dvcv.get_dataset_version("starcop_mini", lock_path)


def _write_dvc_script(tmp_path: Path, body: str) -> Path:
    script = tmp_path / "fake-dvc"
    script.write_text(f"#!/bin/sh\n{body}\n")
    script.chmod(0o755)
    return script


def _init_tiny_dvc_repo(tmp_path: Path) -> Path:
    """Builds a real, minimal DVC repo with one patch_extract@mini stage.

    Stage names DVC generates via `foreach:` (e.g. patch_extract@mini) can't
    be written as a literal top-level key in dvc.yaml -- DVC only recognizes
    them when produced by an actual foreach block, matching how the real
    pipeline's patch_extract stage is defined. Paths must stay relative to
    the repo root: DVC treats absolute output paths as "external outputs"
    and refuses to cache them ("not supported since DVC 3.0").
    """
    subprocess.run([str(REAL_DVC_BINARY), "init", "--no-scm", "-q"], cwd=tmp_path, check=True)

    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "in.txt").write_text("hello")

    dvc_yaml = {
        "stages": {
            "patch_extract": {
                "foreach": ["mini"],
                "do": {
                    "cmd": "cp src/in.txt out.txt",
                    "deps": ["src/in.txt"],
                    "outs": ["out.txt"],
                },
            }
        }
    }
    (tmp_path / "dvc.yaml").write_text(yaml.safe_dump(dvc_yaml))

    subprocess.run([str(REAL_DVC_BINARY), "repro", "-q"], cwd=tmp_path, check=True)
    return tmp_path / "out.txt"


@pytest.mark.skipif(not REAL_DVC_BINARY.exists(), reason="research env's dvc binary not installed")
class TestIsDatasetDirty:
    def test_returns_false_for_a_clean_real_pipeline(self, tmp_path):
        _init_tiny_dvc_repo(tmp_path)

        assert dvcv.is_dataset_dirty("mini", tmp_path, dvc_binary=REAL_DVC_BINARY) is False

    def test_returns_true_after_a_tracked_output_is_mutated(self, tmp_path):
        output_file = _init_tiny_dvc_repo(tmp_path)
        output_file.write_text("mutated, no longer matches dvc.lock")

        assert dvcv.is_dataset_dirty("mini", tmp_path, dvc_binary=REAL_DVC_BINARY) is True

    def test_raises_clear_error_when_dvc_binary_path_does_not_exist(self, tmp_path):
        _init_tiny_dvc_repo(tmp_path)
        bogus_binary = tmp_path / "no-such-dvc-binary"

        with pytest.raises((FileNotFoundError, RuntimeError), match="no-such-dvc-binary"):
            dvcv.is_dataset_dirty("mini", tmp_path, dvc_binary=bogus_binary)

    def test_default_resolution_uses_venv_dvc_when_present(self, tmp_path):
        """No dvc_binary passed, and repo_root has a .venv/bin/dvc -- must use
        that one (train_mac.sh/train_desktop.sh's convention), not PATH."""
        _init_tiny_dvc_repo(tmp_path)
        venv_bin = tmp_path / ".venv" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "dvc").symlink_to(REAL_DVC_BINARY)

        assert dvcv.is_dataset_dirty("mini", tmp_path) is False

    def test_default_resolution_falls_back_to_path_dvc_when_no_venv(self, tmp_path, monkeypatch):
        """No dvc_binary passed, and repo_root has no .venv at all (e.g. Colab,
        where dvc is pip-installed straight into the system environment) --
        must fall back to shutil.which("dvc") instead of assuming a venv."""
        _init_tiny_dvc_repo(tmp_path)

        def fake_which(name):
            assert name == "dvc"
            return str(REAL_DVC_BINARY)

        monkeypatch.setattr(dvcv.shutil, "which", fake_which)

        assert dvcv.is_dataset_dirty("mini", tmp_path) is False

    def test_default_resolution_raises_clear_error_when_dvc_is_nowhere(self, tmp_path, monkeypatch):
        """No dvc_binary, no .venv, and PATH has no dvc either -- should still
        fail with a clear FileNotFoundError, not a confusing one."""
        _init_tiny_dvc_repo(tmp_path)

        def fake_which(name):
            assert name == "dvc"
            return None

        monkeypatch.setattr(dvcv.shutil, "which", fake_which)

        with pytest.raises(FileNotFoundError, match=r"\.venv[/\\]bin[/\\]dvc"):
            dvcv.is_dataset_dirty("mini", tmp_path)

    def test_nonzero_dvc_exit_reports_decoded_stderr(self, tmp_path):
        dvc_binary = _write_dvc_script(tmp_path, "echo 'status failed' >&2\nexit 3")

        with pytest.raises(RuntimeError) as exc_info:
            dvcv.is_dataset_dirty("mini", tmp_path, dvc_binary=dvc_binary)

        assert str(exc_info.value) == (
            "dvc status failed for 'patch_extract@mini' (exit 3): status failed\n"
        )

    def test_empty_successful_status_is_clean(self, tmp_path):
        dvc_binary = _write_dvc_script(tmp_path, "exit 0")

        assert dvcv.is_dataset_dirty("mini", tmp_path, dvc_binary=dvc_binary) is False

"""Characterization tests for STARCOP paths and stable CLI entrypoints."""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[5]


@pytest.mark.parametrize(
    ("entrypoint", "expected_help"),
    (
        ("src/baselines/starcop/training/train.py", "Hydra"),
        ("scripts/import_starcop_hf_baseline.py", "mag1c_rgb"),
        ("scripts/run_starcop_baseline_evaluation.py", "--device"),
        ("scripts/run_live_verify.py", "--base-url"),
    ),
)
def test_entrypoint_help_works_outside_repository(entrypoint, expected_help, tmp_path):
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / entrypoint), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert expected_help in result.stdout


def test_depth_sensitive_paths_resolve_from_clean_process(tmp_path):
    code = f"""
import sys
from pathlib import Path
repo = Path({str(REPO_ROOT)!r})
sys.path[:0] = [
    str(repo / relative)
    for relative in (
        'src/baselines/starcop/training',
        'src/training',
        'src/baselines/starcop/registry',
        'src/registry',
        'src/baselines/starcop/evaluation',
        'src/baselines/starcop/serving',
        'src/serving',
    )
]
import train
import hf_baseline_import
import _vendor_starcop_evaluation as evaluation_vendor
import _vendor_starcop_serving as serving_vendor
expected_vendor = repo / 'vendor' / 'starcop'
expected_importer = repo / 'src/baselines/starcop/registry/hf_baseline_import.py'
expected_evaluation_seam = repo / 'src/baselines/starcop/evaluation/_vendor_starcop_evaluation.py'
expected_serving_seam = repo / 'src/baselines/starcop/serving/_vendor_starcop_serving.py'
assert Path(train.__file__).resolve() == repo / 'src/baselines/starcop/training/train.py'
assert train._REPO_ROOT == repo
assert Path(hf_baseline_import.__file__).resolve() == expected_importer
assert train._OVERLAY_PATH == repo / 'configs' / 'training' / 'overlay.yaml'
assert Path(train._VENDOR_CONFIG_DIR) == expected_vendor / 'scripts' / 'configs'
assert hf_baseline_import._REPO_ROOT == repo
assert Path(evaluation_vendor.__file__).resolve() == expected_evaluation_seam
assert evaluation_vendor._VENDOR_STARCOP == expected_vendor
assert Path(serving_vendor.__file__).resolve() == expected_serving_seam
assert serving_vendor._VENDOR_STARCOP == expected_vendor
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr

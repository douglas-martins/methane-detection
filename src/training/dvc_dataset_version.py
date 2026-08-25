"""Derives an MLflow ``dataset_version`` tag from ``dvc.lock``, and checks
whether the corresponding pipeline stage has drifted from what's on disk.

See mlops-methane-detection-plan.md TASK-2.2 decision 4: the md5 recorded in
dvc.lock for the exact stage that produced the training patches is the
precise, content-addressed dataset identifier -- not `dvc status`, which
reports drift, not identity.
"""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import yaml


def get_dataset_version(dataset_name: str, dvc_lock_path: Path) -> str:
    """Returns the md5 output hash of the patch_extract stage for dataset_name."""
    try:
        lock = yaml.safe_load(Path(dvc_lock_path).read_text())
    except yaml.YAMLError as exc:
        raise ValueError(f"Malformed dvc.lock at {dvc_lock_path}: {exc}") from exc

    stage_key = f"patch_extract@{dataset_name}"
    stages = (lock or {}).get("stages", {})
    if stage_key not in stages:
        raise ValueError(
            f"Stage {stage_key!r} not found in {dvc_lock_path}. Available stages: {sorted(stages)}"
        )

    outs = stages[stage_key].get("outs") or []
    if not outs:
        raise ValueError(f"Stage {stage_key!r} in {dvc_lock_path} has no outputs")

    return outs[0]["md5"]


def is_dataset_dirty(dataset_name: str, repo_root: Path, dvc_binary: Optional[Path] = None) -> bool:
    """Returns True if patch_extract@dataset_name has drifted from dvc.lock."""
    if dvc_binary is None:
        venv_dvc = Path(repo_root) / ".venv" / "bin" / "dvc"
        if venv_dvc.exists():
            dvc_binary = venv_dvc
        else:
            # No project-local venv (e.g. Colab, where dvc is pip-installed
            # straight into the system/site environment rather than a uv-managed
            # .venv) -- fall back to whatever `dvc` resolves to on PATH.
            path_dvc = shutil.which("dvc")
            dvc_binary = Path(path_dvc) if path_dvc else venv_dvc

    stage_key = f"patch_extract@{dataset_name}"
    try:
        result = subprocess.run(
            [str(dvc_binary), "status", stage_key, "--json"],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"dvc binary not found at {dvc_binary}: {exc}") from exc

    if result.returncode != 0:
        raise RuntimeError(
            f"dvc status failed for {stage_key!r} (exit {result.returncode}): {result.stderr}"
        )

    status = json.loads(result.stdout or "{}")
    return bool(status)

"""Logs a Phase-1/2 `evaluate_variant()` result as a permanent run in the
`starcop-paper-eval` MLflow experiment -- see
track-a-paper-benchmark-reproduction-plan.md Phase 3.

Pure, unit-tested builders (`is_git_dirty`, `git_submodule_sha`,
`dvc_tracked_dir_hash`, `dependency_manifest`, `paper_eval_run_name`,
`build_paper_eval_tags`, `paper_eval_metrics`, `load_paper_reference_metrics`,
`render_paper_comparison`) plus `check_registry_version_matches` (Test Size:
Medium, real sqlite MlflowClient, same convention as
src/registry/__tests__/test_mlflow_registry.py). `log_paper_eval_run` itself
is thin SDK glue, Large-boundary, validated by a real run instead of a unit
test -- same pattern as hf_baseline_import.import_variant().
"""

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Optional

import yaml

_REGISTRY_DIR = str(Path(__file__).resolve().parents[1] / "registry")
if _REGISTRY_DIR not in sys.path:
    sys.path.insert(0, _REGISTRY_DIR)
_TRAINING_DIR = str(Path(__file__).resolve().parents[1] / "training")
if _TRAINING_DIR not in sys.path:
    sys.path.insert(0, _TRAINING_DIR)

import hf_baseline_import  # noqa: E402
import mlflow_registry  # noqa: E402
import mlflow_utils  # noqa: E402
from validation_metrics import extract_scalar_metrics  # noqa: E402


def resolve_git_binary(which_fn: Callable = shutil.which) -> str:
    """Same class of bug as `resolve_uv_binary` (see its docstring),
    applied preemptively: `git` happens to already be safe under launchd's
    restricted worker PATH (`/usr/bin/git` ships on every Mac, and that
    directory is in launchd's own default PATH), but that safety is
    coincidental, not designed -- fixed for consistency so this file has
    no bare-command PATH lookups left, per this project's own "always
    resolve explicitly in flow-reachable code" rule (see
    `deploy/prefect/README.md`)."""
    found = which_fn("git")
    if found:
        return found
    return "/usr/bin/git"


def _run_git_status(repo_root) -> str:
    result = subprocess.run(
        [resolve_git_binary(), "status", "--porcelain"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def is_git_dirty(repo_root, run_git_status_fn: Callable = _run_git_status) -> bool:
    """Returns True if `git status --porcelain` against repo_root has any
    output -- whole-repo scope, matching MLflow's own `mlflow.source.git.commit`
    auto-tag's scope (which captures HEAD's sha but not dirtiness, so a
    companion boolean is what's actually new here -- the sha itself doesn't
    need a custom tag, see Phase 3's resolved implementation details)."""
    return bool(run_git_status_fn(repo_root).strip())


def _run_git_rev_parse_head(repo_dir) -> str:
    result = subprocess.run(
        [resolve_git_binary(), "rev-parse", "HEAD"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def git_submodule_sha(
    submodule_dir, run_git_rev_parse_fn: Callable = _run_git_rev_parse_head
) -> str:
    """Returns the commit sha `submodule_dir` (e.g. vendor/starcop) is
    currently checked out at -- not captured by MLflow's own git auto-tagging,
    which only ever sees this repo's own HEAD."""
    return run_git_rev_parse_fn(submodule_dir)


def dvc_tracked_dir_hash(dvc_file_path) -> str:
    """Reads a plain single-output .dvc pointer file (e.g. data/starcop_raw.dvc)
    and returns its tracked content hash directly -- the dataset-version
    identity for data tracked as a whole directory, not via a dvc.lock
    pipeline stage (dvc_dataset_version.py's get_dataset_version reads the
    wrong artifact for this use case -- see Phase 3's resolved implementation
    details)."""
    contents = yaml.safe_load(Path(dvc_file_path).read_text())
    outs = (contents or {}).get("outs") or []
    if not outs:
        raise ValueError(f"No outs in {dvc_file_path}")
    return outs[0]["md5"]


def resolve_uv_binary(which_fn: Callable = shutil.which) -> str:
    """Resolves an absolute path to the `uv` binary rather than relying on
    a bare "uv" PATH lookup -- real bug, caught live 2026-08-22: the
    eval-baseline Prefect flow's launchd-supervised worker process gets
    launchd's own minimal default PATH (/usr/bin:/bin:/usr/sbin:/sbin, no
    ~/.local/bin), so a bare "uv" lookup raised FileNotFoundError even
    though uv is installed and used successfully everywhere else in this
    project (interactive shells always have the fuller PATH). Falls back
    to this machine's real install location (~/.local/bin/uv, confirmed
    via `which uv`) if the PATH lookup fails, raising a clear error only
    if neither resolves."""
    found = which_fn("uv")
    if found:
        return found
    fallback = Path.home() / ".local" / "bin" / "uv"
    if fallback.exists():
        return str(fallback)
    raise RuntimeError(
        f"could not find the `uv` binary on PATH or at the usual {fallback} fallback"
    )


def _run_uv_pip_freeze(python_executable) -> str:
    result = subprocess.run(
        [resolve_uv_binary(), "pip", "freeze", "--python", str(python_executable)],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def dependency_manifest(python_executable, run_freeze_fn: Callable = _run_uv_pip_freeze) -> str:
    """Returns a frozen dependency manifest for `python_executable` via
    `uv pip freeze` -- plain `pip freeze` doesn't work in Environment A at all
    (its venv has no `pip` module installed), but `uv pip freeze` works
    against any venv regardless of how it was created (verified live)."""
    return run_freeze_fn(python_executable)


def paper_eval_run_name(variant: str, date: str) -> str:
    """e.g. starcop-baseline-mag1c-rgb-paper-eval-2026-08-21."""
    return f"{hf_baseline_import.registry_model_name(variant)}-paper-eval-{date}"


def build_paper_eval_tags(
    variant: str,
    registry_version,
    checkpoint_sha256: str,
    dvc_dataset_version: str,
    n_test_scenes: int,
    resolved_device: str,
    eval_code_dirty: bool,
    vendor_starcop_sha: str,
) -> dict:
    """Builds the MLflow run tags identifying exactly what data/code/checkpoint
    a paper-eval run used -- pure, no SDK calls (mirrors
    src/training/mlflow_utils.py::build_run_tags's own split)."""
    return {
        "variant": variant,
        "registry_model_name": hf_baseline_import.registry_model_name(variant),
        "registry_version": str(registry_version),
        "checkpoint_sha256": checkpoint_sha256,
        "dvc_dataset_version": dvc_dataset_version,
        "n_test_scenes": str(n_test_scenes),
        "paper_reference": "true",
        "resolved_device": resolved_device,
        "eval_code_dirty": str(eval_code_dirty),
        "vendor_starcop_sha": vendor_starcop_sha,
    }


def paper_eval_metrics(corrected_metrics: dict, run_validation_metrics: dict) -> dict:
    """Combines the corrected Table 1/2 headline numbers (already explicitly
    named -- strong_f1score/weak_f1score/no_plume_FPR/auprc -- distinguished
    from run_validation's own uncorrected easy_*/hard_* keys) with the full
    aggregate metrics via validation_metrics.extract_scalar_metrics under a
    `raw_` prefix, so the two sets can never collide."""
    full = extract_scalar_metrics(run_validation_metrics, prefix="raw")
    return {**full, **corrected_metrics}


def load_paper_reference_metrics(path) -> dict:
    """Parses the fenced ```yaml block from paper_reference_metrics.md (the
    project's one canonical, hand-entered source for the paper's Table 1/2
    values) -- not the prose tables above it, which exist for human citation
    only."""
    text = Path(path).read_text()
    match = re.search(r"```yaml\n(.*?)\n```", text, re.DOTALL)
    if not match:
        raise ValueError(f"No fenced yaml block found in {path}")
    return yaml.safe_load(match.group(1))


def render_paper_comparison(variant: str, this_run_metrics: dict, reference: dict) -> str:
    """Renders a small markdown table comparing this run's corrected metrics
    against the paper's own reported mean/std for `variant`. Values as
    percentages, matching the paper's own table formatting."""
    ref = reference[variant]
    metric_keys = [
        ("strong_f1score", "Strong F1"),
        ("weak_f1score", "Weak F1"),
        ("no_plume_FPR", "FPR (tile-level)"),
        ("auprc", "AUPRC"),
    ]
    lines = [
        f"# Paper comparison — {variant}",
        "",
        f"Paper source: {ref['citation']}",
        "",
        "| Metric | Paper | This run |",
        "| --- | --- | --- |",
    ]
    for key, label in metric_keys:
        paper_mean = ref[key]["mean"] * 100
        paper_std = ref[key]["std"] * 100
        this_value = this_run_metrics[key] * 100
        lines.append(f"| {label} | {paper_mean:.2f} ± {paper_std:.2f} | {this_value:.2f} |")
    return "\n".join(lines) + "\n"


def collect_docs_asset_artifacts(emit_docs_assets_dir, variant: str) -> dict:
    """Finds the files `evaluate_variant(..., emit_docs_assets_dir=...)`
    wrote for `variant` that Phase 5's `live_verify.py` needs in MLflow, not
    just on local disk: the curated sample-mask PNGs
    (`{variant}_{scene_id}.png`, written directly under
    `emit_docs_assets_dir` by run_baseline_eval.py) and the offline
    prediction mask-digest/confidence JSON files
    (`persist_offline_predictions`'s output, under
    `emit_docs_assets_dir/offline_predictions/`). Pure filesystem selection,
    no MLflow calls -- `log_paper_eval_run` does the actual uploading.
    Returns empty lists, never raises, when nothing was emitted for this
    variant (e.g. a variant whose curated picks were empty).

    `offline_predictions` is filtered by the same `{variant}_` prefix as
    `sample_masks` -- all three variants share one staging directory
    within a single `eval_baseline` flow run, so an unfiltered glob here
    would upload a later variant's leftover scene JSONs sitting in the
    same shared folder as this variant's own artifact (confirmed live
    2026-08-22: see `persist_offline_predictions`'s docstring for the
    real failure this caused)."""
    emit_docs_assets_dir = Path(emit_docs_assets_dir)
    sample_masks = sorted(emit_docs_assets_dir.glob(f"{variant}_*.png"))
    offline_predictions = sorted(
        (emit_docs_assets_dir / "offline_predictions").glob(f"{variant}_*.json")
    )
    return {"sample_masks": sample_masks, "offline_predictions": offline_predictions}


def check_registry_version_matches(
    client, model_name: str, checkpoint_sha256: str, stage: str = "Staging"
):
    """Looks up the version currently at `stage` for `model_name` and raises
    ValueError if its own `checkpoint_sha256` tag doesn't match
    `checkpoint_sha256` -- catches registry/eval drift (e.g. a promotion race
    between Phase 2's import and this logging step) instead of silently
    tagging a registry version that may not correspond to what was actually
    evaluated. Returns the matching ModelVersion."""
    version = mlflow_registry.resolve_stage_version(client, model_name, stage)
    run = client.get_run(version.run_id)
    registered_sha256 = run.data.tags.get("checkpoint_sha256")
    if registered_sha256 != checkpoint_sha256:
        raise ValueError(
            f"registry drift: {model_name!r} at {stage!r} (version {version.version}) "
            f"was registered with checkpoint_sha256={registered_sha256!r}, but this run "
            f"evaluated checkpoint_sha256={checkpoint_sha256!r} -- the registry's {stage} "
            "version has moved since it was resolved for this eval; re-import or re-resolve "
            "before logging."
        )
    return version


def log_paper_eval_run(
    variant: str,
    result: dict,
    repo_root: Path,
    reference_metrics_path: Path,
    dvc_file_path: Optional[Path] = None,
    date: Optional[str] = None,
    emit_docs_assets_dir: Optional[Path] = None,
) -> str:
    """Logs `result` (evaluate_variant()'s return dict, extended with
    `joined_scene_results`/`run_validation_metrics`/`checkpoint_provenance`/
    `device`) as one run in `starcop-paper-eval`. Large-boundary, real-run
    validated (not unit tested) -- same pattern as
    hf_baseline_import.import_variant(). When `emit_docs_assets_dir` is
    given (the same directory passed to `evaluate_variant(...,
    emit_docs_assets_dir=...)` for this call), also uploads the curated
    sample-mask PNGs and offline-prediction mask-digest/confidence JSON
    files `collect_docs_asset_artifacts` finds there -- this is the artifact
    Phase 5's `live_verify.py` diffs the live API's response against; without
    it, this run has nothing to verify a served model against. Returns the
    MLflow run id."""
    import datetime
    import json

    import mlflow
    import pandas as pd
    from mlflow.tracking import MlflowClient

    repo_root = Path(repo_root)
    if dvc_file_path is None:
        dvc_file_path = repo_root / "data" / "starcop_raw.dvc"

    mlflow_utils.require_mlflow_tracking_env()
    mlflow.set_experiment("starcop-paper-eval")

    checkpoint_sha256 = result["checkpoint_provenance"]["checkpoint_sha256"]
    model_name = hf_baseline_import.registry_model_name(variant)
    client = MlflowClient()
    version = check_registry_version_matches(client, model_name, checkpoint_sha256)

    dvc_dataset_version = dvc_tracked_dir_hash(dvc_file_path)
    eval_code_dirty = is_git_dirty(repo_root)
    vendor_sha = git_submodule_sha(repo_root / "vendor" / "starcop")
    manifest = dependency_manifest(sys.executable)
    reference = load_paper_reference_metrics(reference_metrics_path)
    comparison_md = render_paper_comparison(variant, result["metrics"], reference)

    tags = build_paper_eval_tags(
        variant=variant,
        registry_version=version.version,
        checkpoint_sha256=checkpoint_sha256,
        dvc_dataset_version=dvc_dataset_version,
        n_test_scenes=result["n_scenes"],
        resolved_device=result["device"],
        eval_code_dirty=eval_code_dirty,
        vendor_starcop_sha=vendor_sha,
    )
    metrics = paper_eval_metrics(result["metrics"], result["run_validation_metrics"])

    run_name = paper_eval_run_name(variant, date or datetime.date.today().isoformat())
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(tags)
        mlflow.log_metrics(metrics)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)

            csv_path = tmp_dir / "per_scene_results.csv"
            pd.DataFrame.from_records(result["joined_scene_results"]).to_csv(csv_path, index=False)
            mlflow.log_artifact(str(csv_path))

            metrics_json_path = tmp_dir / "run_validation_metrics.json"
            metrics_json_path.write_text(json.dumps(result["run_validation_metrics"], default=str))
            mlflow.log_artifact(str(metrics_json_path))

            manifest_path = tmp_dir / "dependency_manifest.txt"
            manifest_path.write_text(manifest)
            mlflow.log_artifact(str(manifest_path))

            comparison_path = tmp_dir / "paper_comparison.md"
            comparison_path.write_text(comparison_md)
            mlflow.log_artifact(str(comparison_path))

            if emit_docs_assets_dir is not None:
                docs_assets = collect_docs_asset_artifacts(emit_docs_assets_dir, variant)
                for png_path in docs_assets["sample_masks"]:
                    mlflow.log_artifact(str(png_path), artifact_path="sample_masks")
                for json_path in docs_assets["offline_predictions"]:
                    mlflow.log_artifact(str(json_path), artifact_path="offline_predictions")

    marker_line = mlflow_utils.write_run_id_marker(str(repo_root), run.info.run_id)
    print(marker_line)
    return run.info.run_id

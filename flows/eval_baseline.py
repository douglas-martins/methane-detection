"""Phase 4: Prefect flow that makes the paper-eval benchmark a repeatable,
auditable, on-demand run -- pulls the dataset, ensures MultiSTARCOP is
registered, runs the full evaluation for all three variants, live-checks
the two servable Hyper variants against a throwaway local `bentoml serve`
process, aggregates everything into one combined comparison table, and
atomically publishes it to the canonical docs-asset directory. See
track-a-paper-benchmark-reproduction-plan.md Phase 4.

Same `@task`/`@flow` shape and injectable-callable testing pattern as
flows/retrain.py (its `pull_dataset`/`notify`/`build_failure_message`
helpers are reused directly, not reinvented, via `import retrain`).

Run manually (Environment B):
    .venv/bin/python flows/eval_baseline.py
"""

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

import requests
from mlflow.tracking import MlflowClient
from prefect import flow, task

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src" / "evaluation"))
sys.path.insert(0, str(_REPO_ROOT / "src" / "registry"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import hf_baseline_import  # noqa: E402
import live_verify  # noqa: E402
import mlflow_registry  # noqa: E402
import paper_eval_mlflow  # noqa: E402
import retrain  # noqa: E402

# Hardcoded rather than read from env, matching retrain.py's own
# MLFLOW_TRACKING_URI constant: it's a public HTTPS endpoint, not a secret,
# and this flow's worker process never sources train_mac.sh's .env.mlflow.
MLFLOW_TRACKING_URI = "https://methane-detection-mlflow.ghostface.tech"

# Same reasoning as MLFLOW_TRACKING_URI above -- a public S3-compatible
# endpoint hostname, not a secret, so it's a code constant explicitly
# injected into subprocess env rather than a .env.prefect line. Real bug,
# caught live 2026-08-22: without this, boto3 defaults to real AWS S3
# instead of Backblaze B2, and a B2 key against real AWS fails with
# InvalidAccessKeyId, not a connectivity error -- confusing to diagnose
# from that message alone. `.env.mlflow` has always had this var (used
# successfully all session for manual uploads); `.env.prefect` never did,
# because the earlier audit that copied AWS_ACCESS_KEY_ID/
# AWS_SECRET_ACCESS_KEY across used a grep pattern that can't match a var
# name containing a digit ("S3"), so this one was invisible to it.
MLFLOW_S3_ENDPOINT_URL = "https://s3.us-east-005.backblazeb2.com"

VARIANTS = ("varon", "mag1c_only", "mag1c_rgb")
SERVABLE_VARIANTS = live_verify.SERVABLE_VARIANTS
CANONICAL_DOCS_DIR = _REPO_ROOT / "docs" / "assets" / "paper_eval"
_REFERENCE_METRICS_PATH = _REPO_ROOT / "internal-docs" / "plans" / "paper_reference_metrics.md"
_METRIC_KEYS = ("strong_f1score", "weak_f1score", "no_plume_FPR", "auprc")

BENTOML_PORT = 3001
# 20 minutes: sized from this session's own observed worst-case B2/IPv6
# connectivity stall while validating Phase 5 locally (over 20 minutes in
# one case) -- must be bounded, not left to hang the whole flow run
# indefinitely (see Phase 4's own "Startup timeout" resolved-details note).
HEALTH_TIMEOUT_SECONDS = 1200


def multistarcop_registered_and_current(
    client,
    resolve_checkpoint_fn: Callable = hf_baseline_import.resolve_checkpoint,
    resolve_stage_version_fn: Callable = mlflow_registry.resolve_stage_version,
) -> bool:
    """True if `starcop-baseline-varon`'s Staging version's own
    checkpoint_sha256 tag already matches the checkpoint
    resolve_checkpoint_fn would resolve right now. varon is a local,
    DVC-tracked checkpoint (hf_baseline_import._LOCAL_CHECKPOINT_PATHS), so
    this never makes a network call. Returns False (never raises) for "not
    registered yet" or "registered but stale" -- both are expected, handled
    cases here, unlike paper_eval_mlflow.check_registry_version_matches's
    stricter fail-fast check."""
    with tempfile.TemporaryDirectory() as tmp:
        _checkpoint_path, _config_path, provenance = resolve_checkpoint_fn("varon", Path(tmp))
    try:
        version = resolve_stage_version_fn(client, "starcop-baseline-varon", "Staging")
    except ValueError:
        return False
    run = client.get_run(version.run_id)
    return run.data.tags.get("checkpoint_sha256") == provenance["checkpoint_sha256"]


def ensure_multistarcop_registered(
    client,
    is_registered_fn: Callable = multistarcop_registered_and_current,
    import_fn: Callable = hf_baseline_import.import_variant,
) -> None:
    """Idempotent check-then-import: only calls `import_fn` (a real MLflow
    registry write) when varon isn't already registered with a matching
    checkpoint."""
    if not is_registered_fn(client):
        import_fn("varon", "Staging")


def run_evaluation_for_variant(
    repo_root: Path, variant: str, staging_dir: Path, cmd_runner: Callable = subprocess.run
) -> str:
    """Shells to scripts/run_starcop_baseline_evaluation.py for `variant`,
    always with `--emit-docs-assets staging_dir`, never `--limit` (this
    phase's own "flow never passes --limit" rule) -- and returns the
    `MLFLOW_RUN_ID` sentinel from stdout via retrain.py's own
    `parse_run_id`. Runs under Environment A (`vendor/starcop/.venv`,
    torch 1.13.1, Phase 0's decision), not the root `.venv` this flow
    process itself runs under -- an explicit environment crossing, not an
    inherited PATH lookup. Explicitly injects MLFLOW_TRACKING_URI into the
    subprocess env: this flow's own process only has it as a Python
    constant, which doesn't propagate to a child process on its own --
    caught live 2026-08-22 when a real varon run completed its full
    342-scene pass and then crashed at the MLflow-logging step for exactly
    this reason."""
    python = str(repo_root / "vendor" / "starcop" / ".venv" / "bin" / "python")
    script = str(repo_root / "scripts" / "run_starcop_baseline_evaluation.py")
    env = {
        **os.environ,
        "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI,
        "MLFLOW_S3_ENDPOINT_URL": MLFLOW_S3_ENDPOINT_URL,
    }
    result = cmd_runner(
        [python, script, variant, "--emit-docs-assets", str(staging_dir)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        stderr_tail = "\n".join(result.stderr.splitlines()[-40:])
        raise RuntimeError(
            f"evaluation failed for variant={variant!r} "
            f"with exit code {result.returncode}\n{stderr_tail}"
        )
    return retrain.parse_run_id(result.stdout)


def start_bentoml_serve(
    repo_root: Path, variant: str, port: int, popen: Callable = subprocess.Popen
):
    """Launches a throwaway `bentoml serve` process for `variant` on
    `port`, pinned via MODEL_NAME/MODEL_STAGE env vars -- never the real
    deployed production service (Phase 4's own "Live-check HTTP target"
    decision). Also injects MLFLOW_TRACKING_URI/MLFLOW_S3_ENDPOINT_URL
    explicitly -- same gap as `run_evaluation_for_variant`'s (see its
    docstring): `service.py`'s `__init__` requires the former in
    `os.environ`, and `model_loader.py` needs the latter to actually
    download the served model's weights from B2, which this flow's own process
    only has as a Python constant."""
    python = str(repo_root / ".venv" / "bin" / "python")
    env = {
        **os.environ,
        "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI,
        "MLFLOW_S3_ENDPOINT_URL": MLFLOW_S3_ENDPOINT_URL,
        "MODEL_NAME": hf_baseline_import.registry_model_name(variant),
        "MODEL_STAGE": "Staging",
    }
    return popen(
        [
            python,
            "-m",
            "bentoml",
            "serve",
            "src.serving.service:MethaneDetectionService",
            "--port",
            str(port),
        ],
        cwd=repo_root,
        env=env,
    )


def wait_for_health(
    base_url: str,
    timeout_seconds: float = HEALTH_TIMEOUT_SECONDS,
    http_post: Callable = requests.post,
    sleep_fn: Callable = time.sleep,
    time_fn: Callable = time.monotonic,
) -> dict:
    """Polls `POST /health` (service.py's own convention, not `GET`) until
    it responds 200, or raises TimeoutError past `timeout_seconds` -- an
    explicit deadline is required here: an indefinitely-hung `bentoml
    serve` startup would hang the whole flow run, not just fail one
    "non-fatal" task."""
    deadline = time_fn() + timeout_seconds
    last_error = None
    while time_fn() < deadline:
        try:
            response = http_post(f"{base_url}/health", timeout=10)
            if response.status_code == 200:
                return response.json()
            last_error = RuntimeError(f"HTTP {response.status_code}")
        except requests.exceptions.RequestException as exc:
            last_error = exc
        sleep_fn(2)
    raise TimeoutError(
        f"bentoml serve at {base_url!r} did not become healthy "
        f"within {timeout_seconds}s ({last_error})"
    )


def run_live_check_for_variant(
    repo_root: Path,
    variant: str,
    tracking_uri: str,
    port: int = BENTOML_PORT,
    start_serve_fn: Callable = start_bentoml_serve,
    wait_for_health_fn: Callable = wait_for_health,
    verify_fn: Callable = live_verify.verify_variant,
) -> dict:
    """Non-fatal: a serving-side outage shouldn't block regenerating the
    numbers. Starts a throwaway `bentoml serve` process, waits for it, runs
    Phase 5's `verify_variant`, and always tears the process down
    (`finally`), even on failure. Returns a status dict
    (`"passed"`/`"failed"`/`"not_run"`) -- never raises.

    Unlike `run_evaluation_for_variant`/`start_bentoml_serve`, `verify_fn`
    runs in this flow's own process, not a subprocess -- there's no `env=`
    to inject into for it, so `MLFLOW_S3_ENDPOINT_URL` is set directly via
    `os.environ` (once, only if absent) right before the call. Real bug,
    caught live 2026-08-22: without it, `verify_variant`'s own MLflow
    artifact download defaults to real AWS S3 instead of Backblaze B2 and
    fails, caught by this function's own `except Exception` and reported
    as `"not_run"` -- a live check that silently never actually checked
    anything."""
    base_url = f"http://localhost:{port}"
    process = start_serve_fn(repo_root, variant, port)
    try:
        wait_for_health_fn(base_url)
        os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", MLFLOW_S3_ENDPOINT_URL)
        result = verify_fn(
            variant,
            base_url=base_url,
            tracking_uri=tracking_uri,
            test_csv_path=str(repo_root / "data" / "starcop_raw" / "test.csv"),
            root_folder=str(repo_root / "data" / "starcop_raw" / "STARCOP_test"),
        )
        return {"status": "passed" if result["passed"] else "failed", "detail": result}
    except Exception as exc:
        return {"status": "not_run", "detail": str(exc)}
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=30)
            except Exception:
                pass


def validate_run_completeness(variant_results: dict, servable_variants=SERVABLE_VARIANTS) -> None:
    """Raises ValueError unless every expected variant has a `run_id` and
    every servable variant has a live-check status recorded -- a live check
    must have a recorded status, never be silently missing. A partial run
    must never reach publish."""
    missing_runs = [v for v in VARIANTS if not variant_results.get(v, {}).get("run_id")]
    if missing_runs:
        raise ValueError(f"missing run_id for variant(s): {missing_runs}")
    missing_live_checks = [
        v for v in servable_variants if "live_check" not in variant_results.get(v, {})
    ]
    if missing_live_checks:
        raise ValueError(
            f"missing live-check status for servable variant(s): {missing_live_checks}"
        )


def render_aggregate_comparison(variant_results: dict, reference: dict) -> str:
    """Render one raw Markdown comparison table per model variant.

    Separate tables keep the paper-versus-run comparison readable while each
    variant's live spot-check status remains explicit.
    """
    metric_labels = (
        ("strong_f1score", "Strong F1"),
        ("weak_f1score", "Weak F1"),
        ("no_plume_FPR", "FPR (tile-level)"),
        ("auprc", "AUPRC"),
    )
    variant_labels = {
        "varon": "MultiSTARCOP — Varon ratio",
        "mag1c_only": "HyperSTARCOP — mag1c only",
        "mag1c_rgb": "HyperSTARCOP — mag1c + RGB",
    }
    lines = ["# Paper comparison — all variants"]
    for variant in VARIANTS:
        ref = reference.get(variant, {})
        metrics = variant_results[variant]["metrics"]
        live_check = variant_results[variant].get("live_check")
        live_status = (
            live_check["status"] if live_check else "out of scope (MultiSTARCOP not deployed live)"
        )
        lines.extend(
            [
                "",
                f"## {variant_labels[variant]}",
                "",
                f"**Variant:** `{variant}` · **Live API check:** {live_status}",
                "",
                "| Metric | Paper | Reproduced |",
                "| --- | ---: | ---: |",
            ]
        )
        for key, label in metric_labels:
            paper_value = ref.get(key)
            paper_str = (
                f"{paper_value['mean'] * 100:.2f} ± {paper_value['std'] * 100:.2f}"
                if paper_value
                else "n/a"
            )
            this_value = metrics[key] * 100
            lines.append(f"| {label} | {paper_str} | {this_value:.2f} |")
    return "\n".join(lines) + "\n"


def publish_staging_dir(staging_dir, canonical_dir) -> None:
    """Atomically replaces `canonical_dir`'s contents with `staging_dir`'s
    -- write-to-sibling-then-`rename`, not an in-place file-by-file
    overwrite, so docs/results.md (and any concurrent `make docs-build`)
    only ever sees the previous complete publish or the new complete one,
    never a directory mid-overwrite."""
    canonical_dir = Path(canonical_dir)
    staging_dir = Path(staging_dir)
    canonical_dir.parent.mkdir(parents=True, exist_ok=True)

    tmp_new = canonical_dir.parent / f"{canonical_dir.name}.new"
    tmp_old = canonical_dir.parent / f"{canonical_dir.name}.old"
    if tmp_new.exists():
        shutil.rmtree(tmp_new)
    shutil.copytree(staging_dir, tmp_new)

    if canonical_dir.exists():
        if tmp_old.exists():
            shutil.rmtree(tmp_old)
        canonical_dir.rename(tmp_old)
    tmp_new.rename(canonical_dir)
    if tmp_old.exists():
        shutil.rmtree(tmp_old)


def build_success_message(variant_results: dict) -> str:
    parts = []
    for variant in VARIANTS:
        info = variant_results[variant]
        line = f"{variant}: run {info['run_id']}"
        if "live_check" in info:
            line += f" (live check: {info['live_check']['status']})"
        parts.append(line)
    return "eval_baseline completed:\n" + "\n".join(parts)


def build_failure_message(step: str, error: Exception) -> str:
    return f"eval_baseline flow failed at step '{step}': {error}"


def run_eval_baseline_cycle(
    repo_root: Path,
    tracking_uri: str,
    reference_metrics_path: Path,
    canonical_docs_dir: Path,
    pushover_user_key: str,
    pushover_api_token: str,
    client: Optional[MlflowClient] = None,
    pull_fn: Callable = retrain.pull_dataset,
    ensure_registered_fn: Callable = ensure_multistarcop_registered,
    evaluate_fn: Callable = run_evaluation_for_variant,
    live_check_fn: Callable = run_live_check_for_variant,
    aggregate_fn: Callable = render_aggregate_comparison,
    publish_fn: Callable = publish_staging_dir,
    notify_fn: Callable = retrain.notify,
    load_reference_fn: Callable = paper_eval_mlflow.load_paper_reference_metrics,
    fetch_metrics_fn: Callable = mlflow_registry.fetch_run_metrics,
) -> dict:
    """Plain-Python orchestration of the paper-eval cycle, kept free of any
    Prefect decorator so it's directly unit-testable (Test Size: Small)
    with fakes injected for every side-effecting step -- same pattern as
    retrain.py's own `run_retraining_cycle`. `eval_baseline_flow` (the real
    `@flow`, below) calls this with `@task`-wrapped versions of each fn."""
    if client is None:
        client = MlflowClient(tracking_uri=tracking_uri)

    current_step = "pull_dataset"
    variant_results: dict = {}
    try:
        pull_fn(repo_root)

        current_step = "ensure_multistarcop_registered"
        ensure_registered_fn(client)

        with tempfile.TemporaryDirectory() as staging:
            staging_dir = Path(staging)

            current_step = "run_evaluation"
            for variant in VARIANTS:
                run_id = evaluate_fn(repo_root, variant, staging_dir)
                variant_results[variant] = {"run_id": run_id}

            current_step = "live_check"
            for variant in SERVABLE_VARIANTS:
                variant_results[variant]["live_check"] = live_check_fn(
                    repo_root, variant, tracking_uri
                )

            current_step = "aggregate"
            validate_run_completeness(variant_results)
            reference = load_reference_fn(reference_metrics_path)
            for variant in VARIANTS:
                metrics = fetch_metrics_fn(client, variant_results[variant]["run_id"])
                variant_results[variant]["metrics"] = {key: metrics[key] for key in _METRIC_KEYS}
            combined_md = aggregate_fn(variant_results, reference)
            (staging_dir / "paper_comparison.md").write_text(combined_md)

            current_step = "publish"
            publish_fn(staging_dir, canonical_docs_dir)
    except Exception as exc:
        notify_fn(pushover_user_key, pushover_api_token, build_failure_message(current_step, exc))
        raise

    notify_fn(pushover_user_key, pushover_api_token, build_success_message(variant_results))
    return variant_results


@task
def pull_dataset_task(repo_root: Path) -> None:
    retrain.pull_dataset(repo_root)


@task
def ensure_registered_task(client) -> None:
    ensure_multistarcop_registered(client)


@task
def evaluate_variant_task(repo_root: Path, variant: str, staging_dir: Path) -> str:
    return run_evaluation_for_variant(repo_root, variant, staging_dir)


@task
def live_check_task(repo_root: Path, variant: str, tracking_uri: str) -> dict:
    return run_live_check_for_variant(repo_root, variant, tracking_uri)


@task
def aggregate_task(variant_results: dict, reference: dict) -> str:
    return render_aggregate_comparison(variant_results, reference)


@task
def publish_task(staging_dir, canonical_dir) -> None:
    publish_staging_dir(staging_dir, canonical_dir)


@task
def notify_task(user_key: str, api_token: str, message: str) -> None:
    retrain.notify(user_key, api_token, message)


@flow(name="eval-baseline")
def eval_baseline_flow() -> dict:
    return run_eval_baseline_cycle(
        repo_root=_REPO_ROOT,
        tracking_uri=MLFLOW_TRACKING_URI,
        reference_metrics_path=_REFERENCE_METRICS_PATH,
        canonical_docs_dir=CANONICAL_DOCS_DIR,
        pushover_user_key=os.environ["PUSHOVER_USER_KEY"],
        pushover_api_token=os.environ["PUSHOVER_API_TOKEN"],
        pull_fn=pull_dataset_task,
        ensure_registered_fn=ensure_registered_task,
        evaluate_fn=evaluate_variant_task,
        live_check_fn=live_check_task,
        aggregate_fn=aggregate_task,
        publish_fn=publish_task,
        notify_fn=notify_task,
    )


if __name__ == "__main__":
    eval_baseline_flow()

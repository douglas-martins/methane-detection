"""TASK-7.2: Prefect flow orchestrating the retraining cycle end-to-end.

Runs under Environment B (root .venv, Python 3.12 -- where `prefect` is
installed) as a flow run on the `mac-mps` Process work pool. See
mlops-methane-detection-plan.md's Phase 7 section for the design history:
D-04 decided the trigger is a cron schedule (see prefect.yaml), and D-01's
service-account note covers the unattended `dvc pull` credential this flow
assumes is already configured in .dvc/config.local on the worker machine.

Run manually (Environment B):
    .venv/bin/python flows/retrain.py
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

import requests
from mlflow.tracking import MlflowClient
from prefect import flow, task

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src" / "registry"))

import promote_model  # noqa: E402

# Hardcoded rather than read from env, matching scripts/train_mac.sh's own
# hardcoded MLFLOW_TRACKING_URI: it's a public HTTPS endpoint, not a secret,
# and the mac-mps worker process (where this flow actually runs) never
# sources train_mac.sh's .env.mlflow -- only MLFLOW_TRACKING_USERNAME/
# _PASSWORD (real secrets, read implicitly by mlflow's own client from env)
# need to be in the worker's own .env.prefect.
MLFLOW_TRACKING_URI = "https://methane-detection-mlflow.ghostface.tech"

MODEL_NAME = "starcop-baseline-mag1c-rgb"
RUN_ID_SENTINEL_PREFIX = "MLFLOW_RUN_ID="
GITHUB_OWNER = "douglas-martins"
GITHUB_REPO = "methane-detection"
CD_DISPATCH_URL = (
    f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/actions/workflows/cd.yml/dispatches"
)
PUSHOVER_MESSAGES_URL = "https://api.pushover.net/1/messages.json"


def parse_run_id(stdout: str) -> str:
    """Extracts the MLFLOW_RUN_ID sentinel train.py prints on success.

    Scans line-by-line rather than a single regex over the whole blob --
    train.py's stdout also carries heavy Hydra/Lightning logging noise.
    """
    for line in stdout.splitlines():
        if line.startswith(RUN_ID_SENTINEL_PREFIX):
            return line[len(RUN_ID_SENTINEL_PREFIX) :]
    raise RuntimeError("MLFLOW_RUN_ID sentinel not found in training output")


def pull_dataset(repo_root: Path, cmd_runner: Callable = subprocess.run) -> None:
    """Pulls the dataset via the repo's own DVC remote (Google Drive).

    Assumes the worker machine's .dvc/config.local already has the
    service-account credentials configured -- this function does not
    provision or check for them (see D-01's readiness-review note).
    """
    dvc_bin = str(repo_root / ".venv" / "bin" / "dvc")
    result = cmd_runner([dvc_bin, "pull"], cwd=repo_root)
    if result.returncode != 0:
        raise RuntimeError(f"dvc pull failed with exit code {result.returncode}")


def run_training(repo_root: Path, cmd_runner: Callable = subprocess.run) -> str:
    """Runs training as a single blocking subprocess (D-10: one step, not
    submit-then-poll) and returns the MLflow run_id it produced."""
    train_script = str(repo_root / "scripts" / "train_mac.sh")
    result = cmd_runner([train_script], cwd=repo_root, capture_output=True, text=True)
    if result.returncode != 0:
        stderr_tail = "\n".join(result.stderr.splitlines()[-40:])
        raise RuntimeError(f"training failed with exit code {result.returncode}\n{stderr_tail}")
    return parse_run_id(result.stdout)


def promote(tracking_uri: str, run_id: str, model_name: str = MODEL_NAME):
    """Promotes `run_id` via promote_model's own decide_and_promote.

    The explicit model_name pin matters: promote_model's own default
    (DEFAULT_MODEL_NAME = "methane-cnn-starcop") is a different, unused
    registry name -- omitting it here would promote a model nothing ever
    serves (readiness review point 11).
    """
    client = MlflowClient(tracking_uri=tracking_uri)
    return promote_model.decide_and_promote(client, run_id=run_id, model_name=model_name)


def trigger_cd(token: str, ref: str = "main", http_post: Callable = requests.post) -> None:
    """Fires cd.yml's workflow_dispatch endpoint -- the only CD trigger
    that bypasses cd.yml's file-diff guard (readiness review point 15)."""
    response = http_post(
        CD_DISPATCH_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        },
        json={"ref": ref},
        timeout=30,
    )
    response.raise_for_status()


def notify(
    user_key: str, api_token: str, message: str, http_post: Callable = requests.post
) -> None:
    """Sends a run-summary notification via the project's existing
    Pushover account (readiness review point 17)."""
    response = http_post(
        PUSHOVER_MESSAGES_URL,
        data={"token": api_token, "user": user_key, "message": message},
        timeout=30,
    )
    response.raise_for_status()


def build_failure_message(step: str, error: Exception) -> str:
    return f"Retraining flow failed at step '{step}': {error}"


def build_notification_message(outcome) -> str:
    if outcome.stage is None:
        reasons = "; ".join(outcome.decision.reasons)
        return f"Retraining run {outcome.run_id}: NOT promoted. Reasons: {reasons}"
    return (
        f"Retraining run {outcome.run_id}: promoted to {outcome.stage} as "
        f"{outcome.model_version.name} v{outcome.model_version.version}. CD triggered."
    )


def run_retraining_cycle(
    repo_root: Path,
    tracking_uri: str,
    github_token: str,
    pushover_user_key: str,
    pushover_api_token: str,
    pull_fn: Callable = pull_dataset,
    train_fn: Callable = run_training,
    promote_fn: Callable = promote,
    trigger_cd_fn: Callable = trigger_cd,
    notify_fn: Callable = notify,
):
    """Plain-Python orchestration of the retraining cycle.

    Kept free of any Prefect decorator so it's directly unit-testable
    (Test Size: Small) with fakes injected for every side-effecting step.
    retrain_flow (the real @flow, below) calls this with @task-wrapped
    versions of each fn so Prefect's UI still shows per-step run history
    in production.
    """
    current_step = "pull_dataset"
    try:
        pull_fn(repo_root)

        current_step = "run_training"
        run_id = train_fn(repo_root)

        current_step = "promote"
        outcome = promote_fn(tracking_uri, run_id, MODEL_NAME)

        if outcome.stage is not None:
            current_step = "trigger_cd"
            trigger_cd_fn(github_token)
    except Exception as exc:
        notify_fn(pushover_user_key, pushover_api_token, build_failure_message(current_step, exc))
        raise

    notify_fn(pushover_user_key, pushover_api_token, build_notification_message(outcome))
    return outcome


@task
def pull_dataset_task(repo_root: Path) -> None:
    pull_dataset(repo_root)


@task
def run_training_task(repo_root: Path) -> str:
    return run_training(repo_root)


@task
def promote_task(tracking_uri: str, run_id: str, model_name: str = MODEL_NAME):
    return promote(tracking_uri, run_id, model_name)


@task
def trigger_cd_task(token: str) -> None:
    trigger_cd(token)


@task
def notify_task(user_key: str, api_token: str, message: str) -> None:
    notify(user_key, api_token, message)


@flow(name="retrain")
def retrain_flow() -> None:
    run_retraining_cycle(
        repo_root=_REPO_ROOT,
        tracking_uri=MLFLOW_TRACKING_URI,
        github_token=os.environ["GITHUB_ACTIONS_PAT"],
        pushover_user_key=os.environ["PUSHOVER_USER_KEY"],
        pushover_api_token=os.environ["PUSHOVER_API_TOKEN"],
        pull_fn=pull_dataset_task,
        train_fn=run_training_task,
        promote_fn=promote_task,
        trigger_cd_fn=trigger_cd_task,
        notify_fn=notify_task,
    )


if __name__ == "__main__":
    retrain_flow()

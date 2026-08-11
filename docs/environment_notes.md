# Environment Notes

## Environment A — MLflow client (TASK-2.2)

`vendor/starcop/scripts/train.py` (Environment A, `vendor/starcop/.venv`) logs
every training run to the MLflow server deployed in TASK-2.1
(`https://mlflow.ghostface.tech`).

### Install

The MLflow client is layered on top of the submodule's own pinned
dependencies, same pattern as `requirements/env-a-dev.txt`:

```bash
uv pip install --python vendor/starcop/.venv/bin/python \
  -r vendor/starcop/requirements.txt \
  -r requirements/env-a-mlflow.txt
```

`requirements/env-a-mlflow.txt` pins `protobuf<4` — `mlflow-skinny` accepts
`protobuf>=3.12,<8`, but `wandb==0.13.3` (pinned in
`vendor/starcop/requirements.txt`, kept alongside MLflow per TASK-0.2) ships
protobuf-generated files that only load under `protobuf<4`. An unpinned
install resolves to the newest protobuf satisfying mlflow alone (6.x) and
breaks wandb at import time (`TypeError: Descriptors cannot be created
directly`). Installing `requirements/env-a-mlflow.txt` keeps both loggers
importable together.

### Required environment variables

Not committed anywhere — export these locally (shell profile, or the
per-machine launch scripts from TASK-3.3 once they exist) before running
`train.py`:

| Variable | Purpose |
|---|---|
| `MLFLOW_TRACKING_URI` | `https://mlflow.ghostface.tech` |
| `MLFLOW_TRACKING_USERNAME` | MLflow basic-auth username (the server requires auth — see TASK-2.1) |
| `MLFLOW_TRACKING_PASSWORD` | MLflow basic-auth password |

These are the MLflow client's own standard environment variables — read
automatically by `mlflow.start_run()`/`MLFlowLogger`, no code needed to
consume them. If any of the three is unset, the training run will fail at
the first MLflow API call rather than silently training without tracking.

**A fourth, separate set is needed for artifact uploads** (checkpoint, confusion
matrix PNG, prediction images) — found while running TASK-2.2's live validation:
MLflow's artifact store is Backblaze B2, S3-compatible (TASK-2.1 decision D-02).
The tracking-auth vars above only cover metrics/params/tags API calls; uploading
an artifact goes over boto3's S3 client *directly to B2*, client-side, and boto3
doesn't know about `MLFLOW_TRACKING_*` at all — without these, artifact logging
fails with `botocore.exceptions.NoCredentialsError` mid-run (metrics/params still
log fine up to that point, which is what makes this easy to miss until an artifact
is actually logged):

| Variable | Purpose |
|---|---|
| `MLFLOW_S3_ENDPOINT_URL` | `https://s3.<region>.backblazeb2.com` — same value as `deploy/mlflow/.env.example`'s `MLFLOW_S3_ENDPOINT_URL` on the server |
| `AWS_ACCESS_KEY_ID` | The B2 Application Key ID (server's `.env.example` calls this `B2_APPLICATION_KEY_ID` — same value, but boto3 (used client-side) only recognizes the standard `AWS_*` names, not B2's) |
| `AWS_SECRET_ACCESS_KEY` | The B2 Application Key (server's `B2_APPLICATION_KEY`) |

Same B2 Application Key already provisioned for the MLflow server in TASK-2.1 —
no new key needs creating, just exporting client-side under boto3's expected names.

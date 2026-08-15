# MLflow Tracking Server — Coolify Deployment

Deploys MLflow (Postgres backend store + Backblaze B2 artifact store, MLflow's built-in
basic-auth app) as a Coolify-managed Docker Compose resource. Decision rationale (D-02):
[`mlops-methane-detection-plan.md`, TASK-2.1](../../mlops-methane-detection-plan.md).

Live at `https://mlflow.ghostface.tech` since 2026-08-10.

## Prerequisites

- A Backblaze B2 bucket + Application Key scoped to it (not the master key).
- Coolify running on the VPS with a domain available for the `mlflow` service.

## Import steps

1. Copy `.env.example` to `.env`, fill in real values. **`POSTGRES_PASSWORD` and
   `MLFLOW_ADMIN_PASSWORD` must be hex-only** (e.g. `openssl rand -hex 24`) — Compose
   interpolates `$VAR` across the whole file including values pulled from `.env`, so a
   password containing a literal `$` (common from generated passwords) gets partially
   eaten as a bogus variable reference before the container starts. Hex also sidesteps
   `@`/`:`/`/` breaking the `postgresql://` connection URI itself.
2. In Coolify: **New Resource → Docker Compose**, point it at this repo's
   `deploy/mlflow/docker-compose.yml`, set the env vars from your `.env`.
3. Assign a subdomain with TLS, e.g. `mlflow.ghostface.tech`, and set that same value
   as `MLFLOW_DOMAIN` in the env vars — MLflow 3.5+'s Host-header validation
   (`MLFLOW_SERVER_ALLOWED_HOSTS`) rejects requests otherwise ("Invalid Host header -
   possible DNS rebinding attack detected").
4. Deploy. Confirm both `postgres` (healthcheck: `pg_isready`) and `mlflow` containers
   report healthy in the Coolify UI.

## Auth

MLflow's own built-in basic-auth app (`mlflow server --app-name basic-auth`), backed by
the same Postgres database — not Coolify/Traefik proxy-level auth. `MLFLOW_ADMIN_USERNAME`
/`MLFLOW_ADMIN_PASSWORD` only seed the one admin account on first boot; they are not
re-read on subsequent restarts. **Rotate the admin password after first deploy** —
Settings → user menu in the UI, or `POST /api/2.0/mlflow/users/update-password`.

## Validation

```bash
curl --fail -sS -u admin:<password> https://mlflow.ghostface.tech/api/2.0/mlflow/experiments/search
# -> 200 with a JSON experiment list (or empty list on a fresh server)
```

- UI loads at `https://mlflow.ghostface.tech`, login with the admin account works.
- No errors in `mlflow`/`postgres` container logs.
- A smoke-test run's artifacts land in the B2 bucket under `mlflow-artifacts/` (confirms
  the S3-compatible artifact store path end to end, not just the Postgres backend).

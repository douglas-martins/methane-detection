# MLflow Tracking Server — Coolify Deployment

Deploys MLflow (Postgres backend store + Backblaze B2 artifact store, MLflow's built-in
basic-auth app) as a Coolify-managed Docker Compose resource. Decision rationale (D-02):
[`mlops-methane-detection-plan.md`, TASK-2.1](../../mlops-methane-detection-plan.md).

Live at `https://methane-detection-mlflow.ghostface.tech` since 2026-08-10.

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
3. Assign a subdomain with TLS, e.g. `methane-detection-mlflow.ghostface.tech`, and set that same value
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

**`MLFLOW_SERVER_CORS_ALLOWED_ORIGINS` is required** — found the hard way 2026-08-17:
without it, the UI's own browser-originated `POST` calls (e.g. `runs/search`, used by every
Runs/Models list view) get rejected outright with a literal `403 Forbidden` body reading
"Cross-origin request blocked" — even for the true `is_admin` account, and even though the
request is genuinely same-origin. Root cause, confirmed by reading `v3.15.1`'s installed
`mlflow/server/security.py`/`security_utils.py` directly: MLflow's `block_cross_origin_state_changes`
security middleware rejects any state-changing request (`POST`/`PUT`/`PATCH`/`DELETE`) that
carries a non-empty `Origin` header unless that origin is explicitly allow-listed via
`MLFLOW_SERVER_CORS_ALLOWED_ORIGINS` — with it unset, the allow-list is empty and *every*
such request is blocked, regardless of auth/RBAC/`is_admin` status entirely (this check runs
in a separate `before_request` hook, ahead of and independent from the auth plugin). `curl`
never reproduces this, since curl doesn't send an `Origin` header by default — only a real
browser XHR/fetch call does. Same fix shape as `deploy/prefect/docker-compose.yml`'s
`PREFECT_SERVER_CORS_ALLOWED_ORIGINS`: set to the full origin (scheme + host), i.e.
`https://${MLFLOW_DOMAIN}`, not just the bare hostname `MLFLOW_SERVER_ALLOWED_HOSTS` uses.
(A red herring ruled out along the way: MLflow's newer "workspaces" RBAC layer and its
`grant_default_workspace_access` setting — confirmed via the live `/api/3.0/mlflow/server-info`
endpoint that `workspaces_enabled: false` on this deployment, so that setting was never in play.)

## Validation

```bash
curl --fail -sS -u admin:<password> https://methane-detection-mlflow.ghostface.tech/api/2.0/mlflow/experiments/search
# -> 200 with a JSON experiment list (or empty list on a fresh server)
```

- UI loads at `https://methane-detection-mlflow.ghostface.tech`, login with the admin account works.
- No errors in `mlflow`/`postgres` container logs.
- A smoke-test run's artifacts land in the B2 bucket under `mlflow-artifacts/` (confirms
  the S3-compatible artifact store path end to end, not just the Postgres backend).

# Prefect Orchestration Server — Coolify Deployment

Deploys a Prefect 3 server (Postgres backend store, native Basic Auth) as a
Coolify-managed Docker Compose resource, mirroring
[`deploy/mlflow/`](../mlflow/)'s and [`deploy/bentoml/`](../bentoml/)'s
established pattern. Full task spec, readiness review, and the D-10 decision
this deployment implements: [`mlops-methane-detection-plan.md`,
TASK-7.1](../../mlops-methane-detection-plan.md).

## Prerequisites

- Coolify running on the VPS with a domain available for the `prefect`
  service.
- No dependency on D-04 (retraining trigger strategy) — this deploys the
  orchestration server itself, not a specific flow.

## What's checked into this directory

- `docker-compose.yml` — `postgres` (own instance, not shared with
  `deploy/mlflow/`'s) + `prefect` (pinned to `3.7.7-python3.12`, exactly
  matching the `prefect` version already locked in the repo's root
  `uv.lock`/Environment B, so the server and every client — CLI, workers,
  flow runs — speak the same API version).
- `.env.example` — credentials and domain, see comments for where each
  value comes from.

## Import steps

1. Copy `.env.example` to `.env`, fill in real values. **`POSTGRES_PASSWORD`
   and the password half of `PREFECT_API_AUTH_STRING` must be hex-only**
   (e.g. `openssl rand -hex 24`) — same Compose `$VAR`-interpolation
   footgun as `deploy/mlflow/`'s README documents.
2. In Coolify: **New Resource → Docker Compose**, point it at this repo's
   `deploy/prefect/docker-compose.yml`, set the env vars from your `.env`.
3. Assign a subdomain with TLS, e.g. `methane-detection-prefect.ghostface.tech`
   (Coolify's automatic Let's Encrypt), and set that same value as
   `PREFECT_DOMAIN` in the env vars.
4. Deploy. Confirm both `postgres` (healthcheck: `pg_isready`) and `prefect`
   (healthcheck: `GET /api/health`) containers report healthy in the
   Coolify UI.

## Auth

**Correction to an earlier pass of the plan**: TASK-7.1's readiness review
originally stated Prefect's self-hosted OSS server has no built-in
authentication and recommended a Traefik/Coolify Basic Auth proxy
middleware as a workaround. That turned out to be wrong for the actual
installed version — confirmed by reading
`prefect/server/api/server.py` (`PREFECT_SERVER_API_AUTH_STRING`,
`token_validation` middleware) directly, not from general knowledge, and
then live-verified against a real running container (see "Local
validation" below).

Prefect **does** ship a native, built-in auth gate: `PREFECT_SERVER_API_AUTH_STRING`
is a single `"username:password"`-shaped string that the server compares
verbatim (`hmac.compare_digest`) against the decoded `Authorization: Basic`
header on every API request — except `GET /api/health` and `GET /api/ready`,
which stay open for Coolify's own healthcheck. The Prefect UI reads this
same setting via its `/ui-settings` endpoint and automatically prompts the
browser for Basic Auth once it's set — no separate admin-account bootstrap
step, no proxy-level middleware needed. This is set via the
`PREFECT_API_AUTH_STRING` env var in `.env` and passed through to the
container as `PREFECT_SERVER_API_AUTH_STRING` (see `docker-compose.yml`'s
comment on that line for why the names differ).

Every client that talks to this server — the Mac/desktop worker processes
(see [`../../scripts/prefect_worker_mac.sh`](../../scripts/prefect_worker_mac.sh)),
any local `prefect` CLI session, TASK-7.2's flow code — needs the same
value set as `PREFECT_API_AUTH_STRING` (client-side setting name, distinct
from the server's `PREFECT_SERVER_API_AUTH_STRING`) alongside
`PREFECT_API_URL=https://<PREFECT_DOMAIN>/api`.

## Mac worker (D-10)

Per TASK-7.1's D-10 decision, the Mac reaches this server as a **native
Prefect worker process**, not the other way around — no SSH, no
port-forwarding, no Tailscale. `scripts/prefect_worker_mac.sh` starts a
Process-type worker against a `mac-mps` work pool, polling this server's
public API outbound-only.

**One-time setup**, once this resource is live in Coolify:

1. `.venv` (Environment B, Python 3.12) already has `prefect==3.7.7`
   installed — same version this resource's `docker-compose.yml` pins the
   server image to.
2. Create `.env.prefect` at the repo root (git-ignored, same pattern as
   `.env.mlflow` — see `training-runbook.md`'s "Credentials" section):

   ```text
   PREFECT_API_AUTH_STRING=<the same value set as PREFECT_API_AUTH_STRING in this resource's .env>
   ```

3. Run once manually to confirm it connects: `./scripts/prefect_worker_mac.sh`
   — should log `Worker ... started!` and the `mac-mps` pool should appear
   in the Prefect UI's **Work Pools** page. `Ctrl-C` to stop once confirmed.
4. Supervise it with `launchd` so it survives reboots/logouts:

   ```bash
   sed "s|__REPO_ROOT__|$(pwd)|g" scripts/com.methane-detection.prefect-worker.plist \
     > ~/Library/LaunchAgents/com.methane-detection.prefect-worker.plist
   mkdir -p logs
   launchctl load ~/Library/LaunchAgents/com.methane-detection.prefect-worker.plist
   ```

   `KeepAlive`+`RunAtLoad` in the plist restart the worker on crash and on
   every login — the tradeoff D-10 explicitly accepted (the Mac must be
   awake with this process running whenever a flow might fire) in exchange
   for not opening any inbound access to this machine. Logs land in
   `logs/prefect-worker.{,err.}log` (git-ignored).

The same pattern is meant to generalize to a future `desktop-rtx5070`
worker on the Windows/NVIDIA machine once TASK-3.1/D-06/D-07 unblock —
nothing in this resource's compose file, auth setup, or the worker script
itself is Mac-specific (the work-pool name is a plain CLI argument).

## Validation

```bash
# Health probe, unauthenticated (explicitly exempted from the auth gate):
curl --fail -sS https://methane-detection-prefect.ghostface.tech/api/health
# -> true

# Any real API call requires auth:
curl -sS -o /dev/null -w '%{http_code}\n' https://methane-detection-prefect.ghostface.tech/api/flows/count
# -> 401

curl --fail -sS -X POST -u '<your-PREFECT_API_AUTH_STRING-value>' \
  https://methane-detection-prefect.ghostface.tech/api/flows/count
# -> 200, a JSON integer
```

- UI loads at `https://methane-detection-prefect.ghostface.tech`, browser
  prompts for Basic Auth, logging in with the `PREFECT_API_AUTH_STRING`
  value (as `username:password`, split on the first `:`) works.
- No errors in `prefect`/`postgres` container logs.
- `prefect worker start` on the Mac (see above) shows the `mac-mps` worker
  as online in the UI's **Work Pools** page.
- A trivial test flow submitted to the `mac-mps` work pool runs to
  `Completed`.

## Local validation (done before any Coolify import)

Before writing this README, the compose file was validated end to end
against a real local `docker compose up` (temporary project, torn down
after — same approach `deploy/monitoring/README.md` used):

- `docker compose config` confirmed the schema resolves correctly,
  including the `PREFECT_UI_API_URL`/`PREFECT_SERVER_API_AUTH_STRING`
  interpolation.
- A real `docker compose up` pulled `prefecthq/prefect:3.7.7-python3.12`
  (confirmed to exist on Docker Hub before pinning it) and both containers
  reported `healthy` immediately, with clean startup logs.
- `GET /api/health` → `200` with no `Authorization` header.
- `POST /api/flows/count` with no `Authorization` header → `401`.
- The same request with the correct `PREFECT_API_AUTH_STRING` value as
  HTTP Basic Auth → `200`.
- `GET /ui-settings` → `{"api_url": "https://methane-detection-prefect.ghostface.tech/api", ..., "auth": "BASIC", ...}`,
  confirming both `PREFECT_UI_API_URL` and the auth gate are wired
  correctly from the UI's own perspective, not just the raw API's.

Not yet done (needs the real VPS/Coolify and a real Mac worker, out of
reach from this local validation): the actual Coolify import, DNS/TLS for
`methane-detection-prefect.ghostface.tech`, and a live `mac-mps` worker
run.

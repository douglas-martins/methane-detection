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
   (Coolify's automatic Let's Encrypt), **and separately set that same
   value as `PREFECT_DOMAIN` in the env vars** — Coolify's Domains field
   and this resource's environment variables are not the same thing, and
   Coolify does not populate one from the other. Skipping this step used
   to fail silently (see below); `docker-compose.yml` now refuses to
   start at all with a clear error if `PREFECT_DOMAIN` is missing.
4. Deploy. Confirm both `postgres` (healthcheck: `pg_isready`) and `prefect`
   (healthcheck: `GET /api/health`) containers report healthy in the
   Coolify UI.

**If the dashboard shows "Can't connect to Server API at `https:///api`"**
(note the missing hostname): `PREFECT_DOMAIN` isn't actually set in this
resource's env vars, even if a domain is assigned in Coolify's Domains
field for TLS routing — those are two separate settings (step 3 above).
Hit for real on the first live import. `docker-compose.yml`'s `:?` guard
on `PREFECT_DOMAIN` now makes this fail the deploy outright instead of
starting a "healthy" container with a broken UI, but if you're on an
older image tag from before that guard existed: set `PREFECT_DOMAIN` in
Coolify's env vars, then **Redeploy — not just Restart**, since env var
changes only take effect on a full redeploy (same gotcha
`deploy/monitoring/README.md` already documents).

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

Two more settings harden this beyond the auth string alone, both defense
in depth rather than the primary gate: `PREFECT_SERVER_CSRF_PROTECTION_ENABLED=true`
requires a `Prefect-Csrf-Token` header (fetched from `GET /csrf-token`,
itself auth-protected) on every `POST`/`PUT`/`PATCH`/`DELETE` — real
clients (workers, CLI, flow runs) handle this transparently via
`PREFECT_CLIENT_CSRF_SUPPORT_ENABLED` (on by default), it only affects
hand-rolled requests like a raw curl `POST`. `PREFECT_SERVER_CORS_ALLOWED_ORIGINS=https://${PREFECT_DOMAIN}`
replaces the image's wide-open `*` default, so a browser tab on some
unrelated origin can't script cross-origin requests against this server
using a visitor's already-authenticated session.

## Mac worker (D-10)

Per TASK-7.1's D-10 decision, the Mac reaches this server as a **native
Prefect worker process**, not the other way around — no SSH, no
port-forwarding, no Tailscale. `scripts/prefect_worker_mac.sh` starts a
Process-type worker against a `mac-mps` work pool, polling this server's
public API outbound-only.

### Rule: never rely on ambient PATH or env vars in code a flow can reach

**The worker's environment is not your interactive shell's environment.**
It's a `launchd`-supervised process (see the "Mac worker: install as a
launchd service" section below) — `launchctl print
gui/$(id -u)/com.methane-detection.prefect-worker` shows its real `default
environment` is just `PATH => /usr/bin:/bin:/usr/sbin:/sbin`, launchd's own
minimal default. It does **not** inherit `~/.zshrc`/`~/.zprofile`, `asdf`,
Homebrew's PATH additions, or anything else a normal terminal session sets
up. Every subprocess a flow spawns — directly, or transitively through a
script it shells out to — inherits this same restricted environment.

Hit for real, twice, building `flows/eval_baseline.py` (Phase 4 of
`track-a-paper-benchmark-reproduction-plan.md`, 2026-08-22): a bare `"uv"`
subprocess call raised `FileNotFoundError` (it lives at `~/.local/bin/uv`,
outside this PATH), and a Python-level constant (`MLFLOW_TRACKING_URI`,
set the same way `retrain.py` already did for its own direct `MlflowClient`
calls) never reached a child subprocess because a constant in one
process's memory isn't automatically an environment variable in another's.

**The rule, going forward, for any script reachable from a flow**:

- Never invoke a bare command name (`"uv"`, `"git"`, `"dvc"`) and rely on
  PATH to find it. Either build an absolute path explicitly (e.g.
  `repo_root / ".venv" / "bin" / "dvc"`, `retrain.pull_dataset`'s own
  pattern), or resolve it defensively with a `shutil.which(...)`-first,
  known-fallback-second helper (see
  `paper_eval_mlflow.resolve_uv_binary`/`resolve_git_binary` for the
  pattern to copy).
- Never assume a Python-level constant or an already-imported module's
  state reaches a subprocess. If a subprocess needs an environment
  variable, build its `env=` dict explicitly
  (`{**os.environ, "THE_VAR": the_value}`) at the call site — don't just
  set the constant somewhere else and hope it propagates.
- Prefer explicit `env=` construction close to the actual `subprocess.run`/
  `Popen` call, not a module-level `os.environ[...] = ...` mutation
  elsewhere, so it's obvious at the call site exactly what the child
  process needs and gets.
- When a new secret genuinely does need to reach every flow-spawned
  subprocess (not just one call site), add it to `.env.prefect` — the one
  file `prefect_worker_mac.sh` sources with `set -a` before starting the
  worker, so every subprocess inherits it automatically (see
  `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`'s entry below for the
  precedent). Don't invent a second way to get secrets to a worker
  subprocess.

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
   launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.methane-detection.prefect-worker.plist
   ```

   `bootstrap`/`bootout` are Apple's current recommended subcommands (the
   older `load`/`unload` are legacy — `man launchctl`); use them for this
   agent throughout, not just here. `KeepAlive`+`RunAtLoad` in the plist
   restart the worker on crash and on every login — the tradeoff D-10
   explicitly accepted (the Mac must be awake with this process running
   whenever a flow might fire) in exchange for not opening any inbound
   access to this machine. Logs land in `logs/prefect-worker.{,err.}log`
   (git-ignored).

   **Check its actual state**, don't assume from `launchctl list | grep
   ...` alone — the label is `com.methane-detection.prefect-worker` (a dot
   before `prefect-worker`, easy to mistype as a hyphen and get a false
   "not running"):

   ```bash
   launchctl print gui/$(id -u)/com.methane-detection.prefect-worker
   # state = running / pid = ... / last exit code = 0 means it's healthy
   ```

   If `bootstrap` (or the legacy `load`) fails with `Load failed: 5: Input/
   output error`, that almost always means it's *already* loaded, not that
   something's broken — launchd refuses to double-register a label.
   Confirm with `launchctl print` above before assuming anything is wrong.

   **It's meant to run forever** — this isn't something to start/stop per
   flow run, it's meant to sit there polling so it's ready whenever a flow
   fires next, the same way any always-on service works. It's cheap to
   leave running (a poll loop, ~0% CPU, well under 100MB RSS observed after
   several days up), not a per-run cost. A plain `kill` on its PID won't
   stop it either — `KeepAlive` just restarts it within seconds. To stop it
   deliberately:

   ```bash
   # Stops it now, but it comes back at your next login -- the plist is
   # still in ~/Library/LaunchAgents/, which gets rescanned and reloaded
   # on every login (the same mechanism that makes it survive reboots).
   launchctl bootout gui/$(id -u)/com.methane-detection.prefect-worker

   # Actually permanent -- stops it now AND at every future login:
   launchctl bootout gui/$(id -u)/com.methane-detection.prefect-worker
   rm ~/Library/LaunchAgents/com.methane-detection.prefect-worker.plist
   ```

The same pattern is meant to generalize to a future `desktop-rtx5070`
worker on the Windows/NVIDIA machine once TASK-3.1/D-06/D-07 unblock —
nothing in this resource's compose file, auth setup, or the worker script
itself is Mac-specific (the work-pool name is a plain CLI argument).

## TASK-7.2 setup — retraining flow prerequisites

`flows/retrain.py` (D-04: cron trigger, weekly) and its `prefect.yaml`
deployment definition are checked in, but three things need manual,
one-time setup outside this repo before the deployment can run for real.
The deployment's `schedules[0].active` is `false` in `prefect.yaml` on
purpose — leave it that way until you've done a manual
`prefect deployment run 'retrain/retrain-weekly'` and watched it succeed.

1. **Google service account for unattended `dvc pull`** (D-01's
   unattended-pull decision — see `mlops-methane-detection-plan.md`):
   - GCP Console → same project as the existing DVC OAuth client → **IAM &
     Admin → Service Accounts → Create Service Account**. No project roles
     needed — Drive access is granted by sharing the folder, not IAM.
   - **Keys → Add Key → Create new key → JSON**, download it, and place it
     somewhere durable on the Mac (not inside the repo — e.g.
     `~/.config/methane-detection/gdrive-service-account.json`).
   - Open Google Drive, find the folder DVC's `gdrive://` remote points at
     (`.dvc/config`'s `url` field has the folder ID), **Share** it with the
     service account's email (looks like
     `<name>@<project-id>.iam.gserviceaccount.com`) as at least Viewer.
   - Point DVC at it locally:

     ```bash
     dvc remote modify --local gdrive gdrive_use_service_account true
     dvc remote modify --local gdrive gdrive_service_account_json_file_path \
       ~/.config/methane-detection/gdrive-service-account.json
     dvc pull  # should now run with no browser prompt at all
     ```

     This writes to `.dvc/config.local` (already git-ignored, same file the
     interactive OAuth client's `gdrive_client_id`/`gdrive_client_secret`
     live in today).

2. **GitHub PAT for the CD trigger** (readiness review point 15 — `cd.yml`'s
   own file-diff guard means `workflow_dispatch` is the only trigger that
   actually redeploys on a retraining run):
   - GitHub → Settings → Developer settings → **Fine-grained tokens** →
     Generate new token, scoped to this repo only, **Actions: Read and
     write** permission.
   - Add it to `.env.prefect` (git-ignored, repo root — the same file
     `scripts/prefect_worker_mac.sh` already sources):

     ```text
     GITHUB_ACTIONS_PAT=<the token>
     ```

3. **Pushover credentials for the notify step** (readiness review point 17
   — reuses the same account already wired for Grafana alerts and Coolify
   deploys, see `deploy/monitoring/.env.example`): add the same
   `PUSHOVER_API_TOKEN`/`PUSHOVER_USER_KEY` values to `.env.prefect`.

`.env.prefect` should now have four lines: `PREFECT_API_AUTH_STRING`
(already there per the Mac worker setup above), `GITHUB_ACTIONS_PAT`,
`PUSHOVER_API_TOKEN`, `PUSHOVER_USER_KEY`. Since `prefect_worker_mac.sh`
sources this file with `set -a` before starting the worker, every flow run
subprocess the worker spawns inherits all four automatically — no separate
secrets mechanism needed. The flow also needs `MLFLOW_TRACKING_URI`
already required by `scripts/train_mac.sh`/`.env.mlflow`.

**`flows/eval_baseline.py` (track-a-paper-benchmark-reproduction-plan.md
Phase 4) needs two more lines in `.env.prefect`**: `AWS_ACCESS_KEY_ID` and
`AWS_SECRET_ACCESS_KEY` (the B2 artifact-store credentials, same values as
`.env.mlflow`). Unlike `retrain.py`, which shells out to `train_mac.sh` (a
shell script that self-sources `.env.mlflow` independently), this flow
shells out to plain Python CLIs (`scripts/run_starcop_baseline_evaluation.py`,
`scripts/run_live_verify.py`) that only ever see whatever environment the
worker process itself already has — so these two credentials have to live
in `.env.prefect` directly, the same way `GITHUB_ACTIONS_PAT`/Pushover
creds already do.

**Editing `.env.prefect` does nothing to an already-running worker — it
only gets read at process startup.** Hit for real, 2026-08-22: added
`AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` to the file, then a flow run
still failed with `botocore.exceptions.NoCredentialsError: Unable to
locate credentials` — the worker process had been running since the
previous Monday, five days before that edit, so it was still running with
its original (credential-less) environment; the file on disk had changed,
the live process hadn't. Confirmed directly: `ps eww <pid> | grep AWS_`
showed nothing on the stale process. **Any `.env.prefect` change requires
bouncing the worker to take effect** — same `bootout`/`bootstrap` sequence
as the "Check its actual state" / stop instructions in "Mac worker (D-10)"
step 4 above:

```bash
launchctl bootout gui/$(id -u)/com.methane-detection.prefect-worker
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.methane-detection.prefect-worker.plist
# Verify the new process actually picked up the change:
launchctl print gui/$(id -u)/com.methane-detection.prefect-worker | grep pid
ps eww <the-new-pid> | tr ' ' '\n' | grep AWS_
```

**Deploy and validate manually before trusting the schedule:**

```bash
.venv/bin/prefect deploy --all          # registers retrain-weekly (inactive) and eval-baseline
.venv/bin/prefect deployment run 'retrain/retrain-weekly'
.venv/bin/prefect deployment run 'eval-baseline/eval-baseline'
```

Watch each run to `Completed` in the UI. Once you're satisfied,
flip `retrain-weekly`'s schedule on with
`prefect deployment schedule resume 'retrain/retrain-weekly'` —
`eval-baseline` has no schedule to flip; it's on-demand only, triggered
manually whenever the vendor pin, dataset snapshot, or eval code changes.

## Validation

```bash
# Health probe, unauthenticated (explicitly exempted from the auth gate):
curl --fail -sS https://methane-detection-prefect.ghostface.tech/api/health
# -> true

# Any real API call requires auth:
curl -sS -o /dev/null -w '%{http_code}\n' https://methane-detection-prefect.ghostface.tech/api/admin/version
# -> 401

curl --fail -sS -u '<your-PREFECT_API_AUTH_STRING-value>' \
  https://methane-detection-prefect.ghostface.tech/api/admin/version
# -> 200, e.g. "3.7.7"
```

A plain `GET` is used above rather than a `POST` on purpose: `PREFECT_SERVER_CSRF_PROTECTION_ENABLED` (see docker-compose.yml) only guards `POST`/`PUT`/`PATCH`/`DELETE` — a raw `curl -X POST` with just Basic Auth and no `Prefect-Csrf-Token` header now correctly gets `403 Missing CSRF token`, since it isn't going through the real Prefect client library (which fetches and attaches that token automatically via `PREFECT_CLIENT_CSRF_SUPPORT_ENABLED`, on by default). The Mac/desktop workers and `prefect` CLI are unaffected — this only trips up hand-rolled `POST` requests like a manual curl smoke test.

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
- `prefect config view --show-defaults` run inside the container confirmed
  `PREFECT_SERVER_CSRF_PROTECTION_ENABLED='true' (from env)` and
  `PREFECT_SERVER_CORS_ALLOWED_ORIGINS='https://methane-detection-prefect.ghostface.tech' (from env)`
  actually took effect (not just accepted by the compose schema).
- With CSRF protection on: `GET /api/admin/version` (authenticated) → `200`;
  a raw `POST /api/flows/count` with Basic Auth but no CSRF headers → `403
  Missing CSRF token` (expected — see the Validation section above); fetching
  a token first (`GET /api/csrf-token?client=...`, itself Basic-Auth
  protected) and replaying the `POST` with `Prefect-Csrf-Token`/
  `Prefect-Csrf-Client` headers → `200`, confirming the mechanism itself
  works end to end, not just that it blocks unauthenticated requests.

Not yet done (needs the real VPS/Coolify and a real Mac worker, out of
reach from this local validation): the actual Coolify import, DNS/TLS for
`methane-detection-prefect.ghostface.tech`, and a live `mac-mps` worker
run.

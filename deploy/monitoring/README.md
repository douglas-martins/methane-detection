# Prometheus + Grafana — Coolify Deployment

Deploys Prometheus (scraping the inference API's public `/metrics`) and Grafana
(dashboards + alerting) as one Coolify-managed Docker Compose resource, mirroring
[`deploy/mlflow/`](../mlflow/)'s and [`deploy/bentoml/`](../bentoml/)'s established
pattern. Full task spec and rationale: [`mlops-methane-detection-plan.md`,
TASK-6.1](../../mlops-methane-detection-plan.md).

## Prerequisites

- TASK-5.2 complete (✅ — `https://api-methane-detection.ghostface.tech/metrics` is live
  and unauthenticated).
- A Pushover account with an Application API token (`Pushover -> Your Applications ->
  Create New Application`) and your account's User Key.

## What's checked into this directory

- `docker-compose.yml` — `prometheus` (built from `prometheus/Dockerfile`, 35-day
  retention — see its own comment for why) + `grafana` (built from
  `grafana/Dockerfile`, provisioning-file-driven, no manual UI clicks needed for
  datasource, dashboards, or the Pushover contact point).
- `prometheus/{Dockerfile,prometheus.yml}` — scrapes
  `https://api-methane-detection.ghostface.tech/metrics` over the public internet, not
  an internal Coolify network. See `prometheus.yml`'s own header comment and the plan's
  TASK-6.1 decision for the tradeoff (no auth on that endpoint today). The Dockerfile
  bakes `prometheus.yml` into the image at build time rather than bind-mounting it —
  see "A real Coolify deployment bug, and its actual fix" below for why.
- `grafana/{Dockerfile,provisioning/}` — same build-time-COPY approach for the whole
  provisioning tree:
  - `provisioning/datasources/prometheus.yml` — auto-registers the Prometheus
    datasource (`uid: prometheus`), pointing at the `prometheus` service by its plain
    Compose service name (same compose file, same default network — no UUID-suffixed
    hostname needed here, unlike cross-resource scraping).
  - `provisioning/dashboards/` — one dashboard (`json/inference-api.json`, 4 panels:
    request count, latency p50/p95/p99, prediction class distribution, error rate),
    auto-loaded into a "Methane Detection" folder.
  - `provisioning/alerting/` — the Pushover contact point, a default routing policy,
    and the detection-rate-deviation alert rule (step 4).

## Import steps

1. Copy `.env.example` to `.env`, fill in real values.
2. In Coolify: **New Resource → Docker Compose**, point it at this repo's
   `deploy/monitoring/docker-compose.yml`, set the env vars from your `.env`.
3. Assign a subdomain with TLS to the **grafana** service only, e.g.
   `grafana.ghostface.tech` — set the same value as `GRAFANA_DOMAIN`. Prometheus has no
   public domain (`expose: ["9090"]`, internal-only).
4. Deploy. Confirm both `prometheus` and `grafana` report healthy in the Coolify UI.
5. Grafana UI → **Connections → Data sources** should already show "Prometheus"
   (provisioned, not manually added) → **Alerting → Contact points** should already
   show "pushover" → **Dashboards → Methane Detection** should already show "Methane
   Detection — Inference API".
6. **Prometheus's own UI** (not public — reach it via `docker compose exec` or a
   temporary port-forward, since it's `expose`-only): **Status → Targets** should show
   the `methane-detection-api` job as `UP`.

## A real Coolify deployment bug, and its actual fix

First import failed at the mount step:

```text
error mounting ".../prometheus.yml" to rootfs at "/etc/prometheus/prometheus.yml":
... not a directory: Are you trying to mount a directory onto a file (or vice-versa)?
```

That looked like Coolify's known file-vs-directory bind-mount bug
([coollabsio/coolify#6056](https://github.com/coollabsio/coolify/issues/6056),
[#4468](https://github.com/coollabsio/coolify/issues/4468),
[#3375](https://github.com/coollabsio/coolify/issues/3375)) — a bind-mount **file**
source that hasn't been staged gets auto-created as an empty **directory**. Moving
`prometheus.yml` into its own subdirectory and mounting the directory instead
(`./prometheus:/etc/prometheus:ro`) cleared that exact error.

**But the real deploy then failed differently**: the container started, but Prometheus
crash-looped with `open /etc/prometheus/prometheus.yml: no such file or directory` —
the mount succeeded, but arrived **empty**. Checked against Coolify's own docs (Docker
Compose Build Packs, Persistent Storage): Coolify's `volumes:` bind mounts **never sync
files from the git checkout at all** — a `source: ./path` entry is either an empty
`is_directory: true` placeholder or a literal inline `content:` block written directly
into the compose YAML. There was never a mechanism that would have populated either
mount from the files actually committed in this directory.

**Actual fix**: stop bind-mounting config from the repo entirely. `prometheus/Dockerfile`
and `grafana/Dockerfile` `COPY` the real files into their respective images at build
time — one source of truth, no bind-mount/persistent-storage gap to fall into. Verified
locally end to end (`docker compose build && docker compose up`): Prometheus's `/api/v1/status/config`
reflects the real scrape config, and Grafana's API confirms the datasource, dashboard,
Pushover contact point, and alert rule are all present — not empty defaults.

**If you ever add another config file here**, bake it into the relevant service's image
via `COPY` in its Dockerfile — don't reach for a bind mount, Coolify won't populate it
from the repo.

## A real code prerequisite, not just infra

The dashboard's "Prediction class distribution" panel and the alert rule both read a
custom `methane_prediction_total` counter added to `src/serving/service.py` (TASK-6.1)
— BentoML's own built-in `/metrics` covers request count/latency/error rate but has no
visibility into what `/predict` actually returned. This is already implemented and
live-validated (`bentoml serve` + curl against a real running instance confirmed the
counter appears and increments correctly on `/metrics`) — nothing further needed here,
just noting the dependency for anyone importing this resource without having read the
plan's audit first.

## Alert rule — validate before trusting

`grafana/provisioning/alerting/rules.yml`'s detection-rate-deviation rule was **not**
live-validated against a real Grafana instance (none existed until this task's own
deploy). It's file-provisioned, so it's read-only from the Grafana UI by design —
editing it there does nothing durable (Grafana reverts UI/API edits to a
file-provisioned rule on the next reload or restart). **Always edit `rules.yml`
itself**, then apply the change one of two ways:

- **Reload without a restart**: `POST /api/admin/provisioning/alerting/reload`
  against the Grafana instance, Basic Auth with the admin account (requires Grafana
  server-admin permissions). Fastest way to pick up a `rules.yml` change.
- **Redeploy**: push the change and redeploy the Coolify resource — provisioning
  files are re-applied on every container start regardless, so this always works too,
  just slower.

On first import:

1. Confirm the rule loaded without a schema error: **Alerting → Alert rules** should
   show "Methane detection rate deviates from 30-day baseline" in the "Methane
   Detection" folder (a `file`-provisioned rule shows a small lock/provisioned icon —
   that's expected, not a fault).
2. The real rule needs 30 days of live traffic before its baseline means anything.
   Validate the *wiring* now instead: temporarily lower the threshold in `rules.yml`
   (edit the Threshold step's `params: [2]` to something a burst of test traffic will
   clearly cross) or shorten the windows, apply via reload or redeploy as above, send
   the 10+ synthetic `/predict` requests from the Validation section below, confirm a
   Pushover notification actually arrives, then restore the real `2`/`1h`/`30d` values
   in `rules.yml` and apply again.

## Validation

```bash
curl --fail -sS https://grafana.ghostface.tech/api/health
# -> 200

# Prometheus scraping the live API:
curl -s https://api-methane-detection.ghostface.tech/metrics | grep methane_prediction_total
```

- Grafana UI loads at `https://grafana.ghostface.tech`, login with the admin account
  works, and the "Methane Detection — Inference API" dashboard renders (may be mostly
  empty until real `/predict` traffic exists).
- Sending 10 real `/predict` requests against the live API produces a visible spike in
  the dashboard's "Request count" panel within one scrape interval (15s).
- A manually-lowered-threshold test (see "Alert rule — validate before trusting" above)
  successfully delivers a Pushover notification.

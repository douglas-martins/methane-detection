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

- `docker-compose.yml` — `prometheus` (35-day retention — see its own comment for why)
  + `grafana` (provisioning-file-driven, no manual UI clicks needed for datasource,
  dashboards, or the Pushover contact point).
- `prometheus.yml` — scrapes `https://api-methane-detection.ghostface.tech/metrics`
  over the public internet, not an internal Coolify network. See its own header comment
  and the plan's TASK-6.1 decision for the tradeoff (no auth on that endpoint today).
- `grafana/provisioning/datasources/prometheus.yml` — auto-registers the Prometheus
  datasource (`uid: prometheus`), pointing at the `prometheus` service by its plain
  Compose service name (same compose file, same default network — no UUID-suffixed
  hostname needed here, unlike cross-resource scraping).
- `grafana/provisioning/dashboards/` — one dashboard (`json/inference-api.json`, 4
  panels: request count, latency p50/p95/p99, prediction class distribution, error
  rate), auto-loaded into a "Methane Detection" folder.
- `grafana/provisioning/alerting/` — the Pushover contact point, a default routing
  policy, and the detection-rate-deviation alert rule (step 4).

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
deploy). On first import:

1. Open the rule in **Alerting → Alert rules**, confirm it loaded without a schema
   error, and re-**Save** it once from the UI (Grafana re-validates and re-persists on
   save, which is a cheap sanity check even without touching the config).
2. The real rule needs 30 days of live traffic before its baseline means anything.
   Validate the *wiring* now instead: temporarily lower the threshold (edit the
   Threshold step's `params: [2]` to something a burst of test traffic will clearly
   cross) or shorten the windows, save, send the 10+ synthetic `/predict` requests from
   the Validation section below, confirm a Pushover notification actually arrives, then
   put the real `2`/`1h`/`30d` values back.

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

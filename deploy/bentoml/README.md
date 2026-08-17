# BentoML Inference API — Coolify Deployment

Deploys the STARCOP methane-plume segmentation service (`src/serving/`, TASK-5.1) as a
Coolify-managed Docker Compose resource, mirroring [`deploy/mlflow/`](../mlflow/)'s
established pattern (TASK-2.1). Full task spec and rationale:
[`mlops-methane-detection-plan.md`, TASK-5.2](../../mlops-methane-detection-plan.md).

## Prerequisites

- TASK-5.1 complete (✅ — `src/serving/` built and live-validated).
- A real image at `ghcr.io/douglas-martins/methane-detection:<tag>`. TASK-4.3's `cd.yml`
  publishes this on an ongoing basis, but **the very first import needs a bootstrap
  image** since `cd.yml` doesn't exist until this resource's webhook secrets do (see
  "Bootstrapping" below).
- Backblaze B2 application key with read access to the same bucket
  `deploy/mlflow/`'s MLflow server uses for artifacts (`mlflow-artifacts/` prefix).
- Docker logged in to `ghcr.io` with a token scoped to `write:packages` — only needed
  for the one-off bootstrap push below (`cd.yml`, once it exists, authenticates itself
  via `GITHUB_TOKEN`).

## Bootstrapping (first import only)

`cd.yml` (TASK-4.3) needs this resource's Coolify webhook URL + API token to exist as
GitHub secrets before it can call them — but this resource needs a real image to pull
before Coolify will run it. Break the cycle once, by hand:

```bash
docker login ghcr.io -u <your-github-username>  # PAT with write:packages, if not already logged in
bentoml build
bentoml containerize methane_detection_service:latest \
  -t ghcr.io/douglas-martins/methane-detection:bootstrap \
  --opt platform=linux/amd64
docker push ghcr.io/douglas-martins/methane-detection:bootstrap
```

**`--opt platform=linux/amd64` is required if you're building on Apple Silicon** (or any
non-amd64 host) — `bentoml containerize`/`docker buildx` otherwise defaults to the host's
own architecture, producing an image the VPS (`linux/amd64`) can't pull at all (`no
matching manifest for linux/amd64 in the manifest list entries`).

Set `IMAGE_TAG=bootstrap` in Coolify's env vars for the first import (see below), then
switch it to `latest` once TASK-4.3's `cd.yml` is publishing real tags.

## GHCR image visibility

`GITHUB_TOKEN`-pushed GHCR packages are **private by default**, independent of the
repo's own visibility — Coolify's `docker pull` will 401 otherwise. Before or right
after the first import, do one of:

- GitHub → the `methane-detection` package → **Package settings → Change visibility →
  Public**, or
- In Coolify, add a private-registry credential (Settings → Registries, or the
  resource's own registry-auth field) using a GitHub PAT scoped to `read:packages`.

## Import steps

1. Copy `.env.example` to `.env`, fill in real values (see comments in that file for
   where each one comes from).
2. In Coolify: **New Resource → Docker Compose**, point it at this repo's
   `deploy/bentoml/docker-compose.yml`, set the env vars from your `.env`.
3. Assign a subdomain with TLS, e.g. `api-methane-detection.ghostface.tech` (Coolify's
   automatic Let's Encrypt) — kept distinct from the shorter `methane-detection-mlflow.ghostface.tech`
   since this subdomain names the project, not just the tool, to stay legible once
   other Coolify-hosted services exist on the same VPS.
4. Deploy. Confirm the container becomes healthy (`/readyz` healthcheck, see
   `docker-compose.yml`) and resource limits (2 CPU / 4 GB) show as applied under the
   resource's **Resource Limits** tab — they're set in the compose file
   (`deploy.resources.limits`), the UI should just reflect what's already there.

## Deploy webhook + API token

Needed so TASK-4.3's `cd.yml` can trigger a redeploy after pushing a new image. Full
step-by-step (API token scope, webhook URL format, `cd.yml` curl snippet, revocation)
is in the plan's TASK-5.2 section — summary:

1. **Settings → Advanced → API Access** (enable, one-time, instance-level).
2. **Keys & Tokens → API Tokens** → create one scoped to `deploy` only → copy it once
   (this is the Bearer token).
3. This resource → **Configuration → Webhooks** → copy the "Deploy Webhook (auth
   required)" URL.
4. Store both as GitHub Actions repo secrets: `COOLIFY_WEBHOOK_URL`, `COOLIFY_TOKEN`.

## Validation

```bash
curl --fail -sS https://api-methane-detection.ghostface.tech/readyz
# -> 200

curl --fail -sS -X POST https://api-methane-detection.ghostface.tech/health
# -> {"status": "ok", "model_name": "...", "model_version": "...", ...}
```

Manually triggering the webhook (`curl --fail -X GET "$COOLIFY_WEBHOOK_URL" -H "Authorization: Bearer $COOLIFY_TOKEN"`) should show a visible redeploy in the Coolify UI.

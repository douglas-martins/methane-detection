# Design Decisions — Full Rationale

> Merged from the original `mlops-methane-detection-plan.md` Section 3 (Open Decisions table) and Section 8 (Answers Log), plus the fuller rationale recorded inline in each decision's phase/task section. Outcome one-liners (no rationale) are public, on `docs/decisions.md`. **D-06 and D-07 are still open** — flagged at the top since they're this project's critical-path blocker (they gate TASK-3.1 → TASK-3.3b → the entire desktop compute branch, including Phase 7's future `desktop-rtx5070` work pool).

## Open

### D-06 — Primary training compute: local RTX 5070 vs. Google Colab

**Status**: 🔲 Undecided. **Impact**: Phase 3 workflow — which machine training work defaults to.

Not yet decided. Affects priority/whether the desktop becomes the default heavy-training path, not correctness — Colab (TASK-3.3c) isn't hard-blocked by this decision and can be built independently.

### D-07 — WSL2 vs. Arch Linux for GPU training on the Desktop

**Status**: 🔲 Undecided. **Impact**: Phase 0 environment, hard-blocks TASK-3.1 → TASK-3.3b.

Both are available on the Desktop machine; Arch has native CUDA but requires manual setup, WSL2's CUDA-on-WSL2 path is a translation layer needing `LD_LIBRARY_PATH` additions NVIDIA's guide specifies. This isn't just a setup-docs choice — it changes `train_desktop.sh`'s own env-var logic (TASK-3.3b), so the script can't be meaningfully written until this resolves.

**Documentation inconsistency worth resolving alongside this**: TASK-3.1's own heading already says "Configure RTX 5070 Training Environment (Windows/WSL2)" — presuming WSL2 — while simultaneously gated 🔒 on this still-open decision. Either D-07 was informally decided as WSL2 and this table was never updated, or the heading is presumptive and should stay generic until D-07 actually resolves.

**Relevant new input (added 2026-08 during the docs-reorg framing pass, not yet acted on)**: if FPGA-style on-board deployment work is pursued later (hls4ml is one candidate example — see `docs/methodology.md`'s Research Strategy section), its toolchain is Linux-only (no native Windows/macOS support) and would tip the scale toward a Linux-based environment on the Desktop machine regardless of which way this decision resolves for training itself. Worth factoring in when this is finally decided, not a reason to decide it now.

---

## Resolved

### D-01 — DVC remote backend

**Decided**: 2026-07-22 (initial), refined 2026-08-17 (unattended-automation gap). **Answer**: Google Drive, authenticated via a **dedicated Google Cloud OAuth client** (not DVC's shared default app).

The existing 5TB Google plan makes marginal cost $0. The rate-limit risk that made Backblaze B2 attractive is a *shared OAuth quota* problem (the default DVC/pydrive2 app's quota, shared across every DVC user on the internet), not a Drive problem — solved by registering a project-owned OAuth client, historically ~20,000 req/100s dedicated to this project. Note this and the more recent quota figures cited in the fuller comparison are not guaranteed for every project — Google has been known to grandfather prior quotas by project creation date or prior Drive API usage window — so treat them as conditional and confirm in Cloud Console before relying on them. B2 documented as the fallback, not adopted: full trade-off comparison in `internal-docs/plans/dvc-remote-comparison.md`.

**Fallback triggers** (none hit as of the full 59GB/75,353-file push): sustained 403/429 errors after switching to the dedicated OAuth client; push/pull of the full raw dataset exceeding ~2–3 hours; Drive quota pressure pushing the 5TB plan close to its cap. If triggered, B2 is a real migration, not a one-line reconfig: add B2 as a new remote alongside Drive, `dvc push -r b2` every currently-tracked object, validate a clean clone's `dvc pull -r b2` reproduces the same data, then flip the default remote and retire Drive — keeping Drive configured and populated until that validation passes. No pipeline/code changes needed either way (see `internal-docs/plans/dvc-remote-comparison.md` §4 for the full sequence).

**Unattended-automation gap, resolved 2026-08-17**: the `gdrive` remote's interactive OAuth flow is a real blocker for unattended automation specifically (confirmed: no cached token exists anywhere on the Mac that runs TASK-7.2's flow) — an unattended `dvc pull` inside a Prefect task would hang waiting for a browser consent screen that can never complete, and even a seeded token expires ~7 days while the OAuth app stays in Testing status. Considered and rejected: seeding a token and accepting the 7-day re-auth cadence (defers the same failure); Backblaze B2 (rejected on cost — not free at 59GB scale, unlike Drive's $0 marginal cost, so this doesn't reopen the original D-01 reasoning); publishing the OAuth consent screen out of Testing status (would need a Google app-verification review for uncertain payoff, Drive's scope is treated as sensitive). **Chosen fix**: a Google service account in the same GCP project as the existing OAuth client, target Drive folder shared to the service account's email, DVC's `gdrive_use_service_account` + a JSON key. Non-expiring, no new storage provider. Service accounts have no personal Drive quota, but this flow only pulls (reads from a folder the regular account owns), so that doesn't matter here. Implemented 2026-08-17, confirmed `dvc pull` exits 0 unattended.

**The "pull-only" assumption confirmed the hard way, 2026-08-24**: the service account's zero-quota limitation isn't just theoretical — `dvc push` (even from a dev machine with real processed data cached locally) fails outright under the service account: `HttpError 403 ... "Service Accounts do not have storage quota. Leverage shared drives, or use OAuth delegation instead."` This surfaced because `notebooks/train_colab.ipynb`'s TASK-3.3c spike found `normalize@starcop_mini`/`split@starcop_mini`/`patch_extract@starcop_mini` had never actually been pushed to the `gdrive` remote — present in a dev machine's local DVC cache, but not on the remote, so any pull-only worker (Colab included) got `Missing cache files ... neither locally nor on remote`. Fix applied: temporarily flip `.dvc/config.local`'s `gdrive_use_service_account` to `false` (the OAuth client_id/secret are already configured there), `dvc push` the missing stages under the real Google account that owns the Drive folder, then flip the flag back to `true` immediately after — never leave a worker machine's default push path pointed at OAuth, since that reintroduces the exact interactive-prompt problem this decision fixed. **Any future `dvc push` from any machine must go through this same temporary-OAuth dance** — the service account can never push, by Google's own design, not just this project's config.

### D-02 — MLflow artifact store location

**Decided**: 2026-08-09. **Answer**: **Backblaze B2** (S3-compatible) for artifacts, **PostgreSQL** for the backend store, deployed as a **docker-compose** stack on Coolify, auth via **MLflow's own built-in basic-auth app** (`mlflow server --app-name basic-auth`).

Docker-compose (not a single Docker-image resource) gives full control over the `mlflow` service alongside its Postgres backend. Postgres avoids SQLite write-lock contention once concurrent training runs log to the same server, and keeps the model registry robust (SQLite has no registry support at all). B2 keeps model artifacts on the same provider family as the DVC fallback (D-01) without reusing DVC's Google-Drive-specific OAuth setup — MLflow's native S3 artifact store needs no plugin. MLflow's own basic-auth (confirmed shipped natively, `mlflow/server/auth/__init__.py`) gives real per-user accounts with per-experiment permissions, travels with the server rather than being tied to Coolify, unlike a Traefik/Coolify proxy-auth layer.

### D-03 — BentoML vs. FastAPI for serving

**Decided**: 2026-08-15. **Answer**: **BentoML.**

Idiomatic `bentofile.yaml` + `bentoml build`/`bentoml containerize` path (generates the image, including a Dockerfile, from the Bento) is the reason to pick BentoML over hand-writing FastAPI + a Dockerfile in the first place. Also matches a pre-existing soft signal: `bentoml` was already pinned in root `pyproject.toml` since the initial project scaffold, unused until this decision.

### D-04 — Retraining trigger strategy

**Decided**: 2026-08-17. **Answer**: **Cron now, drift-based later.**

TASK-7.2 ships with a native Prefect deployment schedule (weekly, `0 3 * * 1`) as the trigger — zero new integration surface, unblocks implementation immediately. The Grafana-webhook-plus-shim path (reusing the live, alert-firing TASK-6.1/6.2 Prometheus/Grafana pipeline) is deferred as an explicit follow-up: self-hosted Prefect has no inbound webhook receiver (Cloud-only feature — confirmed by reading the installed `prefect==3.7.7` server's mounted routers, no `webhooks`/`/hooks/...` router exists), so a small translation shim would need to sit between Grafana's alert POST and Prefect's `create_flow_run` API — real, if modest, added complexity beyond a plain contact point. Data-based triggering (N new samples) has no supporting infra today (nothing detects new data landing in DVC/Drive) and was not pursued.

### D-05 — Additional datasets beyond STARCOP

**Decided**: 2026-05-07. **Answer**: STARCOP stays primary; candidates tracked in the Dataset Registry (`docs/dataset.md`), not pursued yet. Low urgency, still nominally open as a scope question — see `status.md`'s completion checklist.

### D-08 — Versioning & changelog automation tool

**Decided**: 2026-07-06. **Answer**: **python-semantic-release.**

Parses the Conventional Commits already used on `main`, stamps `pyproject.toml:project.version`, generates `CHANGELOG.md`, tags releases, creates GitHub Releases via the official GitHub Action. No `build_command`/PyPI publish configured — this repo is a research pipeline, not a distributed package, so PSR is used only for version stamping + changelog + tag + Release.

### D-09 — `train_mac.sh` behavior with no `WANDB_API_KEY`

**Decided**: 2026-08-17. **Answer**: **Default `WANDB_MODE=disabled`** when no `WANDB_API_KEY` is set (an explicit `WANDB_MODE` or a real key both still take precedence). MLflow logging is unaffected.

Confirmed as a real, live blocker, not theoretical: TASK-7.2's Prefect flow hit exactly this failure (`wandb.errors.UsageError: api_key not configured (no-tty)`) the first time training ran unattended — `WandbLogger` tries an interactive login prompt when no key is set, which hangs/fails with no TTY. Implemented + bats-tested in `scripts/train_mac.sh`; `train_desktop.sh`/`train_colab.ipynb` not yet touched (out of scope until Phase 3's desktop/Colab paths are built).

### D-10 — Prefect worker architecture (Phase 7, TASK-7.1 step 3 design)

**Decided**: 2026-08-16. **Answer**: **Native Process-type Prefect worker process on each compute machine, no SSH.**

The original plan's "SSH into the MacBook" step had an undocumented reachability gap (no port-forward/Tailscale/dynamic-DNS configured anywhere for the VPS to reach the Mac's SSH port over a home network) and turned out to be the wrong shape for Prefect anyway — Prefect's built-in work-pool types are Process/Docker/Kubernetes/cloud-run variants, no "SSH into a remote host" option, so the original phrasing would have meant hand-written paramiko/fabric code. Every machine that runs training instead gets its own Process-type worker (`prefect worker start`), polling the server's public API outbound-only — no VPS→machine inbound access needed. Explicitly generalized beyond just the Mac: the user will also train on a Windows desktop with an NVIDIA GPU, so the work-pool design is one pool per machine (`mac-mps`, `desktop-rtx5070`, …), not a single Mac-only pool. Trade-off accepted: each machine's worker process must be supervised (`launchd` on macOS, a Windows service/Task Scheduler entry on desktop) and awake whenever a flow might fire.

---

## Reference: Coolify deploy webhook setup (supports D-02/TASK-5.2's deploy automation)

Researched 2026-08-15 against [Coolify's official docs](https://next.coolify.io/docs/core/automation/deploy-webhooks) — kept here since it's the concrete mechanism several of the above decisions (and `docs/pipeline/ci-cd.md`'s public architecture description) depend on:

1. **Enable API access** (one-time, instance-level): Coolify dashboard → Settings → Advanced → API Access.
2. **Create a deploy-scoped API token** (one-time): Keys & Tokens → API Tokens → description (e.g. `github-actions-cd`) + expiration → permission scope **`deploy`** only → Create → copy immediately (shown once). This is the Bearer token, not the webhook URL.
3. **Get the resource's webhook URL** (once the Compose resource is imported): resource → Configuration → Webhooks → copy the "Deploy Webhook (auth required)" URL (`https://<coolify-host>/api/v1/deploy?uuid=<resource-uuid>&force=false`).
4. **Store both as GitHub Actions repo secrets**: `COOLIFY_WEBHOOK_URL` (step 3) and `COOLIFY_TOKEN` (step 2) — **both are required**, an earlier draft of this project's planning assumed the webhook URL alone was sufficient; it is not, the token authorizes the request and Coolify rejects an unauthenticated call.
5. **Call it** (GET with the token in the `Authorization` header):
   ```yaml
   - name: Trigger Coolify deploy
     run: |
       curl --fail --request GET "${{ secrets.COOLIFY_WEBHOOK_URL }}" \
         --header "Authorization: Bearer ${{ secrets.COOLIFY_TOKEN }}"
   ```
   (`--fail` makes curl exit non-zero on a non-2xx response, so a bad/expired token fails the job loudly.)
6. **Revoking**: Keys & Tokens → API Tokens → revoke + regenerate + update the `COOLIFY_TOKEN` secret; the webhook URL itself doesn't need to change.
7. This is a *resource* webhook (targets one Compose service via its `uuid`). Coolify also supports *tag* webhooks (redeploy every resource sharing a tag) — not needed here (one inference-API resource), but relevant if a staging/production pair is ever split into two resources later.

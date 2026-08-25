# Progress Tracker & Completion Checklist

> Descendant of the original `mlops-methane-detection-plan.md` Sections 1 ("Current state" bullets), 6 (Progress Tracker), and 9 (Completion Checklist). Task-level detail and full narrative: [plan.md](plan.md). Decision rationale: [decisions.md](decisions.md).

**Status legend**: 🔲 Not started · 🟡 In progress · ✅ Complete · ❌ Blocked · 🔁 Revisiting

---

## Progress Tracker

| Task ID | Description | Status | Completed | Notes |
|---|---|---|---|---|
| TASK-0.1 | Create GitHub Repository + STARCOP submodule | ✅ Complete | 2026-07-22 | Full folder scaffold, README, LICENSE in place |
| TASK-0.2 | Configure Python Environments (two uv stacks) | ✅ Complete | 2026-07-22 | Env A: Python 3.10.19/torch 1.13.1; Env B: Python 3.12.12/torch 2.12.1, MPS available |
| TASK-0.3 | Set Up Submodule Env & Validate Baseline | ✅ Complete | 2026-07-22 | HyperSTARCOP OA 0.9965/F1 0.9065/IoU 0.8290. Full report: `docs/baseline_metrics.md` |
| TASK-0.4 | Configure Python Semantic Release | ✅ Complete | 2026-07-06 | Baselined at `v0.1.0`, `allow_zero_version=true` |
| TASK-1.1 | Initialize DVC | ✅ Complete | 2026-08-04 | Both datasets tracked/pushed/verified. `models/starcop_baseline/*.ckpt` still untracked |
| TASK-1.2 | Build Preprocessing Pipeline | ✅ Complete | 2026-08-09 | 5-stage `dvc.yaml`, validated on both datasets |
| TASK-1.3 | Validate Dataset Statistics | ✅ Complete | 2026-08-09 | `docs/dataset_report.md`; imbalance ~87:1 (mini) vs ~314:1 (raw) |
| TASK-2.1 | Deploy MLflow on VPS | ✅ Complete | 2026-08-10 | Live at `methane-detection-mlflow.ghostface.tech` |
| TASK-2.2 | Integrate MLflow into Training | ✅ Complete | 2026-08-10 | Composition-only, `src/training/train.py` + 8 modules |
| TASK-2.3 | Define Registry Promotion Workflow | ✅ Complete | 2026-08-11 | `src/registry/promote_model.py` live and tested |
| TASK-3.1 | RTX 5070 Training Setup (WSL2) | 🔲 Not started | — | Blocked by D-06, D-07 |
| TASK-3.2 | M4 Pro Training Setup (MPS) | ✅ Complete | 2026-08-12 | ~2× speedup vs CPU; `resolved_device=mps:0` confirmed |
| TASK-3.3 | Training Launch Scripts (overview) | 🟡 In progress | — | Split into 3.3a/b/c, unequal readiness |
| TASK-3.3a | Mac Training Launch Script | ✅ Complete | 2026-08-12 | `scripts/train_mac.sh`, validated with a real `FINISHED` MPS run |
| TASK-3.3b | Desktop Training Launch Script | 🔲 Not started | — | Hard-blocked on TASK-3.1, itself blocked on D-06/D-07 |
| TASK-3.3c | Colab Training Launch Notebook | 🔲 Not started | — | Not hard-blocked by D-06; next step is a spike |
| TASK-4.1 | Write Unit Tests | 🟡 In progress | — | 4/5 targets done; ≥90% coverage gate blocked on a scoping decision |
| TASK-4.2 | Create CI Workflow | 🟡 Substantially complete | — | `tests.yml`+`commitlint.yml`+`lint.yml` live; PR coverage comment pending |
| TASK-4.3 | Create CD Workflow | 🟡 Implemented, not live-validated | 2026-08-15 | `.github/workflows/cd.yml` written, never exercised by a real run |
| TASK-4.4 | Automated Release Workflow | ✅ Complete | 2026-07-06 | Live through v0.9.1+; not gated on CI passing (design gap) |
| TASK-5.1 | Build Inference API | ✅ Core complete | 2026-08-15 | `src/serving/` (BentoML), live-validated. ENVI input not implemented |
| TASK-5.2 | Deploy API to VPS | ✅ Complete | 2026-08-15 | Live at `https://api-methane-detection.ghostface.tech` |
| TASK-6.1 | Deploy Prometheus + Grafana | ✅ Complete | 2026-08-16 | `deploy/monitoring/` live, carrying real traffic |
| TASK-6.2 | Input Data Drift Detection | ✅ Complete | 2026-08-16 | `src/serving/{band_baseline,drift}.py`, proven to fire a real alert |
| TASK-7.1 | Deploy Prefect Server | ✅ Complete | 2026-08-17 | Live at `https://methane-detection-prefect.ghostface.tech` |
| TASK-7.2 | Build Retraining Flow | 🟢 Functionally complete | — | Live-validated end-to-end except the CD-trigger call itself |

---

## Completion Checklist

> Derived from a full audit of the original plan document (2026-08-17). Check items off here as they close, and update the relevant task/decision doc too so this list doesn't drift out of sync.

### Open Decisions still outstanding

- [ ] **D-05** — Additional datasets beyond STARCOP (scope question, low urgency)
- [ ] **D-06** — Training target: local RTX 5070 vs. Colab as primary
- [ ] **D-07** — WSL2 vs. Arch Linux for GPU training on Desktop
  - D-06/D-07 are the critical-path blocker: they gate TASK-3.1 → TASK-3.3b → the entire desktop compute branch, including Phase 7's future `desktop-rtx5070` work pool.

### Phase 3 remaining — Training Pipeline

- [ ] **TASK-3.1** — RTX 5070/WSL2 environment setup (not started, blocked on D-06/D-07)
- [ ] **TASK-3.3b** — `train_desktop.sh` (blocked on TASK-3.1)
- [ ] **TASK-3.3c** — `train_colab.ipynb` (not started; next step is the Python/torch-compat + DVC OAuth device-flow spike, not hard-blocked by D-06)

### Phase 4 remaining — CI/CD

- [ ] **TASK-4.1** — Resolve the `train.py`/`prepare_data()` coverage-gate exclusion decision so the ≥90% target measures the right denominator; optionally close the narrower `test_dataset_loader.py` gap
- [ ] **TASK-4.2** — Execute `internal-docs/plans/codecov-coverage-upload.md` (PR coverage comments)
- [ ] **TASK-4.2** — Enable branch-protection required status checks on `main` (currently none of `tests`/`lint`/`commitlint` can block a merge)
- [ ] **TASK-4.2** — Fix the 71-finding Ruff baseline, especially the 2 `F811` shadowed duplicate test functions in `test_normalize.py` that currently never run
- [ ] **TASK-4.3** — Run `cd.yml` for real via `workflow_dispatch` to validate build → push → webhook → smoke test end-to-end; confirm the GH-hosted runner's disk margin holds for the ~10GB image
- [ ] **TASK-4.4** — Gate `release.yml` on `tests.yml` passing (currently just serialized against it, not conditioned on green)

### Phase 5 remaining — Serving

- [ ] **TASK-5.1** — ENVI-format `/predict` input support (currently `.npy` only)
- [x] **TASK-5.1** — `train.py`'s `mlflow.pytorch.log_model` call site now detects `serialization_format` support via `inspect.signature` instead of always passing it — found live on Colab (2026-08-24): Environment A's pinned `mlflow<3.7` predates that kwarg entirely, so the unconditional pin from the note below broke it, forwarding the unrecognized kwarg straight into `torch.save()`. `hf_baseline_import.py`'s call site still passes it unconditionally (fine there — only ever run under Environment B's newer mlflow).
- [ ] **TASK-5.2** — Add explicit `deploy.resources.limits` to `deploy/mlflow/docker-compose.yml` (only service still missing them)

### Phase 6 remaining — Monitoring

- [ ] TASK-6.1's detection-rate-deviation alert can't be fully validated until 30 real days of production traffic accumulate — not actionable now, just needs revisiting then (by design, not a gap)

### Phase 7 remaining — Retraining Loop

- [ ] **TASK-7.2** — Exercise the CD-trigger call (step 6) for real once a training run actually clears the promotion gate
- [ ] Decide whether/when to flip `retrain-weekly`'s schedule from inactive to active
- [ ] (Deferred by design, not urgent) Build the drift-based trigger: Grafana `webhook` contact point + a small shim translating its POST into Prefect's `create_flow_run` API, since self-hosted Prefect has no native inbound webhook receiver

### Loose ends

- [ ] DVC-track `models/starcop_baseline/*.ckpt` checkpoints (flagged in TASK-1.1, never picked up)
- [ ] (Optional) Move GHCR image from a mutable `:latest` tag to per-build tags for real rollback/traceability
- [ ] (Optional, cosmetic) De-duplicate the badge-commit retry logic shared between `tests.yml`'s two jobs

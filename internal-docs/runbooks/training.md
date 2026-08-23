# Training Runbook (TASK-2.2)

How to actually run `src/training/train.py` — the MLflow-tracked STARCOP
training entrypoint built in TASK-2.2. Full design rationale lives in
`internal-docs/plan.md` (Phase 2); this doc is the practical
"how do I run it" companion, written from what running it for real actually
required.

## Prerequisites

- Environment A set up (`vendor/starcop/.venv`, Python 3.10) per TASK-0.2/0.3.
- MLflow client installed: `uv pip install --python vendor/starcop/.venv/bin/python -r vendor/starcop/requirements.txt -r requirements/env-a-mlflow.txt`
- The dataset already processed by the DVC pipeline (TASK-1.2) —
  `data/processed/<dataset_name>/{patches,splits}/` must exist.

**A stale `vendor/starcop/.venv` can mask a real gap** — found via CI, not
locally: `pytorch_lightning==1.6.4`/`wandb==0.13.3` (both pinned in
`vendor/starcop/requirements.txt`, both 2022-era) only import cleanly under
`setuptools<81` (84+ removed `pkg_resources`, which 1.6.4 still imports
unconditionally) and `numpy<2` (wandb uses the removed `np.float_` alias).
`requirements/env-a-mlflow.txt` pins both, but only a **fresh** venv install
actually re-resolves them — an existing `.venv` set up before these pins
were added will silently already have a working (older) setuptools/numpy and
never notice. If Environment A works locally but fails on a clean checkout
or in CI, re-run the install command above against a fresh
`vendor/starcop/.venv` first.

## Credentials — three separate sets

Never commit these. Keep them in a git-ignored file (e.g. `.env.mlflow`,
already in `.gitignore`) and load them with `set -a` (see gotcha below).

```bash
# 1. Tracking auth (metrics/params/tags)
MLFLOW_TRACKING_URI=https://methane-detection-mlflow.ghostface.tech
MLFLOW_TRACKING_USERNAME=<mlflow basic-auth username>
MLFLOW_TRACKING_PASSWORD=<mlflow basic-auth password>

# 2. Artifact upload (checkpoint / images / confusion matrix -- goes straight
#    to Backblaze B2 via boto3, client-side, NOT proxied through the
#    tracking server, and NOT covered by the vars above)
MLFLOW_S3_ENDPOINT_URL=https://s3.<region>.backblazeb2.com
AWS_ACCESS_KEY_ID=<B2 Application Key ID>
AWS_SECRET_ACCESS_KEY=<B2 Application Key>
```

Use a **separate, least-privilege B2 Application Key** for this — don't reuse
the MLflow server's key (`deploy/mlflow/.env.example`'s `B2_APPLICATION_KEY*`).
The server's key needs broad access to manage the whole artifact bucket; a
training client only ever needs to write new run artifacts, so it should hold
a key scoped to this bucket with `listFiles`/`readFiles`/`writeFiles` only, no
`deleteFiles` or account-level capabilities. Export the resulting Key ID/Key
under boto3's expected `AWS_*` names (boto3 doesn't recognize B2's own
`B2_APPLICATION_KEY*` naming). See
`internal-docs/setup/environment-notes.md` for the full
provisioning/rotation/revocation procedure.

**Gotcha**: if the credentials file uses plain `VAR=value` lines (no
`export`), a bare `source .env.mlflow` sets shell-local variables that never
reach the Python subprocess. Always:

```bash
set -a
source .env.mlflow
set +a
```

Missing set (1) fails immediately at the first MLflow API call. Missing set
(2) fails later, at the first artifact upload (`botocore.exceptions.
NoCredentialsError`) — easy to miss since metrics/params log fine before that.

## Running it

**Prefer the per-machine launch script** (TASK-3.3a/3.3b): `./scripts/train_mac.sh
[dataset_name]` on the M4 Pro, `./scripts/train_desktop.sh [dataset_name]`
on the RTX 5070 desktop (both default `dataset_name` to `starcop_mini`).
Each wraps everything in this section — credential sourcing with
`set -a`/`set +a`, the hardcoded `MLFLOW_TRACKING_URI`, the accelerator
flags — into one command, plus a pre-flight check that fails loudly if a
required credential is missing rather than failing deep into the run. Extra
Hydra overrides can be passed after the dataset name, e.g.
`./scripts/train_desktop.sh starcop_mini training.max_epochs=5`. The
arg-building logic lives in `src/training/launch_profiles.py` (unit-tested;
see `test_launch_profiles.py`), so both `.sh` files stay thin.

**Important divergence**: `train_mac.sh` runs under Environment A
(`vendor/starcop/.venv/bin/python`), but `train_desktop.sh` runs under
**Environment B** (`.venv/bin/python`) — Environment A's stock
`torch==1.13.1` silently corrupts compute on the RTX 5070's Blackwell
architecture rather than erroring (see TASK-3.1 in
`internal-docs/setup/environment-notes.md` for the full spike). Colab
(TASK-3.3c) is still pending. The manual steps below remain the reference
for understanding exactly what each script does under the hood.

```bash
cd /path/to/methane-detection
set -a; source .env.mlflow; set +a
vendor/starcop/.venv/bin/python src/training/train.py \
  +machine=desktop \
  +dataset_name=starcop_mini \
  experiment_name=my-run-name
```

`machine` (`desktop`/`macbook`/`colab`) and `dataset_name`
(`starcop_mini`/`starcop_raw`) are required — the run fails fast with a
clear `ValueError` if either is missing (`configs/training/overlay.yaml`).
Everything else falls back to STARCOP's own defaults
(`vendor/starcop/scripts/configs/config.yaml`, never edited — see
[[feedback-vendor-starcop-composition-only]]), overridable the normal Hydra way, e.g.:

```bash
  training.max_epochs=5 \
  training.accelerator=cpu \
  dataloader.batch_size=4 \
  dataloader.num_workers=0
```

On Apple Silicon (M4 Pro tested), `training.accelerator=mps
training.devices=1` gives a real ~2x speedup over `accelerator=cpu` (see
TASK-3.2 in `internal-docs/setup/environment-notes.md` for full findings and benchmark
numbers). It needs two extra things beyond the base install, both
documented there in full:

```bash
uv pip install --python vendor/starcop/.venv/bin/python \
  "pytorch-lightning>=1.7.0,<2.0" "torch==1.13.1"   # once, per venv
export PYTORCH_ENABLE_MPS_FALLBACK=1                 # every run
```

Without the `pytorch-lightning` upgrade, `accelerator=mps` silently
**falls back to CPU with no error** (the pinned `pytorch-lightning==1.6.4`
predates Lightning's own MPS support) — a run would complete `FINISHED`
with real metrics while never touching the GPU. `train.py` guards against
this: it asserts the resolved device actually is `mps` and tags every run
with `resolved_device`, so check that tag in the MLflow UI rather than
trusting a `FINISHED` status alone.

On the RTX 5070 desktop (Arch Linux, native CUDA — TASK-3.1),
`training.accelerator=gpu training.devices=1` under **Environment B**
gives a real ~2.3x speedup over the CPU baseline and ~1.16x over MPS (see
TASK-3.1 in `internal-docs/setup/environment-notes.md` for the full spike,
the four composition-only fixes needed, and benchmark numbers). Nothing
extra needs installing per-run — `scikit-image` in root `pyproject.toml`
and the `lightning2_compat`/`optimizer_compat` composition fixes in
`train.py` are already in place. The CUDA toolkit itself (`sudo pacman -S
cuda`, once per machine) is a one-time system-level install, separate from
either Python environment.

`WANDB_MODE=disabled` is useful for a tracking-only smoke test without
setting up real W&B credentials — training and MLflow logging both work
fully with it set; only W&B-side logging no-ops.

## Where results land

- **MLflow UI**: `https://methane-detection-mlflow.ghostface.tech` — params, per-epoch
  metrics (including `val_f1score_background`), tags (`dataset_version`,
  `dataset_dirty`, `machine`, `sensor`), and artifacts (`checkpoint/`,
  `confusion_matrix.png`, `images/`).
- **Local**: `experiments/<experiment_name>/<timestamp>/` (Hydra's own run
  dir, gitignored — checkpoints, `.hydra/` config snapshot, `train.log`).
  Safe to delete after a run; nothing there is the source of truth once
  logged to MLflow.

## Known limitation

The two post-training diagnostic calls to STARCOP's own unmodified
`starcop.validation.run_validation` assume the test split has both `"easy"`
and `"hard"` no-plume examples. Small/skewed splits (like `starcop_mini`'s
9-scene test set) can lack that variety and raise `KeyError` inside STARCOP's
code, which can't be patched (composition-only rule — see
[[feedback-vendor-starcop-composition-only]]). `train.py` catches and logs a warning rather than failing the run; the
MLflow tracking this task validates has already fully succeeded by that
point regardless. Expect a `run_validation (...) failed` warning in the logs
on small datasets — not a bug in this project's own code.

## Verified

2026-08-10: 5-epoch run against `starcop_mini`, run
`ef9b1c7172e1447e8db0ff765032faf9`, status `FINISHED`. 88 params, all
expected metrics/tags, checkpoint (80MB) + 10 prediction images +
confusion-matrix PNG (visually confirmed non-empty) all logged correctly.

2026-08-12: same config on Apple M4 Pro with `accelerator=mps devices=1`,
run `71e388fabefd40e892483f552a97efbb`, status `FINISHED`,
`resolved_device` tag = `mps:0` (confirmed real GPU use, not a silent CPU
fallback). ~2x faster than the CPU run above (258.4s vs 519.8s total). Full
findings — three real bugs hit and fixed getting here — in
`internal-docs/setup/environment-notes.md` (TASK-3.2 section).

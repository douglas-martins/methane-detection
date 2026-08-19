# Environment Notes

## Environment A — MLflow client (TASK-2.2)

`src/training/train.py` (Environment A, `vendor/starcop/.venv`) logs
every training run to the MLflow server deployed in TASK-2.1
(`https://methane-detection-mlflow.ghostface.tech`).

### Install

The MLflow client is layered on top of the submodule's own pinned
dependencies, same pattern as `requirements/env-a-dev.txt`:

```bash
uv pip install --python vendor/starcop/.venv/bin/python \
  -r vendor/starcop/requirements.txt \
  -r requirements/env-a-mlflow.txt
```

`requirements/env-a-mlflow.txt` pins `protobuf<4` — `mlflow-skinny` accepts
`protobuf>=3.12,<8`, but `wandb==0.13.3` (pinned in
`vendor/starcop/requirements.txt`, kept alongside MLflow per TASK-0.2) ships
protobuf-generated files that only load under `protobuf<4`. An unpinned
install resolves to the newest protobuf satisfying mlflow alone (6.x) and
breaks wandb at import time (`TypeError: Descriptors cannot be created
directly`). Installing `requirements/env-a-mlflow.txt` keeps both loggers
importable together.

### Required environment variables

Not committed anywhere — export these locally (shell profile, or the
per-machine launch scripts from TASK-3.3 once they exist) before running
`train.py`:

| Variable | Purpose |
|---|---|
| `MLFLOW_TRACKING_URI` | `https://methane-detection-mlflow.ghostface.tech` |
| `MLFLOW_TRACKING_USERNAME` | MLflow basic-auth username (the server requires auth — see TASK-2.1) |
| `MLFLOW_TRACKING_PASSWORD` | MLflow basic-auth password |

These are the MLflow client's own standard environment variables — read
automatically by `mlflow.start_run()`/`MLFlowLogger`. Note that
`mlflow.start_run()` does *not* fail on its own when `MLFLOW_TRACKING_URI` is
unset — it silently falls back to a local file/sqlite tracking store, so a
missing var would otherwise train without ever reaching the server. To avoid
that, `train.py` calls `mlflow_utils.require_mlflow_tracking_env()` up front,
before any MLflow call, which raises `RuntimeError` if any of the three is
unset.

**A fourth, separate set is needed for artifact uploads** (checkpoint, confusion
matrix PNG, prediction images) — found while running TASK-2.2's live validation:
MLflow's artifact store is Backblaze B2, S3-compatible (TASK-2.1 decision D-02).
The tracking-auth vars above only cover metrics/params/tags API calls; uploading
an artifact goes over boto3's S3 client *directly to B2*, client-side, and boto3
doesn't know about `MLFLOW_TRACKING_*` at all — without these, artifact logging
fails with `botocore.exceptions.NoCredentialsError` mid-run (metrics/params still
log fine up to that point, which is what makes this easy to miss until an artifact
is actually logged):

| Variable | Purpose |
|---|---|
| `MLFLOW_S3_ENDPOINT_URL` | `https://s3.<region>.backblazeb2.com` — same value as `deploy/mlflow/.env.example`'s `MLFLOW_S3_ENDPOINT_URL` on the server |
| `AWS_ACCESS_KEY_ID` | A dedicated client-side B2 Application Key ID — **not** the server's `B2_APPLICATION_KEY_ID` (see below); boto3 (used client-side) only recognizes the standard `AWS_*` names, not B2's |
| `AWS_SECRET_ACCESS_KEY` | That same dedicated key's secret — **not** the server's `B2_APPLICATION_KEY` |

**Provision a separate, least-privilege B2 Application Key for client-side
uploads — do not reuse the MLflow server's key** (`deploy/mlflow/.env.example`'s
`B2_APPLICATION_KEY*`, which needs broader access to manage the whole artifact
bucket server-side).

- **Scope**: B2 dashboard → Application Keys → Add a New Application Key →
  restrict to the `methane-detection` bucket → capabilities `listFiles` +
  `readFiles` + `writeFiles` only (no `deleteFiles`, no account-level
  capabilities) — a training client only ever needs to write new run
  artifacts, never delete or manage existing ones.
- **Distribution**: export the resulting Key ID/Key under boto3's expected
  `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` names in each training
  machine's `.env.mlflow` (git-ignored, never committed).
- **Rotation**: this key is copied out to every training machine (laptops,
  and eventually unattended Prefect workers) — rotate it on a regular
  cadence, and immediately if any machine holding it is decommissioned,
  lost, or suspected compromised.
- **Revocation**: B2 dashboard → Application Keys → delete the key — takes
  effect immediately. Update `.env.mlflow` on every machine still using it
  first, or client uploads start failing with a 403
  `botocore.exceptions.ClientError`.

## Environment A — Apple MPS training (TASK-3.2)

Real 5-epoch `starcop_mini` training on Apple Silicon (M4 Pro), verified end
to end. Three real, unrelated blockers were found and fixed getting there —
each is a genuine gap in a pinned dependency, not something specific to this
project's own code:

### 1. pytorch-lightning==1.6.4 silently falls back to CPU on `accelerator=mps`

Lightning's `MPSAccelerator` was only added in 1.7.0; 1.6.4's accelerator
registry has no `mps.py` at all. `Trainer(accelerator="mps")` under 1.6.4
does **not** raise — it silently resolves to `CPUAccelerator` (`GPU
available: False, used: False` printed, no exception), so a run would
complete `FINISHED` with real metrics while never touching the GPU.

Fix: upgrade to `pytorch-lightning==1.9.5` (highest 1.x compatible with the
pinned `torch==1.13.1` — an unconstrained resolve wants to bump torch too).
Can't be pinned inside `requirements/env-a-mlflow.txt` alongside vendor's
own exact `==1.6.4` pin (uv/pip combine constraints across `-r` files rather
than letting a later file win), so run as a **separate command** after the
normal install, only needed for `accelerator=mps` runs:

```bash
uv pip install --python vendor/starcop/.venv/bin/python \
  "pytorch-lightning>=1.7.0,<2.0" "torch==1.13.1"
```

`src/training/train.py` also guards against this class of bug going
forward: after constructing the `Trainer`, it asserts the resolved
`trainer.strategy.root_device` is actually `mps` when `accelerator=mps` was
requested (`src/training/accelerator_check.py`), raising instead of
trusting the request matches what Lightning resolved, and tags every run
with `resolved_device` so it's visible in the MLflow UI without digging
through logs.

### 2. STARCOP's `DataNormalizer` builds int64 Parameters that crash MPS's `clamp`

`vendor/starcop/starcop/data/normalizer_module.py` builds
`offsets_input`/`factors_input`/`clip_min_input`/`clip_max_input` via
`torch.from_numpy(np.array(python_ints_or_floats))`. When every value for
the active `input_products` happens to be a plain int (e.g. `clip: (0, 2)`
for the AVIRIS bands + mag1c used by `starcop_mini`), numpy infers `int64`.
`torch.clamp` on CPU implicitly promotes an int64 bound against a float32
input; **MPS's clamp kernel cannot broadcast a dtype-mismatched pair and
aborts the whole process** (`LLVM ERROR: Failed to infer result type(s)`,
not a catchable Python exception — the process dies with SIGABRT).

Fix (composition, no vendor edit): `src/training/normalizer_dtype_fix.py`
casts those Parameters to float32 in place after model construction. This
is numerically a no-op (same values, wider dtype — `int64` bounds already
implicitly promote to float in the clamp computation itself) and safe on
every backend, so it runs unconditionally rather than gated on accelerator.

### 3. `torch.unique` has no MPS kernel in torch 1.13.1

torchmetrics 0.10.0's `BinaryConfusionMatrix.update()` calls
`torch.unique(target)` for validation — `NotImplementedError: The operator
'aten::_unique2' is not currently implemented for the MPS device`. PyTorch's
own error message names the fix: set
`PYTORCH_ENABLE_MPS_FALLBACK=1`, which makes unimplemented MPS ops silently
fall back to CPU per-op (slower for that op only, everything else still
runs on MPS). Required for every `accelerator=mps` run:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

### 4. (Unrelated to MPS) unpinned `mlflow` breaks `mlflow.pytorch.log_model` under torch 1.13.1

Found while validating the MPS run above, but affects **every** accelerator
equally — it's an environment drift bug, not an MPS-specific one.
`requirements/env-a-mlflow.txt` pinned `mlflow` with no version bound at
all; a fresh resolve landed on `mlflow==3.15.1`, whose
`mlflow.pytorch.save_model()` does `from torch.export import Dim as
ExportDim` **unconditionally at the top of the function body** — not gated
behind the opt-in `export_model` flag, so it breaks every call to
`mlflow.pytorch.log_model()`, not just export-format ones. `torch.export`
doesn't exist before torch 2.1; Environment A pins `torch==1.13.1`. Any run
(CPU or MPS) that reached the model-logging step would fail with
`ModuleNotFoundError: No module named 'torch.export'`, marking the MLflow
run `FAILED` even though training itself completed successfully.

Fix: `requirements/env-a-mlflow.txt` now pins `mlflow<3.7` (verified 3.7.0
and below have no `torch.export` reference in `mlflow/pytorch/__init__.py`;
3.10.0+ does — resolved to `mlflow==3.6.0`).

### Verified (2026-08-12)

5-epoch `starcop_mini` run, `accelerator=mps devices=1`, run
`71e388fabefd40e892483f552a97efbb`: status `FINISHED`, `resolved_device`
tag = `mps:0` (not silently CPU — see blocker 1), checkpoint +
confusion-matrix PNG + prediction images all logged, real metrics
(`val_loss=0.0494`, `val_f1score_background=0.998`,
`val_iou=0.118`). Same known limitation as TASK-2.2's CPU run: the
post-training `run_validation` diagnostic calls warn-and-skip on
`starcop_mini`'s small/skewed test split (documented there already, not a
new issue).

**Benchmark vs TASK-2.2's CPU baseline** (same `starcop_mini` config, same
5 epochs, same M4 Pro hardware, run `ef9b1c7172e1447e8db0ff765032faf9`):

| Metric | CPU (`accelerator=cpu`) | MPS (`accelerator=mps devices=1`) |
| --- | --- | --- |
| Total wall-clock (5 epochs) | 519.8s (8.66 min) | 258.4s (4.31 min) |
| Per-epoch | ~104.0s | ~51.7s |
| Speedup | — | **~2.0x** |

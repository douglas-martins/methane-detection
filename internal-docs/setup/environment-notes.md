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
- **Rotation** (routine, no known exposure): this key is copied out to every
  training machine (laptops, and eventually unattended Prefect workers) —
  rotate it on a regular cadence. Provision the replacement key and update
  `.env.mlflow` on every machine still using it *before* revoking the old
  key, so client uploads don't fail with a 403
  `botocore.exceptions.ClientError` in the gap.
- **Revocation after loss or compromise**: if a machine holding this key is
  decommissioned, lost, or suspected compromised, delete the key
  immediately (B2 dashboard → Application Keys → delete the key — takes
  effect immediately) — don't wait to update `.env.mlflow` on other
  machines first, since the exposure risk outweighs the resulting 403
  `botocore.exceptions.ClientError` on trusted machines. Provision a
  replacement key and restore `.env.mlflow` on the trusted machines
  afterward.

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

## Desktop — RTX 5070 CUDA training, and why it runs under Environment B (TASK-3.1)

**This machine's training runs under Environment B (`.venv`), not Environment
A** (`vendor/starcop/.venv`) — the one deliberate deviation from every other
machine's launch script. Reason, found by actually spiking it (2026-08-23),
not assumed:

### The blocker: Environment A's stock torch silently corrupts Blackwell compute

`vendor/starcop/requirements.txt` pins `torch==1.13.1` exactly (submodule-
owned, composition-only — not editable). Official 1.13.1 wheels were built
against CUDA 11.6/11.7 and contain no compiled kernels for Blackwell's
`sm_120` compute capability (PyTorch added Blackwell kernel support around
the 2.7 line). Real spike result, under Environment A on this machine:

- `torch.cuda.is_available()` → `True`
- `torch.cuda.get_device_capability(0)` → `(12, 0)` (sm_120)
- PyTorch warns at import (*"...is not compatible with the current PyTorch
  installation. The current PyTorch install supports CUDA capabilities
  sm_37 sm_50 sm_60 sm_70 sm_75 sm_80 sm_86"*) but **does not raise**
- A CPU tensor `[0.3005, 0.5537, 0.9759, 0.5056]` becomes `[0., 1., 1., 1.]`
  after `.to('cuda')`; `x + 1` on that GPU tensor then yields all zeros

This is **silent data corruption**, not the crash a Blackwell/pre-cu128
mismatch would normally produce and not the silent-CPU-fallback class of bug
TASK-3.2 found for MPS. The device tag stays `cuda:0` throughout, so neither
`accelerator_check.assert_resolved_accelerator` nor a `FINISHED` MLflow run
would catch it — a real training run under this exact torch build could
complete with plausible-looking-but-wrong metrics. **Do not run training
under Environment A's stock torch on this GPU, for any reason.**

Upgrading torch inside Environment A to fix this was considered and rejected:
it would cascade into `pytorch-lightning`, `torchmetrics`,
`segmentation-models-pytorch`, and `kornia` all needing bumps too, at which
point Environment A's venv would no longer resemble "the original 2022
STARCOP paper stack" (its entire reason for existing, TASK-0.2/TASK-0.3) —
on this one machine only, undermining the whole point of a separate
Environment A.

### The fix: Environment B already works, confirmed real

Environment B (`torch==2.12.1+cu130` on this machine) passed the identical
tensor-corruption check cleanly (values survive `.to('cuda')`/`+1`
unchanged), plus three escalating real tests: the full import chain
(`pytorch_lightning`, `torchmetrics`, `segmentation_models_pytorch`,
`kornia`, vendor's own `ModelModule`/`Permian2019DataModule`), a real
forward+backward pass through the actual model-construction path on
`cuda:0` (correct output, finite gradients), and
`Trainer(accelerator="gpu", devices=1)` resolving to `CUDAAccelerator`/
`cuda:0`.

### Four real bugs found and fixed getting an actual training run green

Same composition-only discipline as TASK-3.2's attempts above — each is a
genuine Environment-B-vs-vendor-code version mismatch:

1. **Missing `scikit-image`** — `data_module.prepare_data()` imports
   `skimage` transitively (via vendor's `sampling_dataset.py` →
   `mask_creation.py`), never declared in root `pyproject.toml`.
   `ModuleNotFoundError: No module named 'skimage'`. Fixed: added
   `scikit-image` to `pyproject.toml` `dependencies` (permanent, same class
   of gap as TASK-5.1's missing `boto3`/`wandb`/`rasterio`).
2. **Lightning 2.x rejects STARCOP's pre-2.0 hooks** — `ModelModule`
   implements `validation_epoch_end`/`test_epoch_end` (removed in Lightning
   2.0); its own configuration validator raises `NotImplementedError`
   merely because the method is *present*, regardless of whether it's
   called. Fixed via composition: `src/training/lightning2_compat.py`
   shadows the old names and binds `on_validation_epoch_end`/
   `on_test_epoch_end` instead — version-gated (no-op under Lightning
   <2.0), so Environment A is unaffected.
3. **`ReduceLROnPlateau` dropped `verbose`** — `ModelModule.
   configure_optimizers` still passes `verbose=True`; a later torch release
   removed the kwarg. `TypeError: ReduceLROnPlateau.__init__() got an
   unexpected keyword argument 'verbose'`. Fixed via composition:
   `src/training/optimizer_compat.py`, gated by introspecting the installed
   `ReduceLROnPlateau` signature (not a hardcoded torch version).
4. **mlflow's `log_model` default changed** — this mlflow version (3.14.0)
   defaults `serialization_format` to `'pt2'` (torch.export tracing,
   requires `input_example`) instead of `'pickle'`. `train.py` already had
   this exact fix in one other place (`src/registry/hf_baseline_import.py`)
   — just never carried into `train.py` itself since it had never run under
   torch≥2.1 before. One-line fix: `serialization_format="pickle"`.

### Verified (2026-08-23)

5-epoch `starcop_mini` run via `./scripts/train_desktop.sh starcop_mini
training.max_epochs=5 dataloader.batch_size=4 dataloader.num_workers=0`, run
`f16aa1f8330248df855ceea77eb1a281`: status `FINISHED`, `resolved_device` tag
= `cuda:0` (confirmed via `MlflowClient.get_run` directly against the live
server, not inferred from console output), real metrics (`val_loss=0.0729`,
`val_accuracy=0.9102`, `val_f1score=0.0136` — low methane-class F1 expected
at this tiny 5-epoch scale). Same known limitation as TASK-2.2/TASK-3.2's
runs: `run_validation`'s post-training diagnostic warn-and-skips — this time
via a different root cause (`torchmetrics.ConfusionMatrix(num_classes=2)`
missing the now-required `task=` kwarg under `torchmetrics==1.9.0`, not
TASK-2.2's documented skewed-split `KeyError`), same non-fatal handling.

**Benchmark vs TASK-2.2's CPU baseline and TASK-3.2's MPS baseline** (same
`starcop_mini` config, same 5 epochs):

| Metric | CPU (M4 Pro) | MPS (M4 Pro) | CUDA (RTX 5070, Environment B) |
| --- | --- | --- | --- |
| Total wall-clock (5 epochs) | 519.8s | 258.4s | ~222.7s |
| Per-epoch | ~104.0s | ~51.7s | ~44.5s |
| Speedup vs CPU | — | ~2.0x | **~2.3x** |
| Speedup vs MPS | — | — | **~1.16x** |

The RTX 5070's speedup over MPS is modest for a discrete desktop GPU —
likely dominated by fixed per-step overhead at this tiny scale
(`batch_size=4`, `num_workers=0`, deliberately matching TASK-3.2's config
for a fair comparison rather than this GPU's actual throughput ceiling). GPU
utilization wasn't captured via concurrent `nvidia-smi` this run — the
`resolved_device` tag plus the real-op correctness evidence above is
stronger proof of genuine GPU use than a utilization percentage alone
(same reasoning TASK-3.2 used for skipping a visual Activity-Monitor check).

### CUDA toolkit install (Arch Linux, native)

```bash
sudo pacman -S cuda   # 13.3.1-1 at the time of writing, 2.20 GiB download
```

`nvcc` lands at `/opt/cuda/bin/nvcc` — not on `PATH` by default on Arch.
Driver requirement (≥ 570 for Blackwell/CUDA 12.8+) is separate from the
toolkit and was already satisfied on this machine (driver `610.57.04`,
`nvidia-smi`).

## Colab — Environment A on a free-tier T4 (TASK-3.3c spike, 2026-08-24)

Manual spike run interactively in a Colab notebook (not scripted yet — this
records what `notebooks/train_colab.ipynb`'s install cells need to do).
**Outcome: Environment A's exact pinned stack runs correctly on Colab's free
T4 runtime, no Environment B pivot needed** (unlike TASK-3.1's Desktop
case) — three real, fixable blockers along the way, none of them a hardware
compute-correctness problem like the Desktop's Blackwell issue.

### 0. The runtime type you pick changes the Python version, not just the GPU

A fresh Colab notebook defaults to a CPU-only runtime on **Python 3.13.15**
— `torch==1.13.1` has no wheel for that (earliest available: `2.5.0`).
Switching **Runtime → Change runtime type → Hardware accelerator → GPU (T4)**
doesn't just attach a GPU — it swaps the underlying image to **Python
3.11.13**, which *does* have a `torch==1.13.1` `cp311` wheel. Do the runtime-
type switch before anything else; a `pip install torch==1.13.1` failure on a
fresh notebook is most likely this, not a real incompatibility.

Free tier confirmed via the "Change runtime type" dialog: only **T4 GPU**
and TPU options are selectable; A100/L4/H100/G4 are visible but greyed out
behind a "Purchase additional compute units" prompt. `!nvidia-smi` (once the
GPU runtime is selected) confirms: Tesla T4, driver `580.82.07`, CUDA
`13.0` (driver ceiling, not what's installed — see below), 15360 MiB VRAM.

### 1. `pytorch-lightning==1.6.4`'s wheel metadata is malformed — pip ≥24.1 refuses it

```text
Requested pytorch-lightning==1.6.4 ... has invalid metadata: .* suffix can
only be used with `==` or `!=` operators
    torch (>=1.8.*)
Please use pip<24.1 if you need to use this version.
```

A real upstream packaging bug in that 2022 release (`>=1.8.*` is invalid —
`.*` only pairs with `==`/`!=`), not a project issue. Colab ships a pip
newer than 24.1 by default. Fix — downgrade pip first, and verify it
actually took (a plain `!pip install "pip<24.1"` can silently not apply if
run out of order or in the wrong cell):

```python
!python -m pip install -q "pip<24.1"
!pip --version   # confirm < 24.1 before installing pytorch-lightning
```

### 2. Colab's pre-installed NumPy 2.x breaks torch 1.13.1's numpy interop

`torch==1.13.1` (2022-era) was compiled against NumPy 1.x's ABI; Colab ships
NumPy `2.0.2` by default. Installing torch alone doesn't downgrade numpy
(torch's own metadata doesn't pin it), so the first numpy-touching torch
call throws:

```text
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.0.2 ...
UserWarning: Failed to initialize NumPy: _ARRAY_API not found
```

Same root cause as the existing `numpy<2` pin in
`requirements/env-a-mlflow.txt` (found there via `wandb==0.13.3`'s
`np.float_` usage) — install `numpy<2` explicitly rather than trusting an
unpinned resolve, and do it **before** any cell that imports `torch` and
touches a real tensor, since a mid-session numpy downgrade after torch has
already initialized its (broken) numpy interop may need a runtime restart
(**Runtime → Restart session**) to take effect cleanly — numpy's C
extension state doesn't hot-swap.

### Confirmed working install order

```python
# 1. pip's own fix for pytorch-lightning==1.6.4's malformed metadata
!python -m pip install -q "pip<24.1"

# 2. must land before torch touches numpy for real
!pip install -q "numpy<2"

# 3. torch itself (uninstalls Colab's pre-installed torch==2.6.0+cu124 --
#    orphans torchvision/torchaudio/torchdata/accelerate, all pinned to
#    2.6.0 and unrelated to this project; harmless noise in this session)
!pip install -q torch==1.13.1

# 4. rest of Environment A's pinned stack
!pip install -q \
  "pytorch-lightning==1.6.4" \
  "torchmetrics==0.10.0" \
  "kornia==0.6.7" \
  "wandb==0.13.3" \
  segmentation_models_pytorch \
  "setuptools<81" \
  "protobuf<4"
```

`segmentation_models_pytorch` (unpinned in `vendor/starcop/requirements.txt`)
installed cleanly against the orphaned-torchvision state above — no conflict
surfaced.

### Real compute confirmed (2026-08-24)

Not just "install succeeded" — same bar TASK-3.1/TASK-3.2 used. A toy
`pl.LightningModule` with `Trainer(accelerator="gpu", devices=1)` ran one
real `.fit()` step, with in-code assertions on `batch.device.type` and
`next(model.parameters()).device.type`, plus an explicit check of
`trainer.strategy.root_device`:

```text
INFO:pytorch_lightning.utilities.rank_zero:GPU available: True, used: True
INFO:pytorch_lightning.accelerators.gpu:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
✅ Real compute confirmed on cuda:0
```

Unlike TASK-3.2's MPS case, `pytorch-lightning==1.6.4` resolved
`accelerator="gpu"` to `CUDAAccelerator`/`cuda:0` correctly out of the box —
no Lightning version upgrade needed (GPU/CUDA support long predates 1.6.4,
unlike `MPSAccelerator` which was added in 1.7.0). And unlike TASK-3.1's
Desktop/Blackwell case, no silent tensor corruption — T4 (Turing, compute
capability 7.5) is well inside the range torch 1.13.1's bundled CUDA 11.7
runtime supports, so this isn't expected to need an Environment B pivot the
way the RTX 5070 did.

**Not yet done**: this was a toy model/dataset, the same scope TASK-3.2 used
for its own Lightning-resolution check before the real `starcop_mini` run.
Still open for TASK-3.3c: a real `starcop_mini` run through `train.py`
itself, and confirming `mlflow`/`boto3` install cleanly alongside this
stack (not attempted in this spike — it only covered the compute path). The
DVC service-account pull itself **is** now confirmed — see below.

### DVC pull via the D-01 service account — confirmed working (2026-08-24)

Reuses the same Google service account JSON key already created for the
`mac-mps` Prefect worker (D-01) — no new key needed. Three more real,
fixable blockers found getting to a clean non-interactive pull, none of
them specific to this project's own code:

**1. Wrong pull target — `data/processed/<dataset>` isn't a real DVC output.**
`dvc.yaml`'s stages each declare their own `outs:` subdirectory
(`data/processed/<dataset>/selected`, `/splits`, `/patches`, `/stats`,
`/coordinates`) — the bare parent path isn't one of them, so DVC rejects it
as neither an output nor a stage name (`NoOutputOrStageError`). For a
credential/connectivity smoke test, pull the raw dataset instead — a plain
`.dvc`-tracked directory, unambiguous:

```python
!dvc pull data/starcop_mini -v
```

To pull what training actually reads, address the real `dvc.yaml`
foreach-stage instances by name instead:

```python
!dvc pull "normalize@starcop_mini" "split@starcop_mini" \
  "patch_extract@starcop_mini" "stats@starcop_mini" "coordinates@starcop_mini" -v
```

**2. `pip install dvc[gdrive]` as its own command re-resolves `protobuf` upward,
undoing the earlier `numpy`/`protobuf` pins.** Same class of issue as the
`pip<24.1` fix above — pip doesn't remember a constraint from a *separate*
prior command. An unconstrained `pip install "dvc[gdrive]==3.67.1"` pulled
`protobuf` up to `6.x` (driven by Colab's own preinstalled
`tensorflow`/`google-ai-generativelanguage`/etc., all irrelevant to this
project), which would have broken `wandb`/`pytorch-lightning`'s import-time
protobuf usage again. Fix: pin `protobuf<4` in the *same* install command:

```python
!pip install -q "dvc[gdrive]==3.67.1" "protobuf<4"
```

(Resolves to `protobuf==3.20.3` in practice — technically one patch above
`pytorch-lightning`'s stated `<=3.20.1` ceiling, but same minor version,
`3.20.x`, so the generated `_pb2.py` ABI stays compatible; no import
failure observed.)

**3. `pydrive2`'s legacy `oauth2client` auth path is incompatible with any
single modern `pyOpenSSL`/`cryptography` pairing.** `dvc-gdrive` → `pydrive2`
→ `googleapiclient`'s legacy `_auth.py` compatibility shim → the deprecated
`oauth2client` library (last meaningfully updated ~2020) for its
`ServiceAccountCredentials` JWT-signing path. This collided twice, in
opposite directions:

- With whatever `pyOpenSSL` pip resolved by default (too *old* for Colab's
  preinstalled `cryptography`): `AttributeError: module 'lib' has no
  attribute 'GEN_EMAIL'` — `cryptography` restructured its low-level OpenSSL
  bindings in a version newer than that `pyOpenSSL` expects.
- After upgrading to the latest `pyOpenSSL` (`pip install -U pyOpenSSL`) to
  fix that: `AttributeError: module 'OpenSSL.crypto' has no attribute
  'sign'` — modern `pyOpenSSL` (≥23) removed the legacy `crypto.sign()`/
  `crypto.verify()` functions `oauth2client`'s signer calls directly.

No single `pyOpenSSL` version satisfies both `cryptography` (needs new) and
`oauth2client`'s `crypto.sign()` call (needs old) at once. Rather than pin
an exact three-way version triangle, the robust fix is to **remove
`pyOpenSSL` entirely** — `oauth2client.crypt` tries `pyOpenSSL` first but
falls back to a pure-Python RSA signer (`rsa` + `pyasn1-modules`, no C
extension, no ABI to break) when `pyOpenSSL` isn't importable at all:

```python
!pip uninstall -y pyOpenSSL
!pip install -q rsa pyasn1-modules
```

### Confirmed working DVC install + pull recipe (2026-08-24)

```python
!pip install -q "dvc[gdrive]==3.67.1" "protobuf<4"
!pip uninstall -y pyOpenSSL
!pip install -q rsa pyasn1-modules
```

```python
from google.colab import userdata
from pathlib import Path
import os

sa_json = userdata.get("DVC_GDRIVE_SERVICE_ACCOUNT_JSON")
key_path = Path("/content/gdrive-service-account.json")
key_path.write_text(sa_json)
os.chmod(key_path, 0o600)
```

```python
!dvc remote modify --local gdrive gdrive_use_service_account true
!dvc remote modify --local gdrive gdrive_service_account_json_file_path /content/gdrive-service-account.json
```

```python
!dvc pull data/starcop_mini -v
```

**Verified (2026-08-24)**: 418 files fetched, 417 added, `A
data/starcop_mini/` — clean, silent, no browser/OAuth prompt at any point.
Confirms the exact mechanism `deploy/prefect/README.md` documents for the
`mac-mps` worker also works unattended from a fresh Colab VM, closing the
one open point TASK-3.3c's pre-implementation review flagged.

# STARCOP baseline integration

This directory owns this project's adapters around the unmodified STARCOP source
in [`../../../vendor/starcop/`](../../../vendor/starcop/). It is the baseline's
single home; reusable infrastructure remains in the top-level `src/` directories.

## Components

| Directory | Responsibility | Environment |
| --- | --- | --- |
| `training/` | Hydra composition, processed-data wiring, metrics, launch profiles, and Lightning/torch compatibility | Baseline and research |
| `registry/` | Import pinned STARCOP checkpoints into the shared MLflow registry | Research for real imports; tests also cover supported baseline behavior |
| `evaluation/` | Full paper-test evaluation, paper metrics, MLflow comparison artifacts, and local live verification | Baseline and research |
| `serving/` | STARCOP model loading, input assembly/inference, band baselines, and the BentoML service | Research only |

Each component imports upstream objects through its own `_vendor_starcop*.py`
seam. Baseline adapters may import shared helpers from `src/training/`,
`src/registry/`, and `src/serving/`; those shared directories must not import this
baseline.

## Commands

Run from the repository root. The stable script wrappers are the supported user
entrypoints:

```bash
# Training: baseline env on Apple Silicon; research env on Blackwell desktop
./scripts/train_mac.sh starcop_mini
./scripts/train_desktop.sh starcop_mini

# Direct training equivalents
vendor/starcop/.venv/bin/python src/baselines/starcop/training/train.py \
  +machine=macbook +dataset_name=starcop_mini
.venv/bin/python src/baselines/starcop/training/train.py \
  +machine=desktop +dataset_name=starcop_mini

# Checkpoint import and a bounded paper-evaluation smoke (research env)
.venv/bin/python scripts/import_starcop_hf_baseline.py mag1c_rgb --stage Staging
.venv/bin/python scripts/run_starcop_baseline_evaluation.py mag1c_rgb --limit 5

# Local service (research env, using credentials from the ignored env file)
set -a; source .env.mlflow; set +a
.venv/bin/bentoml serve \
  src.baselines.starcop.serving.service:MethaneDetectionService
```

Training and remote MLflow operations require the credentials described in
[`../../../internal-docs/runbooks/training.md`](../../../internal-docs/runbooks/training.md).
The service exposes `POST /predict`, `POST /health`, and BentoML's `GET /metrics`.
The stable live-check wrapper is `scripts/run_live_verify.py`; it targets an
explicit local service and does not contact production by default.

## Shared boundaries

- `src/training/`: generic MLflow/DVC training helpers.
- `src/registry/`: generic registry lookup and promotion policy.
- `src/serving/`: only shared drift calculations and `BandStats`, not an inference
  API.
- `src/comparison/`: future candidate-versus-baseline reports. Paper-specific
  evaluation remains here under `evaluation/`.
- `src/models/<candidate>/`: independent model implementations; they do not import
  baseline training orchestration to obtain generic helpers.

Do not add old-path aliases or duplicate baseline logic. A serialization or
external-command compatibility surface requires a demonstrated consumer, a test,
an owner, and a removal condition.

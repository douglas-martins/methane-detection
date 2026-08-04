# Methane Detection — MLOps Pipeline

![tests](docs/badges/tests.svg)
![coverage](docs/badges/coverage.svg)

Master's Final Project — Semantic Segmentation of Methane Plumes using CNN on Hyperspectral Imagery.

## Overview

An end-to-end MLOps pipeline for detecting methane plumes from hyperspectral satellite imagery (AVIRIS-NG, EMIT). Built on top of the [STARCOP](https://github.com/spaceml-org/STARCOP) baseline, with production-grade ML engineering: data versioning (DVC), experiment tracking (MLflow), CI/CD (GitHub Actions), model serving (BentoML/FastAPI), and monitoring (Prometheus + Grafana).

## Repository Structure

```
├── vendor/starcop/     # STARCOP reference implementation (git submodule)
├── src/
│   ├── data/           # Dataset loaders, preprocessing
│   ├── models/         # Architecture definitions
│   ├── training/       # Training scripts (updated Lightning 2.x stack)
│   └── serving/        # Inference API
├── tests/              # Unit and integration tests
├── configs/            # Hydra configuration files
├── notebooks/          # Exploratory analysis
├── docker/             # Dockerfiles for serving and CI
├── docs/               # Project documentation and reports
├── data/               # DVC-tracked — not committed to Git
└── models/             # DVC-tracked — not committed to Git
```

## Getting Started

### Clone (includes STARCOP submodule)

```bash
git clone --recurse-submodules https://github.com/ghostface/methane-detection-mlops
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init
```

### Environment A — STARCOP original (Python 3.10, torch 1.13.1)

For running and inspecting the original STARCOP baseline:

```bash
cd vendor/starcop
uv venv --python 3.10
uv pip install -r requirements.txt
uv pip install -e .
```

### Environment B — MLOps project (Python 3.12, torch ≥ 2.5)

For active development on the pipeline:

```bash
uv venv --python 3.12
uv sync
```

## Testing

Unit tests live in `__tests__/` folders next to the module they cover (e.g. `src/data/__tests__/`), plus a mirrored `tests/vendor_starcop/` tree for STARCOP submodule scripts, which are never modified in place. Shared fixtures live in the project-root `conftest.py`.

```bash
make test-env-a   # run the Environment A suite (vendor/starcop/.venv)
make coverage      # run with coverage, writes junit.xml + coverage.xml
make badges        # regenerate docs/badges/*.svg from the latest run
```

New Environment A test dependencies (pytest, gdown, genbadge, …) are layered on top of `vendor/starcop/requirements.txt` via `requirements/env-a-dev.txt`, without editing the submodule's own file:

```bash
uv pip install --python vendor/starcop/.venv/bin/python -r vendor/starcop/requirements.txt -r requirements/env-a-dev.txt
```

## Dataset

Primary dataset: [STARCOP](https://zenodo.org/record/7863350) (AVIRIS-NG + EMIT hyperspectral scenes with pixel-wise methane annotations). Managed with DVC — see Phase 1 of the implementation plan.

Mini dataset for development: `huggingface-cli download previtus/starcop_allbands_mini`

## Implementation Plan

See [`mlops-methane-detection-plan.md`](mlops-methane-detection-plan.md) for the full phase-by-phase task list.

## Infrastructure

| Component | Tool | Location |
|---|---|---|
| Experiment tracking | MLflow | Hostinger KVM 4 VPS (Coolify) |
| Pipeline orchestration | Prefect | Hostinger KVM 4 VPS (Coolify) |
| Model serving | BentoML / FastAPI | Hostinger KVM 4 VPS (Coolify) |
| Monitoring | Prometheus + Grafana | Hostinger KVM 4 VPS (Coolify) |
| Data versioning | DVC | Google Drive remote |
| CI/CD | GitHub Actions | ghcr.io (Docker images) |

## License

MIT
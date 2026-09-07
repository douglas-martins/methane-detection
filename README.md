# Methane Detection — end-to-end deep learning pipeline

![tests (baseline)](docs/badges/tests-baseline.svg)
![coverage (baseline)](docs/badges/coverage-baseline.svg)
![tests (research)](docs/badges/tests-research.svg)
![coverage (research)](docs/badges/coverage-research.svg)
[![codecov](https://codecov.io/gh/douglas-martins/methane-detection/graph/badge.svg)](https://codecov.io/gh/douglas-martins/methane-detection)
[![Lint](https://github.com/douglas-martins/methane-detection/actions/workflows/lint.yml/badge.svg)](https://github.com/douglas-martins/methane-detection/actions/workflows/lint.yml)
[![Docs](https://github.com/douglas-martins/methane-detection/actions/workflows/docs.yml/badge.svg)](https://github.com/douglas-martins/methane-detection/actions/workflows/docs.yml)

> **[Master's Final Project]** — Semantic Segmentation of Methane Plumes using CNN on
> Hyperspectral Imagery. [TODO: institution / advisor / one-line paper citation.]

A methane plume detector, built on the STARCOP baseline and targeting
**on-board/embedded deployment** — the model is the deliverable, with the final
architecture still being finalized to meet on-board constraints. (FPGA-style
acceleration via [hls4ml](https://fastmachinelearning.org/hls4ml/), alongside
Xilinx's Vitis AI toolchain, is the working deployment direction — see the
documentation site for details.) This repo also carries the full MLOps pipeline
(DVC, MLflow, CI/CD, BentoML serving, Prometheus/Grafana monitoring, Prefect
retraining) that makes iterating toward it fast and reproducible: candidates
are proven out cheaply on a small dataset before a full-scale training run.

## 📖 Documentation

Full documentation — methodology, dataset, results, architecture, model registry policy —
lives at **[the MkDocs site](https://douglas-martins.github.io/methane-detection/)**.

## Quick Start

```bash
git clone --recurse-submodules https://github.com/douglas-martins/methane-detection
cd methane-detection

# Research env — active development (Python 3.12)
uv venv --python 3.12
uv sync
```

The original STARCOP baseline stack (Python 3.10, reference-only) and everything
beyond this — testing, architecture, the never-edit `vendor/starcop/` rule — are
covered in [CONTRIBUTING.md](CONTRIBUTING.md). The
[`src/` architecture map](src/README.md) identifies baseline adapters, shared
infrastructure, candidate-model homes, and comparison ownership.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE).

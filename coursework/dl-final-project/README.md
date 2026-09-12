# DL Course Final Project — Methane Plume Segmentation

Course: *Aprendizado Profundo*, PPGCA, Universidade do Vale do Itajaí.
Professor: Felipe Viel — Period: 2026/2.

This directory is deliberately isolated from the rest of the repository —
see the working plan for the full rationale and every decision behind that
isolation (git branch, gate exclusions, MLflow scope, dataset license,
report language, compute budget):

> **Plan**: `dl-course-final-project-plan.md` (repo root, untracked —
> ask whoever is running this branch for a copy if you don't have one).

## What lives here

Three segmentation configurations trained on `starcop_mini`
(392/49/441 train/val/test 128×128 patches, 4-channel `mag1c` +
AVIRIS-NG TOA reflectance input), reusing this repo's normalization
contract and dataset:

- **E1** — small from-scratch CNN encoder-decoder (course baseline).
- **E2** — U-Net + MobileNetV2, built directly from
  `segmentation-models-pytorch` (own implementation for this course, not
  an import of `vendor/starcop` or `src/baselines/starcop/`).
- **E3** — LinkNet + MobileNetV3-small-minimal, reproduced from Herec et
  al. (2026) and trained from scratch on `starcop_mini`.

## Isolation from the main repo

- **Branch**: this work lives on `coursework/dl-final-project`, cut from
  `main` (by way of `chore/coursework-gate-isolation`, which carries the
  gate-exclusion changes below). This branch is never merged into `main`
  as a whole — only specific, reviewed pieces get promoted later via their
  own separate feature branch and PR.
- **Gates**: excluded from `ruff` (`pyproject.toml`'s `[tool.ruff]
  extend-exclude`), `interrogate` (`[tool.interrogate] exclude`), and CI's
  `lint.yml` changed-files filter — the same mechanism `vendor/` uses. Not
  part of `pyproject.toml`'s `testpaths` or any `ENV_RESEARCH_*` test path,
  so `make test` / `make test-research` / CI never touch it.
- **Its own commands**: `make coursework-test`, `make coursework-lint`,
  `make coursework-train` (repo root `Makefile`) — scoped to this
  directory only.
- **MLflow**: tracks to a local `mlruns/` directory scoped to this folder
  (already covered by the repo's existing unanchored `mlruns/`
  gitignore entry), not the shared thesis tracking server.

## Lifecycle

`coursework/dl-final-project/`, its branch, and the `Makefile`'s
`coursework-*` targets are all deleted once the course is graded and any
reusable pieces (see the plan's Section 15) have been promoted into
`src/` through their own normal PR.

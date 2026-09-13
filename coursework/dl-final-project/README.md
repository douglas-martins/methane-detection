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

Three segmentation configurations over a 4-channel `mag1c` + AVIRIS-NG TOA
reflectance input, reusing this repo's normalization contract and dataset:

- **E1** — small from-scratch CNN encoder-decoder (course baseline).
- **E2** — U-Net + MobileNetV2, built directly from
  `segmentation-models-pytorch` (own implementation for this course, not
  an import of `vendor/starcop` or `src/baselines/starcop/`).
- **E3** — LinkNet + MobileNetV3-small-minimal, reproduced from Herec et
  al. (2026) and trained from scratch.

## Two dataset tiers — `starcop_mini` develops, `starcop_raw` confirms

The course asks for small-scale training, and the graded comparison
honors that. But the course's instruction constrains the *training budget
of the deliverable* — it says nothing about whether the code is known to
work on the real dataset, or whether the conclusions hold there. So this
project runs both tiers, and labels every number with the tier it came
from:

- **`starcop_mini`** (392/49/441 train/val/test 128×128 patches) — the
  development loop and the source of the report's headline E1/E2/E3
  comparison.
- **`starcop_raw`** (141k/26.6k/16.8k patches) — the confirmation tier.
  Nothing here is considered done until it has also been exercised
  against real raw data.

Confirmation happens at three levels (the plan's Section 0.1 is the
authority; this is the summary):

| Level | What it confirms | Cost |
| --- | --- | --- |
| **R1** | The *same code path* runs on real `starcop_raw` patches — band order, dtypes, shapes, post-normalization ranges, NaN/nodata handling. Mandatory for every stage that produces code. | Minutes (`make coursework-confirm-raw`) |
| **R2** | The mini-scale *conclusions* survive 10–25× more data, via a seeded scene-disjoint `starcop_raw` subsample trained on the same schedule. Mandatory for the training/evaluation stages. | ~1 h per configuration |
| **R3** | The full 141k-patch `starcop_raw` train split, E2/E3 only — the thesis-facing number. Best-effort, off the critical path. | Overnight |

Consequences for anything written here:

- The dataloader is **dataset-parameterized from the start**
  (`dataset=starcop_mini|starcop_raw`), never hard-coded to either tier.
- `data/processed/starcop_raw/` must exist before any R-level runs — it
  is not built by default, and any existing `dvc.lock` entries for it are
  likely stale against current pipeline code (only `dvc repro`, never
  `dvc pull`, is safe here):
  ```bash
  dvc repro normalize@starcop_raw split@starcop_raw patch_extract@starcop_raw
  dvc repro stats@starcop_raw coordinates@starcop_raw
  ```
  Measured ~11.5 GB of output, ~2–4 hours total — dominated by
  `stats@starcop_raw` (single-threaded, no `num_workers` knob), not by
  `patch_extract` despite its `num_workers: 4` config. See the plan's
  Section 0.1 for the per-stage timing breakdown.
- R2's subsample is a plain CSV filter over
  `patch_extract@starcop_raw`'s train output (seeded flightline sample,
  manifest persisted here) — no new DVC stage or dataset config. See the
  plan's Section 0.1.
- Every MLflow run carries `dataset` and `tier` params, so a number can
  always be traced back to the data that produced it.

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
  `make coursework-train`, `make coursework-confirm-raw` (repo root
  `Makefile`) — scoped to this directory only.
- **What isolation does *not* relax**: the R1 raw confirmation above. The
  gate exclusions exist so the coursework isn't blocked on thesis-scale
  tooling ceremony, not so it can ship code that was never run against
  the real dataset.
- **MLflow**: tracks to a local `mlruns/` directory scoped to this folder
  (already covered by the repo's existing unanchored `mlruns/`
  gitignore entry), not the shared thesis tracking server.

## Lifecycle

`coursework/dl-final-project/`, its branch, and the `Makefile`'s
`coursework-*` targets are all deleted once the course is graded and any
reusable pieces (see the plan's Section 15) have been promoted into
`src/` through their own normal PR.

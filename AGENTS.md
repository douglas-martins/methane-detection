# AGENTS.md

Instructions for AI coding agents (Claude, Cursor, Aider, Codex, etc.) working
in this repository. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the
human-facing setup, environments, and PR workflow this file assumes.

## Project overview

A methane plume detector built on the STARCOP baseline, targeting on-board/embedded deployment (hls4ml/Vitis AI on FPGA-style hardware — see `docs/`). Alongside the model, this repo carries the full MLOps pipeline that makes iterating toward it fast and reproducible: DVC (data versioning), MLflow (experiment tracking + model registry), Prefect (retraining orchestration), BentoML (serving), Prometheus/Grafana (monitoring). This is a master's thesis project — see `CONTRIBUTING.md` for the full human-facing setup and workflow.

## Two Python environments — read before running anything

- **baseline env** (`vendor/starcop/.venv`, Python 3.10, torch 1.13.1) — the original STARCOP stack, reference-only, exact-pinned for reproducing the original paper's checkpoints.
- **research env** (`.venv`, Python 3.12, torch ≥2.5) — active development; where the actual thesis solution is built. This is the default for anything not explicitly baseline work.

Every `make` target and script is environment-specific (see the `ENV_BASELINE_*` / `ENV_RESEARCH_*` variables in `Makefile`) — check which one a command belongs to before running it. Setup commands for both are in `CONTRIBUTING.md`.

## Common commands

```bash
make test-baseline          # baseline env test suite
make test-research          # research env test suite
make test                   # both suites
make coverage                # baseline env, with coverage (junit.xml, coverage.xml)
make coverage-research       # research env, with coverage (junit-research.xml, coverage-research.xml)
make lint                    # ruff check + ruff format --check (research env)
make docstring-coverage      # interrogate docstring-coverage gate (research env, 80% threshold)
make test-scripts            # bats suite for scripts/, in Docker
make docs-serve               # serve MkDocs site locally
```

Run a single test directly with the environment's own interpreter, e.g.:

```bash
.venv/bin/python -m pytest src/baselines/starcop/serving/__tests__/test_inference.py -v
.venv/bin/python -m pytest src/baselines/starcop/serving/__tests__/test_inference.py::test_name -v
vendor/starcop/.venv/bin/python -m pytest src/baselines/starcop/training/__tests__/test_model_forward.py -v
```

Tests live in `__tests__/` folders next to the module they cover, named `test_*.py`; shared fixtures are in the root `conftest.py`. CI's `lint.yml` only lints the PR's *changed* Python files (main carries pre-existing ruff findings), so `make lint` running clean on your changed files is what matters, not a clean full-repo run.

`make lint` and `make docstring-coverage` are a key validation step after
any Python implementation, not an optional cleanup pass — the always-apply
constraint is in
[`.agents/rules/lint-and-docstring-coverage.md`](.agents/rules/lint-and-docstring-coverage.md),
and the operational checklist (scoping lint to your diff, running the
full-repo docstring gate, fixing rather than suppressing) is in
[`.agents/skills/lint-and-docstring-coverage/SKILL.md`](.agents/skills/lint-and-docstring-coverage/SKILL.md)
— `.claude/skills/lint-and-docstring-coverage` symlinks to it so Claude Code
picks it up the same way.

## Architecture

[`src/README.md`](src/README.md) is the contributor-facing ownership map for
baseline adapters, shared infrastructure, candidate models, and comparisons.

### `vendor/starcop/` is never edited — compose from outside

It's a pinned git submodule (the original STARCOP paper code). Nothing under it is ever changed, not even transiently. Everything that needs different behavior does it from outside: subclassing (e.g. `starcop_datamodule.py`'s `ProcessedDatasetDataModule` subclasses `Permian2019DataModule`, overriding only `prepare_data()`), runtime attribute overrides, or `types.MethodType` monkeypatching on one instance (e.g. `train.py` rebinding `model.val_epoch_end`). `pyproject.toml`'s ruff config excludes `vendor/` entirely for the same reason.

### Per-package `_vendor_starcop*.py` seam files

Each consuming package under `src/` has its **own** `_vendor_starcop*.py` module (`src/data/preprocessing/_vendor_starcop.py`, `src/baselines/starcop/training/_vendor_starcop_training.py`, `src/baselines/starcop/evaluation/_vendor_starcop_evaluation.py`, `src/baselines/starcop/registry/_vendor_starcop_baseline.py`, `src/baselines/starcop/serving/_vendor_starcop_serving.py`) rather than one shared import. Each puts `vendor/starcop` on `sys.path` and re-exports the specific STARCOP objects that package composes around. This is deliberately duplicated per-package: pytest's flat/prepend import mode caches modules by bare name, and multiple packages land on `sys.path` simultaneously under `make test-research`, so a single shared `_vendor_starcop.py` would collide in `sys.modules`. Every other file in a package imports STARCOP objects from its own local seam file, never from `starcop.*` directly.

### No `__init__.py` in `src/` — flat import convention throughout.

### Data pipeline (DVC + Hydra)

`dvc.yaml` defines per-dataset stages (`starcop_mini`, `starcop_raw`): `normalize → split → patch_extract`, plus `stats` and `coordinates`. Each stage is a Hydra-configured script under `src/data/preprocessing/`, with base config in `configs/data.yaml` and per-dataset overrides in `configs/dataset/*.yaml` (selected via `dataset=starcop_mini|starcop_raw` on the CLI).

### STARCOP training entrypoint (`src/baselines/starcop/training/train.py`)

Baseline-specific orchestration lives under `src/baselines/starcop/training/`; the generic MLflow and DVC lineage helpers remain under `src/training/`. The entrypoint is not a modification of `vendor/starcop/scripts/train.py` — it imports every STARCOP building block unmodified and composes new behavior around it: Hydra config merges STARCOP's own `vendor/starcop/scripts/configs/config.yaml` with `configs/training/overlay.yaml` (`settings_overlay.py`); data loading subclasses `Permian2019DataModule` to read this project's `data/processed/<dataset>/{patches,splits}/` layout instead of STARCOP's own file-discovery convention; Lightning 2.x / torch compat shims (`lightning2_compat.py`, `optimizer_compat.py`) rebind STARCOP's pre-2.0 hooks as no-ops under newer versions, so baseline env's older pins are unaffected. Run with either environment's interpreter depending on machine/GPU — see the module's own docstring and `internal-docs/setup/environment-notes.md` for which machines need which environment (e.g. Blackwell GPUs need research env; baseline env's exact-pinned torch has no working CUDA kernels for `sm_120`).

### STARCOP evaluation (`src/baselines/starcop/evaluation/`)

Paper-metric evaluation and local live-verification logic live with the baseline.
The stable wrappers remain `scripts/run_starcop_baseline_evaluation.py` and
`scripts/run_live_verify.py`; repo-owned direct imports must use the relocated
baseline path.

### STARCOP serving (`src/baselines/starcop/serving/`)

`service.py` is a thin BentoML wrapper (env var reads, exception→HTTP translation) that loads a model from the MLflow registry at startup and exposes `POST /predict`, `POST /health` (`GET /metrics` is BentoML's own built-in Prometheus endpoint). The actual predict logic (assemble → infer → shape response) lives in `inference.py::predict_response` and is unit tested directly; `service.py` itself is exercised via a real `bentoml serve` + curl run, not unit tests — this "thin glue vs. tested logic" split recurs elsewhere (`train.py`, `hf_baseline_import.py`, `launch_profiles.py`, `promotion_criteria.py`) and explains why some files are intentionally excluded from the coverage gate (see `pyproject.toml`'s `[tool.coverage.report].omit`). Shared rolling drift mathematics and the `BandStats` value type remain under `src/serving/`; shared code does not import the baseline adapter.

### Registry

Shared MLflow registry wiring and promotion policy remain under `src/registry/`.
The STARCOP checkpoint importer and its vendor seam live under
`src/baselines/starcop/registry/`; baseline adapters may depend on shared registry
infrastructure, never the reverse.

### Candidate models and comparison

Independent research models belong under `src/models/<candidate>/`; they do not
modify or hide inside the STARCOP baseline. Cross-model quality and inference-time
reports belong under `src/comparison/`, while STARCOP paper-metric evaluation stays
under `src/baselines/starcop/evaluation/`.

### Retraining loop (`flows/retrain.py`)

A Prefect flow (research env only — where `prefect` is installed) orchestrating trigger → pull → train → registry → gate → redeploy, run as a scheduled flow run on a Process work pool.

### Testing conventions

- Test-first (RED → GREEN → REFACTOR) is this repo's established pattern for non-trivial changes.
- Real fixtures over mocks — a real tmp-path DVC repo, a real tiny GeoTIFF, a real sqlite-backed MLflow store, rather than `Mock()`/interaction checks. Small hand-written fakes exposing only the used surface are fine; broad mocking is not.
- Test method names are intentionally undocumented (no docstrings) — the descriptive name already reads as the spec (`interrogate` config in `pyproject.toml` exempts `__tests__/`, private, magic, and nested functions from the docstring-coverage gate for this reason).

The always-apply constraint behind this — write the failing test first,
before any new Python file exists, especially under `src/` — is in
[`.agents/rules/test-driven-development.md`](.agents/rules/test-driven-development.md).
The full cycle, patterns, and examples are in
[`.agents/skills/test-driven-development/SKILL.md`](.agents/skills/test-driven-development/SKILL.md)
— agents that support the `.agents/skills` convention should load it
directly; `.claude/skills/test-driven-development` symlinks to it so Claude
Code picks it up the same way.

### Mutation testing (rolling out)

Line coverage proves code ran; it doesn't prove a test would catch a
behavior change. This repo is introducing mutmut mutation testing as a
quality gate on top of the existing coverage gates, module by module.
`[tool.mutmut] only_mutate` in `pyproject.toml`, once it exists, is the
authoritative list of modules currently gated; until then this gate hasn't
been bootstrapped and nothing is enforced yet.

The always-apply rule — kill every surviving mutant with a real test or
justify it explicitly (`# pragma: no mutate` / a `do_not_mutate` entry with
a reason), never shrink the gate's scope or disable its settings just to
force a pass — is in
[`.agents/rules/mutation-testing.md`](.agents/rules/mutation-testing.md).
The operational procedure (running mutmut, triaging survivors, growing the
gate) is in
[`.agents/skills/mutation-testing/SKILL.md`](.agents/skills/mutation-testing/SKILL.md)
— `.claude/skills/mutation-testing` symlinks to it so Claude Code picks it
up the same way.

## Commit Guidelines

This repo is trunk-based against a single long-lived `main` — **PR-only, never
commit directly to `main`**.

The always-apply constraints (never commit to `main`, split by type, no `git
add -A`, no AI attribution, …) are in
[`.agents/rules/conventional-commit.md`](.agents/rules/conventional-commit.md)
— follow them any time you're about to commit in this repo, whether or not
you invoke a skill.

An operational, step-by-step checklist for applying those rules lives in
[`.agents/skills/conventional-commit/SKILL.md`](.agents/skills/conventional-commit/SKILL.md)
— agents that support the `.agents/skills` convention should load it
directly; `.claude/skills/conventional-commit` symlinks to it so Claude Code
picks it up the same way.

This repo enforces [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)
via commitlint (`commitlint.config.cjs`) and drives versioning + `CHANGELOG.md`
with `python-semantic-release` on merge to `main`. A badly-typed or
badly-split commit either fails CI or pollutes the changelog, so agents must
follow this process exactly — not just write a conventional-looking subject
line.

### 1. Format

```text
<type>(<scope>)!: <description>

[optional body]

[optional footer(s)]
```

- `description`: imperative mood, lowercase, no trailing period.
- `scope`: optional, lowercase, names the affected area (`api`, `codecov`,
  `serving`, `docs`, …).
- `!` after type/scope, or a `BREAKING CHANGE:` footer, for breaking changes.

### 2. Types and their release effect

python-semantic-release reads the type to compute the version bump:

| Type | Effect |
|---|---|
| `feat` | MINOR bump |
| `fix`, `perf` | PATCH bump |
| `docs`, `style`, `refactor`, `test`, `build`, `ci`, `chore` | no bump (unless `BREAKING CHANGE`) |

Pick the type that matches what the change *does*, not the task ticket it
came from — a single task can and often should produce commits of different
types.

### 3. Splitting changes into commits

Before committing, inspect every staged, unstaged, and untracked file that's
actually part of the current task (`git status`, `git diff`, `git diff
--cached`).

- **Group by type/scope, not by task.** A `fix` mixed into the same commit as
  an unrelated `docs` change dilutes that fix's changelog entry with unrelated
  wording changes, and vice versa.
- **One logical change per commit.** If two files fix unrelated bugs, that's
  two `fix` commits, not one.
- **Stage explicitly.** Use `git add <specific files>` per commit — never
  `git add -A` / `git add .` — so unrelated in-progress or pre-existing
  untracked files aren't swept in. If unsure whether an untracked file
  belongs, leave it out and say so rather than guessing.
- **Worked example** (from splitting a review-fixes task in this repo):
  a coverage-config correctness fix and a terminology/grammar doc cleanup
  touched files in the same task, but shipped as two separate commits —
  `fix(codecov): add missing paths to research coverage flag` and
  `docs: finish Environment A/B to baseline/research env renaming` — so the
  generated changelog shows a real fix, not a fix-that's-actually-just-docs.

### 4. Message content

- Write commit messages in **English**, regardless of the conversation
  language.
- Body/footer explain *why*, not *what* — the diff already shows what
  changed.
- **No AI attribution or co-author trailers** (no `Co-Authored-By`, no
  session links, no "generated by" notices) in commit messages for this
  repo. Commits should read as authored by the contributor.

### 5. Producing the commit

Prefer a heredoc for the message to avoid shell-escaping issues, and never
push unless explicitly asked:

```bash
git add <files-for-this-commit>
git commit -m "$(cat <<'EOF'
<type>(<scope>): <description>

<body>
EOF
)"
```

Report back: the logical groups you identified, the message for each, and
the commands used (or about to be run) — so the split is auditable, not just
the end state.

## Documentation split

- **Public docs** (`docs/`, published via MkDocs to GitHub Pages) — architecture, methodology, dataset, results, model registry policy. Reader-facing, stable. Much of it is still placeholder content pending a later pass.
- **Internal docs** (`internal-docs/`) — implementation journal, decision log, credentials-adjacent setup guides (`internal-docs/setup/`), runbooks, live model-experiments tracker. Not published, but not secret.

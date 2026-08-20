# Contributing to Methane Detection

After cloning, read this before touching anything — this repo runs **two isolated
Python environments** and one **never-edit** rule that shapes most of `src/`.

*Commit using [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) —
this repo's release automation (python-semantic-release) parses your commit type
directly into version bumps and `CHANGELOG.md` entries.*

## Development

### Environments

- **Environment A** (`vendor/starcop/.venv`, Python 3.10, torch 1.13.1) — the
  original STARCOP stack, reference-only.
- **Environment B** (`.venv`, Python 3.12, torch ≥2.5) — active development.
- **[TODO] Environment C** — may be needed if FPGA-style on-board work (see the
  MkDocs site's roadmap; `hls4ml` is one candidate approach, not the only one)
  moves forward: `hls4ml` requires Linux and does not support Windows/macOS
  natively. Not designed yet; flagged here so it isn't a surprise later.

```bash
# Clone (includes STARCOP submodule)
git clone --recurse-submodules https://github.com/douglas-martins/methane-detection
# or, if you already cloned without --recurse-submodules:
git submodule update --init

# Environment A — STARCOP original
cd vendor/starcop
uv venv --python 3.10
uv pip install -r requirements.txt
uv pip install -e .
cd ../..

# Environment B — MLOps project
uv venv --python 3.12
uv sync
```

### Scripts

| Command | What it does |
|---|---|
| `make test-env-a` | Run the Environment A suite |
| `make test-env-b` | Run the Environment B suite |
| `make test` | Run both suites (`test-env-a` + `test-env-b`) |
| `make coverage` / `make coverage-env-b` | Run with coverage, write junit + coverage XML |
| `make badges` / `make badges-env-b` | Regenerate `docs/badges/*.svg` |
| `make lint` | `ruff check` + `ruff format --check` (Environment B) |
| `make docstring-coverage` | `interrogate` docstring-coverage check (Environment B) |
| `make test-scripts` | Run the `bats` suite for `scripts/` in Docker |
| `make docs-serve` | Serve the MkDocs site locally |
| `make docs-build` | Build the MkDocs site (`--strict`) |

### Testing conventions

- **Test-first (RED → GREEN → REFACTOR)** — this repo's established pattern for
  every non-trivial change.
- **Real fixtures over mocks** — prefer a real tmp-path DVC repo, a real tiny
  GeoTIFF, a real sqlite-backed MLflow store over `Mock()`/interaction checks.
  Fakes (small hand-written stand-ins exposing only the used surface) are fine;
  broad mocking is not.
- Tests live in `__tests__/` folders next to the module they cover; shared
  fixtures in the root `conftest.py`.

## Architecture

> [!IMPORTANT]
> **`vendor/starcop/` is never edited — not even transiently, not even to write a
> test.** It's a pinned git submodule. Everything that needs to change STARCOP's
> behavior does it from outside: subclassing, runtime attribute overrides, or
> `types.MethodType` monkeypatching on one instance. See the Pipeline Overview
> page on the [documentation site](https://douglas-martins.github.io/methane-detection/)
> for the full rationale (once that page's content lands).

- No `__init__.py` in `src/` — flat import convention throughout.
- `_vendor_starcop*.py` shim modules are the only files allowed to
  `sys.path.insert(0, "vendor/starcop")`; everything else imports from the shim.

## Documentation

- **Public docs** (`docs/`, published via MkDocs to
  [GitHub Pages](https://douglas-martins.github.io/methane-detection/)) —
  architecture, methodology, dataset, results, model registry policy. Stable,
  reader-facing only.
- **Internal docs** (`internal-docs/`) — implementation journal, decision log,
  credentials-adjacent setup guides, runbooks, and the live model-experiments
  tracker. Not published, but not secret — it's a curation boundary, not an
  access-control one.
- Use GitHub-style alerts (`[!NOTE]`, `[!TIP]`, `[!IMPORTANT]`, `[!WARNING]`,
  `[!CAUTION]`) and Mermaid diagrams where they clarify structure or flow.

To preview docs locally: `make docs-serve`.

## Opening a Pull Request

This is a small research/thesis project on **trunk-based development against a
single long-lived branch, `main`** — no `alpha`/`beta` release channels (there's
no published package with pre-release consumers). Branch from `main`, PR back to
`main`, delete the branch after merge.

```mermaid
gitGraph
    commit id: "v0.4.0"
    branch feat/some-feature
    commit
    commit
    checkout main
    branch fix/some-bug
    commit
    checkout main
    merge fix/some-bug tag: "v0.4.1"
    checkout feat/some-feature
    commit
    checkout main
    merge feat/some-feature tag: "v0.5.0"
```

- `main` is protected against direct pushes — always PR, and at least one
  approving review is required before merge.
- CI (`lint.yml`, `tests.yml`, `scripts-tests.yml`, `commitlint.yml`) must pass,
  and all review conversations must be resolved, before merge.
- On merge, `release.yml` (python-semantic-release) computes the version bump
  from Conventional Commit types (`feat`→minor, `fix`/`perf`→patch), regenerates
  `CHANGELOG.md`, tags, and publishes a GitHub Release — automatically, no
  manual version bookkeeping.

## Publishing

Both the Docker image (`cd.yml` → `ghcr.io` → Coolify deploy) and this
documentation site (`docs.yml` → GitHub Pages) deploy automatically on merge to
`main`. No manual publish steps.

# Add Codecov Upload + README Badge

> **Status**: 🟢 Repo-level changes complete, Codecov account linked by user — awaiting commit/push and post-merge verification

## Context

CI (`.github/workflows/tests.yml`) already produces `coverage.xml` (Environment A) and `coverage-env-b.xml` (Environment B) via `pytest-cov`, uploads both as GitHub Actions build artifacts, and generates **static** SVG badges locally with `genbadge` (`docs/badges/{tests,coverage}-env-{a,b}.svg`), committed back to the repo by a CI bot on every push to `main`. Those badges already appear at the top of `README.md`.

What's missing is a real coverage-tracking service: the XML reports currently only exist as CI run artifacts (90-day retention, not linked anywhere) and a static, point-in-time badge. There's no coverage trend over time, no automatic PR diff-coverage comments, and the badge only updates on a `push` to `main` (not on PRs) because that's when the bot-commit step runs.

The user confirmed the goal: integrate **Codecov** (chosen over Coveralls / over just improving artifact discoverability) — free for public repos (this repo is public: `douglas-martins/methane-detection`), gives coverage trend graphs, PR diff-coverage comments, and a dynamic README badge that doesn't depend on a bot commit.

**Important boundary**: every repo-level change below can be implemented directly. Creating the Codecov account or generating its token cannot — that's an external, one-time manual step for the user (detailed below). The plan is written so the CI change works either way (tokenless upload is supported for public repos), and upgrades automatically once the user adds the token.

## What gets built

### 1. `.github/workflows/tests.yml` — upload each job's coverage XML to Codecov

Add one step to **each** existing job (`environment-a-tests`, `environment-b-tests`), right after their respective `Run tests with coverage` step (so it runs alongside the existing `Upload test reports` artifact step, not replacing it):

```yaml
- name: Upload coverage to Codecov
  uses: codecov/codecov-action@v5
  with:
    files: coverage.xml            # coverage-env-b.xml in the env-b job
    flags: env-a                   # env-b in the env-b job
    name: env-a-coverage           # env-b-coverage in the env-b job
    token: ${{ secrets.CODECOV_TOKEN }}
    fail_ci_if_error: false
```

- `flags:` keeps the two environments' coverage separate in Codecov's UI instead of merging them into one number — mirrors the existing `-env-a`/`-env-b` split already used for badges and artifacts.
- `token` reads a `CODECOV_TOKEN` repo secret if the user adds one (see manual steps below). If absent, the action still works tokenless for public repos — just slightly more prone to rate-limit flakiness on busy days. `fail_ci_if_error: false` means a Codecov outage never fails the build.
- No changes needed to the `on:` triggers — both `pull_request` and `push` are already covered, which is what enables Codecov's PR diff-coverage comments.
- No `permissions:` changes needed (the action only needs the default `contents: read`).

### 2. `codecov.yml` (new, repo root) — flag definitions + sane default status checks

```yaml
coverage:
  status:
    project:
      default:
        target: auto      # compare against the base commit rather than a fixed %
        threshold: 1%      # tolerate small fluctuation instead of blocking every PR
    patch:
      default:
        target: auto
comment:
  layout: "reach, diff, flags, files"
  require_changes: false
flags:
  env-a:
    paths:
      - src/data/download/
      - vendor/starcop/scripts/preprocessing/
      - src/training/
  env-b:
    paths:
      - src/data/preprocessing/
      - src/training/
      - src/registry/
```

The `flags.*.paths` mirror the Makefile's existing `ENV_A_COV_PATHS`/`ENV_B_COV_PATHS` exactly (`Makefile:4,11`), so Codecov's per-flag breakdown lines up with what each environment actually measures. `target: auto` avoids hard-failing PRs against an arbitrary percentage before there's a real baseline — tunable later.

> **Corrected during review**: the original draft of this section omitted `src/training/` from `env-a` and both `src/training/` and `src/registry/` from `env-b`, and cited the wrong Makefile line (9, which is `ENV_B_TEST_PATHS`, not `ENV_B_COV_PATHS` at line 11). Fixed above to actually match the Makefile.

### 3. `README.md` — add one dynamic Codecov badge alongside the existing badges

README.md currently has **five** badges (four test/coverage badges plus a `Lint` badge from the newly added `.github/workflows/lint.yml`, not yet committed as of this plan). The Codecov badge is inserted before the Lint badge; nothing else in the block changes:

```markdown
![tests (env A)](docs/badges/tests-env-a.svg)
![coverage (env A)](docs/badges/coverage-env-a.svg)
![tests (env B)](docs/badges/tests-env-b.svg)
![coverage (env B)](docs/badges/coverage-env-b.svg)
[![codecov](https://codecov.io/gh/douglas-martins/methane-detection/graph/badge.svg)](https://codecov.io/gh/douglas-martins/methane-detection)
[![Lint](https://github.com/douglas-martins/methane-detection/actions/workflows/lint.yml/badge.svg)](https://github.com/douglas-martins/methane-detection/actions/workflows/lint.yml)
```

This is additive, not a replacement — all five existing badges keep working exactly as they do today. A short paragraph near the existing "Testing" section will point at the Codecov dashboard link, consistent with how that section already documents `make coverage`/`make badges`.

> **Corrected during review**: the original draft's snippet only listed the four original test/coverage badges and, if pasted in literally as a block replacement, would have silently dropped the Lint badge. Fixed above — append only.

**Note on the badge URL**: this is Codecov's standard public-repo badge pattern and should resolve once the repo is linked at codecov.io (see manual steps below). If Codecov's dashboard hands the user a slightly different snippet (e.g. with an embedded graph token), swap it in — a live, account-specific URL can't be generated without the account existing.

### 4. ~~`.gitignore` — small adjacent fix~~ (dropped — not actually broken)

The original draft claimed `coverage-env-b.xml`/`junit-env-b.xml` were untracked-but-not-ignored. Verified with `git check-ignore -v`: both are already caught by the wildcard patterns `coverage*.xml` (`.gitignore:66`) and `junit*.xml` (`.gitignore:67`), added after the exact-name entries the original draft was looking at. No change needed here.

## What the user needs to do outside the repo (cannot be automated)

1. Sign in at [codecov.io](https://codecov.io) with GitHub, and enable/link the `douglas-martins/methane-detection` repo.
2. *(Optional but recommended)* Copy the repo's upload token from Codecov's settings page, then add it as a GitHub Actions repo secret named `CODECOV_TOKEN` (GitHub repo → Settings → Secrets and variables → Actions → New repository secret). Without this, uploads still work tokenlessly for public repos, just less reliably.
3. After the first successful CI run uploads data, check the badge URL in step 3 actually renders — if Codecov gave a different exact snippet on its dashboard, swap it into `README.md`.

## Files to be touched (when implemented)

- `.github/workflows/tests.yml` — one new step per job
- `codecov.yml` — new
- `README.md` — one new badge line + a short doc note

## Verification (once this plan is executed)

1. `yamllint .github/workflows/tests.yml codecov.yml` (or a visual review) — valid YAML, matches existing step style in the file.
2. Push to a branch / open a PR: confirm both jobs' new "Upload coverage to Codecov" steps run and report success (won't get real Codecov trend data until the user completes the external signup, but the step itself should complete either way given `fail_ci_if_error: false`).
3. Once the user links the repo at codecov.io and this merges to `main`: confirm the Codecov dashboard shows both `env-a` and `env-b` flags with data, and that the README badge renders (not a broken-image icon).

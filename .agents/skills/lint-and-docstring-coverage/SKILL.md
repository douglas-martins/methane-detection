---
name: lint-and-docstring-coverage
description: >
  Run and satisfy this repo's two static-quality gates — ruff lint/format
  and interrogate docstring coverage — after any Python implementation.
  Use whenever you've finished writing or editing Python code, before
  reporting a task done. Triggers: "run lint", "check docstring coverage",
  "is this ready", "did I break the docstring gate".
---

# Lint and Docstring Coverage (this repo)

The canonical rules live in
[`.agents/rules/lint-and-docstring-coverage.md`](../../rules/lint-and-docstring-coverage.md)
— read it before proceeding if it's not already in context. This file is
the operational checklist for applying those rules.

## Procedure

1. **Identify what you changed.**

   ```bash
   git diff --name-only --diff-filter=ACMR -- '*.py' | grep -v '^vendor/'
   ```

   This is your lint scope — the same base/head-diff approach
   `.github/workflows/lint.yml` uses in CI.

2. **Run ruff against just those files** (matches CI; faster than the
   full-repo `make lint` and doesn't surface `main`'s pre-existing
   findings):

   ```bash
   .venv/bin/ruff check <changed-files>
   .venv/bin/ruff format --check <changed-files>
   ```

   Fix every finding ruff reports on your changed files. If `ruff format
   --check` fails, run `.venv/bin/ruff format <changed-files>` and re-check
   rather than hand-formatting.

3. **Run the full-repo docstring gate** — unlike lint, this one is *not*
   scoped to your diff:

   ```bash
   make docstring-coverage
   ```

   or directly: `.venv/bin/interrogate -v .`

4. **If it fails, find what you introduced.** `interrogate -v .` lists
   every undocumented item; cross-reference against your diff. Add a
   one-line docstring to each new public module, class, function, or
   method you added. Skip (don't force a docstring onto):
   - anything under `__tests__/` (exempted — descriptive test names are
     the spec, per `AGENTS.md`'s testing conventions)
   - private (`_name`) or magic (`__name__`) methods
   - nested functions
   - anything under `vendor/` (never touched at all)

5. **Re-run both gates** after fixes, the same way as steps 2 and 3 — don't
   assume a fix worked without re-running.

6. **Report**: which files you ran ruff against and why (the diff scope),
   whether docstring coverage passed and what you added if it didn't, and
   the exact commands used — so the check is auditable, not just asserted.

## Notes

- Don't run bare `make lint` and treat unrelated pre-existing findings
  elsewhere in the repo as yours to fix — that's `main`'s known state, not
  a regression you introduced (see `lint.yml`'s comment on this).
- Don't lower `fail-under` in `pyproject.toml` or add blanket `# noqa` /
  interrogate exemptions to force a pass — see the rule file for why.
- This gate pair is research-env only (`.venv`, not
  `vendor/starcop/.venv`) — both tools live in the research env's
  dependency set.

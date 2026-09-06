# Rule: Lint and Docstring Coverage

Always-on validation for any agent finishing a Python implementation in this
repository — these gates run every time, whether or not the
`lint-and-docstring-coverage` skill has been explicitly invoked. For the
step-by-step procedure (scoping, running, interpreting failures), use the
skill at
[`.agents/skills/lint-and-docstring-coverage/SKILL.md`](../skills/lint-and-docstring-coverage/SKILL.md).

1. **Never call a Python implementation done without running both gates**
   on the research env tooling: `make lint` and `make docstring-coverage`
   (or the equivalent `.venv/bin/ruff` / `.venv/bin/interrogate`
   invocations when you only touched a subset of files).
2. **Lint is scoped to your changed files, not the whole repo.** `main`
   carries pre-existing ruff findings (see `.github/workflows/lint.yml`),
   so a clean `make lint` on files you didn't touch is neither required nor
   your responsibility. Run ruff against the files you actually added or
   modified (`git diff --name-only --diff-filter=ACMR ... -- '*.py'`) and
   require those to be clean.
3. **Docstring coverage is a whole-repo gate, not scoped to your diff.**
   `make docstring-coverage` enforces `fail-under = 80` (see
   `[tool.interrogate]` in `pyproject.toml`) across the entire tree except
   `__tests__/`, `vendor/`, and `.venv/`. Any new public
   module/class/function/method you add needs a docstring unless it's
   private (`_leading_underscore`), magic (`__dunder__`), a nested
   function, or under `__tests__/` — those are exempted by
   `pyproject.toml`, not by omission.
4. **`vendor/starcop/` is exempt from both gates** — it's excluded from
   ruff (`extend-exclude`) and from interrogate (`exclude`). Never add
   docstrings to it or reformat it to satisfy either gate; compose from
   outside instead (see this file's own vendoring section in `AGENTS.md`).
5. **A failing gate blocks completion, not a follow-up.** Fix ruff findings
   and add missing docstrings in the same change, before reporting the task
   finished — don't defer either to "a follow-up cleanup."
6. **Never bypass either gate** to force a pass — no blanket `# noqa`
   suppressions, no interrogate exemptions added just to dodge a real gap,
   no lowering `fail-under` in `pyproject.toml`. Fix the underlying code or
   add the docstring instead. If a rule genuinely doesn't apply to a given
   line, that's a conversation with the user, not a unilateral suppression.

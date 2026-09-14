# Rule: Mutation Testing

Always-on constraint for any agent touching a module gated by mutation
testing, or extending that gate to a new module — this applies even when the
`mutation-testing` skill hasn't been explicitly invoked. For the full
procedure (running mutmut, triaging survivors, growing the gate), use the
skill at
[`.agents/skills/mutation-testing/SKILL.md`](../skills/mutation-testing/SKILL.md).
For the rollout's phase-by-phase scope and status, see
`mutation-testing-plan.md` at the repo root — an untracked local working
document, not present on other clones, so never treat it as the sole source
of truth for what's currently gated.

0. **This rule only applies once mutation testing is bootstrapped.** If
   `[tool.mutmut]` doesn't yet exist in `pyproject.toml`, the gate hasn't
   landed (see Phase 0 in `mutation-testing-plan.md`) — nothing below applies
   yet.
1. **`[tool.mutmut] only_mutate` in `pyproject.toml` is the authoritative,
   tracked scope** — not the plan file. If you're changing the logic of a
   module listed there, mutation testing applies to your change. If it isn't
   listed, it hasn't been rolled out to that module yet.
2. **Kill-or-justify, no unexplained survivors.** A gated module never
   regresses: a change to it must not leave a new surviving mutant
   unexplained. Either strengthen a test to kill it, or exclude it with
   `# pragma: no mutate` / a `do_not_mutate` entry carrying a one-line reason
   (genuinely equivalent or untestable mutants only — a log string, an
   unreachable defensive branch). "Come back to it later" is not a valid
   state to leave a survivor in.
3. **Never lower the bar to force a pass** — no shrinking `only_mutate`, no
   blanket `do_not_mutate`/pragma additions to dodge a real gap, no disabling
   `mutate_only_covered_lines` or the heavy-module settings just to skip
   work. Same principle as the lint/docstring-coverage gate: fix the
   underlying test or code, don't suppress the check.
4. **Use the correct environment's interpreter and, once available, its
   Makefile target** — `make mutation-research` (`.venv`) vs. `make
   mutation-baseline` (`vendor/starcop/.venv`) — matching whichever env's
   test paths cover the module. During Phase 0, use that environment's
   explicit `mutmut` binary because the Make targets do not exist yet.
5. **Extending the gate to a new module is TDD-shaped work**: run mutmut,
   kill survivors with real assertions (not ones that merely dodge that one
   mutant), confirm the module reaches 0 survived/suspicious with any
   exclusions documented, and only then add it to `only_mutate` — in the same
   change, not a follow-up.
6. **`vendor/starcop/` and `coursework/` are never mutated** — they sit
   outside `source_paths` and are additionally listed in `do_not_mutate`,
   matching how both are already excluded from ruff and interrogate. Never
   add either to `source_paths` or `only_mutate`.
7. **A failing gate blocks completion, not a follow-up** — fix it in the same
   change before reporting the task done.

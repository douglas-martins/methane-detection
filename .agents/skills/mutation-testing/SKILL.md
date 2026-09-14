---
name: mutation-testing
description: >
  Run mutmut mutation testing on a module gated in this repo (or being added
  to the gate), triage survivors, and drive every one to kill-or-justify.
  Use whenever asked to run mutation testing, check a mutation score, add
  mutation coverage for a module, or interpret mutmut survivors. Triggers:
  "run mutation testing", "check mutmut", "mutation score", "kill this
  mutant", "add mutation coverage for X".
---

# Mutation Testing (this repo)

The canonical rules live in
[`.agents/rules/mutation-testing.md`](../../rules/mutation-testing.md) —
read it before proceeding if it's not already in context. The rollout's
phase-by-phase scope and status live in `mutation-testing-plan.md` at the
repo root (untracked, local-only — not the source of truth for current
gated scope; `pyproject.toml`'s `[tool.mutmut] only_mutate` is). This file is
the operational procedure for running mutmut and processing its results.

If `[tool.mutmut]` doesn't exist in `pyproject.toml` yet, mutation testing
hasn't been bootstrapped in this repo (see Phase 0 in
`mutation-testing-plan.md`) — stop and say so rather than improvising ad hoc
mutmut usage outside the planned config.

## Procedure

1. **Identify the environment.** Check whether the module's tests run under
   `ENV_RESEARCH_TEST_PATHS` or `ENV_BASELINE_TEST_PATHS` in the `Makefile`
   (some run under both). Once the mutation targets exist, use `make
   mutation-research` (`.venv`) or `make mutation-baseline`
   (`vendor/starcop/.venv`) accordingly. During the Phase 0 bootstrap, use
   the selected environment's explicit `mutmut` binary — never a bare
   `mutmut` command.

2. **Run scoped to the module**, using the CLI's dotted mutant-name pattern
   syntax — **not** file globs:

   ```bash
   .venv/bin/mutmut run 'registry.promotion_criteria.*'
   ```

   File globs (`src/registry/promotion_criteria.py`) are for
   `only_mutate`/`do_not_mutate` in `pyproject.toml`, a different syntax —
   do not mix the two.

3. **Inspect results:**

   ```bash
   .venv/bin/mutmut results
   .venv/bin/mutmut show <mutant-id>   # diff for one mutant
   .venv/bin/mutmut browse             # interactive triage
   ```

4. **Triage each survivor — kill or justify, nothing left unexplained:**
   - **Kill (default):** write or strengthen a test in the module's
     `__tests__/test_*.py`, following the `test-driven-development` rule —
     confirm the new/changed test fails against the mutant (`mutmut show`
     the diff, apply it locally if needed to verify) and passes against the
     unmutated code.
     Watch for tests that pass because they mock away the exact branch the
     mutant lives in — a kill has to exercise the real logic, not the mock.
   - **Justify (only for genuinely equivalent/untestable mutants** — a
     mutated log string, an unreachable defensive branch): add
     `# pragma: no mutate` inline with a one-line reason, or a
     `do_not_mutate` entry in `pyproject.toml` with the same. Record the
     justification in `mutation-testing-plan.md`'s results log if that file
     is present.

5. **Re-run after each fix** and confirm the survivor count actually drops —
   don't assume a new test kills the mutant without re-running.

6. **Once the module reaches 0 survived / 0 suspicious** (all remaining
   exclusions documented), add it to `[tool.mutmut] only_mutate` in
   `pyproject.toml` in the same change — this is what makes the gate binding
   for that module going forward.

7. **Confirm the gate is green once Phase 1 has added it:**

   ```bash
   make mutation-gate
   ```

   During Phase 0, record the smoke-run result instead; there is no gate
   target yet.

8. **Report:** which module and env, mutants run, killed vs. justified
   counts, which test files changed, and the exact commands used — so the
   result is auditable, not just asserted.

## Heavy modules

Modules with torch/lightning/mlflow/rasterio/bentoml/prefect at module scope
(see the HEAVY groups in `mutation-testing-plan.md`) will hang or segfault
under a naive run. Before mutating one directly, confirm these are set in
`[tool.mutmut]` (introduced at Phase 5 of the rollout):

```toml
mutate_only_covered_lines = true
max_stack_depth = 2
process_isolation = "forkserver"
forkserver_warmup = "collect"
timeout_multiplier = 25
```

## Notes

- `vendor/starcop/` and `coursework/` are never mutated — outside
  `source_paths`, and additionally listed in `do_not_mutate`. Never add
  either to `source_paths` or `only_mutate`.
- Don't lower `only_mutate` scope, add throwaway pragmas, or disable the
  heavy-module settings just to get `make mutation-gate` green — see the
  rule file for why.
- `mutants/` (mutmut's working copy) is gitignored and disposable; deleting
  it resets all cached state and forces a full rerun next time.

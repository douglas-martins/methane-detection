# Rule: Test-Driven Development

Always-on constraint for any agent writing Python in this repository — this
applies even when the `test-driven-development` skill hasn't been explicitly
invoked. For the full cycle, patterns, and examples, use the skill at
[`.agents/skills/test-driven-development/SKILL.md`](../skills/test-driven-development/SKILL.md).

1. **Invoke the skill before creating any new Python file** — a module, a
   script, a Hydra-configured stage under `src/data/preprocessing/`, a
   `_vendor_starcop*.py` seam file, anything. This applies with extra force
   under `src/`, where the thesis solution actually lives. Write the failing
   test first, in the module's `__tests__/test_*.py` per this repo's
   convention, before the implementation file exists.
2. **Invoke it before modifying existing behavior** in an existing `.py`
   file — reproduce a bug with a failing test before attempting the fix (the
   Prove-It Pattern), and extend or add a test before changing existing
   logic, not after.
3. **Never skip straight to implementation** for new logic, edge cases, or
   bug fixes. Pure configuration changes (`configs/*.yaml`), documentation,
   and other non-behavioral changes are exempt — this rule is about `.py`
   files with logic in them.
4. **Use the correct environment's interpreter** to run tests — baseline
   (`vendor/starcop/.venv`) vs. research (`.venv`), per `AGENTS.md` — never
   assume a bare `pytest` picks the right one.
5. **Never report a change done** until the fuller suite for the touched
   env passes (`make test-research` / `make test-baseline` / `make test`)
   with no tests skipped or disabled to get there.

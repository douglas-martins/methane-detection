## What changed and why

<!-- One or two sentences. Link an issue with "Closes #123" if there is one. -->

## How was this tested?

<!-- Commands you ran, e.g. `make test-env-b`, `make lint`. Real fixtures, not mocked interactions -- see CONTRIBUTING.md. -->

## Checklist

- [ ] Commits follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) (`feat`, `fix`, `docs`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`) — `commitlint.yml` gates this.
- [ ] Relevant tests pass locally (`make test-env-a` and/or `make test-env-b`).
- [ ] `make lint` is clean.
- [ ] `vendor/starcop/` was not edited (composition-only — see CONTRIBUTING.md's Architecture section).
- [ ] Public docs (`docs/`) updated if this changes reader-facing behavior; `internal-docs/` updated if this changes internal process/setup.

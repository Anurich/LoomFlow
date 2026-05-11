# Contributing to Loom

Thanks for your interest in helping! This guide covers everything
you need to get a change from idea → merged PR. **Read it once
before opening your first PR.** It will save you a review round.

## Table of contents

- [Quick start](#quick-start)
- [Development setup](#development-setup)
- [Running the guardrails locally](#running-the-guardrails-locally)
- [Finding something to work on](#finding-something-to-work-on)
- [Branch and PR workflow](#branch-and-pr-workflow)
- [Commit message style](#commit-message-style)
- [Coding standards](#coding-standards)
- [Tests](#tests)
- [Documentation](#documentation)
- [Release process](#release-process-maintainers-only)
- [Code of conduct](#code-of-conduct)
- [Asking for help](#asking-for-help)

## Quick start

```bash
# Fork the repo on GitHub, then:
git clone git@github.com:YOUR-USERNAME/LoomFlow.git
cd LoomFlow

python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'

# Verify the guardrails are green on a clean checkout:
ruff check loomflow tests
mypy --strict loomflow
pytest -q

# Now you can branch + change + push + open a PR.
```

## Development setup

Loom requires **Python 3.11+**. We test on 3.11 and 3.12 in CI.

```bash
# Editable install with the dev extras (pytest, mypy, ruff, etc.)
pip install -e '.[dev]'

# Optional: extras for features you'll touch
pip install -e '.[anthropic,openai,otel]'
pip install -e '.[postgres,chroma,redis]'   # only if working on those backends
```

The `[dev]` extra pulls in:
- `pytest>=8` + `anyio[trio]>=4.4` + `hypothesis>=6.100`
- `mypy>=1.10` (we run with `--strict`)
- `ruff>=0.4` (lint + format)
- `import-linter>=2.0` (layer-rule enforcement)
- `bump-my-version>=0.30` (release versioning)

## Running the guardrails locally

CI runs three gates on every PR. **Run them locally before
pushing** — saves a review round when they fail:

```bash
# 1. Lint
ruff check loomflow tests examples

# 2. Type-check (strict)
mypy --strict loomflow

# 3. Tests (offline only by default)
pytest -q
```

All three must pass for the PR to be mergeable.

The `live` test mark hits paid OpenAI / Anthropic endpoints and is
**deselected by default**. Run it explicitly only when you've
touched a live integration and have the env vars set:

```bash
OPENAI_API_KEY=sk-... pytest -q -m live
```

Some backend tests skip without env vars (`JEEVES_TEST_PG_DSN`,
`JEEVES_TEST_REDIS_URL`). That's expected — CI runs them in a
matrix job; locally you can ignore the skips unless you're
touching Postgres / Redis code.

## Finding something to work on

- **Browse open issues**: <https://github.com/Anurich/LoomFlow/issues>
- **Look for the `good-first-issue` label**: small, well-scoped,
  ideal for a first contribution.
- **Look for the `help-wanted` label**: maintainers have approved
  the work but don't have bandwidth.
- **Open a new issue first** for non-trivial changes. Lets us
  agree on the design before you sink time in code; protects
  against "great change, wrong direction" rework.

For tiny fixes (typos, one-line bug fixes, doc tweaks), feel free
to just open the PR — no issue needed.

## Branch and PR workflow

1. **Fork** the repo on GitHub.
2. **Clone your fork** and add the upstream as a remote:

   ```bash
   git clone git@github.com:YOUR-USERNAME/LoomFlow.git
   cd LoomFlow
   git remote add upstream git@github.com:Anurich/LoomFlow.git
   ```

3. **Create a feature branch** off the latest `upstream/main`:

   ```bash
   git fetch upstream
   git checkout -b fix/some-bug upstream/main
   ```

4. **Make changes**. Keep the diff focused — one logical change
   per PR. If you find yourself fixing unrelated stuff, open a
   separate PR for it.

5. **Run the guardrails locally** (see above). Make sure all three
   are green.

6. **Commit** with a [Conventional Commits](#commit-message-style)
   message. Squash work-in-progress commits before opening the PR.

7. **Push to your fork**:

   ```bash
   git push origin fix/some-bug
   ```

8. **Open a PR** against `Anurich/LoomFlow:main`. The PR template
   will auto-fill — answer every section honestly.

9. **CI runs automatically**. If it fails, fix the issue and push
   again to the same branch (no need to open a new PR).

10. **Respond to review feedback** by pushing additional commits
    to the same branch. Don't force-push during active review —
    it loses inline comments. The maintainer will squash on merge.

11. **Merged!** The maintainer squash-merges; your name appears
    in the commit log of `main`. The PR closes automatically.

## Commit message style

We use [Conventional Commits](https://www.conventionalcommits.org/)
with optional scope:

```
<type>(<scope>): <short description>

<longer body explaining why, what tradeoffs, and any caveats>
```

**Types** we use:
- `feat` — new feature
- `fix` — bug fix
- `docs` — docs / examples / README changes only
- `test` — test-only changes
- `refactor` — no behaviour change
- `perf` — performance fix
- `chore` — build / CI / tooling

**Scopes** (optional) — usually a module: `agent`, `memory`,
`workflow`, `observability`, `model`, `tools`, `security`,
`architecture`, `skills`.

**Examples** from recent commits:

```
feat(observability): FileTelemetry — JSONL spans + metrics on disk
fix(workflow): router classifier can be async def
docs(examples): add audit log + telemetry walkthroughs
test(loader): skip unstructured-backend PDF tests when dep missing
```

The squash-merge will preserve your subject line as the final
commit message on `main`, so make it informative.

## Coding standards

- **Type hints everywhere.** `mypy --strict` is non-negotiable.
- **Public API has docstrings.** Tier-1 classes (anything
  exported from `loomflow/__init__.py`) need docstrings that
  explain the user-facing contract, not the implementation.
- **No comments that just rephrase the code.** Only write a
  comment when the *why* is non-obvious — a hidden constraint,
  a workaround, a subtle invariant.
- **No dead code.** If a branch is unreachable, delete it.
- **Errors at the boundary, not in the middle.** When a user
  passes a wrong shape (e.g. `audit_log=["value.log"]`), fail at
  construction with a message that names the fix — not deep in
  the runtime with a cryptic Python error. See existing examples
  of this pattern in `loomflow/workflow/__init__.py` and
  `loomflow/agent/api.py`.
- **Backwards-compatible by default.** Don't break existing
  public API in a patch release. If you must break, ship it in
  a minor bump with a migration note in `CHANGELOG.md`.

## Tests

Every change that's not pure docs needs tests:

- **New features** → add tests covering the happy path + at least
  one failure path.
- **Bug fixes** → add a test that fails before your fix and
  passes after. Prevents regression.
- **Refactors** → existing tests should pass unchanged. If you
  need to change tests, that's a behaviour change — call it out
  in the PR description.

Tests live in `tests/` next to the existing files. Pick a similar
existing test as a template — same fixtures, same imports, same
naming.

## Documentation

- **Public API additions** → update the relevant doc page under
  `docs/` if one exists, and the `README.md` if it's a Tier-1
  feature.
- **Breaking changes** → add a note to `CHANGELOG.md` under
  `[next-release] — unreleased`.
- **New examples** → add to `examples/` with a number prefix
  (`14_*`, `15_*`, ...) and update `examples/README.md`.

## Release process (maintainers only)

```bash
# Patch (bug fixes, small additions): 0.9.35 → 0.9.36
make release BUMP=patch

# Minor (new features, backwards-compatible): 0.9.35 → 0.10.0
make release BUMP=minor

# Major (breaking changes): 0.10.0 → 1.0.0
make release BUMP=major
```

`make release` does pre-flight (lint + mypy + pytest), bumps
versions in `pyproject.toml` + `loomflow/__init__.py`, commits,
tags `v<version>`, pushes the commit + tag to `origin`. The
push triggers `.github/workflows/release.yml` which builds the
sdist + wheel and publishes to PyPI via Trusted Publishing.

## Code of conduct

This project adheres to the
[Contributor Covenant 2.1](CODE_OF_CONDUCT.md). By participating,
you're expected to uphold it. Report unacceptable behaviour to
the email in `CODE_OF_CONDUCT.md`.

## Asking for help

- **Chat with us on Discord** for fast back-and-forth, design
  sanity-checks, and meeting other Loom users:
  <https://discord.gg/X6njWztQ>
- **For usage questions** you want as a permanent searchable
  record, open a GitHub Discussion:
  <https://github.com/Anurich/LoomFlow/discussions>
- **For bugs**: open an issue with the bug-report template.
- **For feature ideas**: open an issue with the feature-request
  template. Big proposals welcome — but **discuss first, build
  second.**
- **For security issues**: do *not* open a public issue. Email
  the maintainer per `SECURITY.md`.

Thanks for contributing!

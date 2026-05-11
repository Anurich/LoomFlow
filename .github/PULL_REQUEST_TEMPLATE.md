<!--
Thanks for opening a PR! Please fill in every section. Sections
left as placeholder text are a strong signal the PR isn't ready
for review yet.
-->

## What this PR does

<!-- 1-3 sentences. What does this change, and why? -->

## Linked issue

<!-- "Closes #123" or "Relates to #456". For trivial fixes (typo /
one-liner) you can put "n/a"; everything else should have an
issue we discussed first. -->

Closes #

## Type of change

<!-- Tick whichever applies. Multiple OK. -->

- [ ] `feat`     — new feature
- [ ] `fix`      — bug fix
- [ ] `docs`     — documentation / examples / README only
- [ ] `test`     — test-only changes
- [ ] `refactor` — no behaviour change
- [ ] `perf`     — performance improvement
- [ ] `chore`    — build / CI / tooling

## How was this tested?

<!--
What did you run locally? Paste relevant command output if it
helps reviewers verify quickly. At minimum:

  ruff check loomflow tests examples
  mypy --strict loomflow
  pytest -q

If you ran live integration tests (`-m live`) or backend-specific
tests against a real Postgres / Redis, mention them here.
-->

## Checklist

<!-- Tick what applies. Honest ticks only — reviewers will check. -->

- [ ] I ran the three guardrails locally and they all pass
- [ ] I added tests for the change (or it's docs-only)
- [ ] I updated `CHANGELOG.md` under `[next-release] — unreleased`
- [ ] I updated relevant documentation under `docs/` or the README
- [ ] My commit messages follow the [Conventional Commits style](../CONTRIBUTING.md#commit-message-style)
- [ ] My code follows [the project's coding standards](../CONTRIBUTING.md#coding-standards)

## Breaking changes

<!--
Does this change any public API surface? If yes:
- List what changed.
- Suggest a migration path users should follow.

If no, write "None — fully backwards compatible."
-->

None — fully backwards compatible.

## Anything else reviewers should know?

<!--
Trade-offs you considered, alternatives you rejected, follow-up
PRs you plan. Optional but appreciated for non-trivial changes.
-->

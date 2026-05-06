# JeevesAgent — release helper.
#
# Daily commits / pushes are plain git. This Makefile only adds a
# `release` target — the explicit "publish to PyPI" flag.
#
# Usage:
#   make release BUMP=patch    # X.Y.Z[.devN]  → X.Y.(Z+1)
#   make release BUMP=minor    # X.Y.Z[.devN]  → X.(Y+1).0
#   make release BUMP=major    # X.Y.Z[.devN]  → (X+1).0.0
#   make release BUMP=dev_n    # X.Y.Z.devN    → X.Y.Z.dev(N+1)
#
# Important: standard semver bumps drop the .devN marker. So the
# very first release of an in-development branch (e.g. shipping
# the work currently sitting at 0.3.0.dev0 as 0.3.0) is a manual
# pyproject edit + tag — see `make first-release` below.
#
# What it does:
#   1. Refuses to run with a dirty working tree.
#   2. bump-my-version edits pyproject.toml + jeevesagent/__init__.py
#      atomically, commits, and tags v<new_version>.
#   3. Pushes the commit and the tag to origin.
#   4. The push of the v* tag fires .github/workflows/release.yml,
#      which builds and uploads to PyPI via trusted publishing.
#
# Prerequisites: bump-my-version (in the [dev] extras —
# `pip install -e '.[dev]'`).

.PHONY: release release-help check show-version first-release

release:
	@if [ -z "$(BUMP)" ]; then \
	  echo ""; \
	  echo "  ✗ missing BUMP="; \
	  echo "  usage: make release BUMP=patch|minor|major"; \
	  echo ""; \
	  exit 1; \
	fi
	@if ! command -v bump-my-version >/dev/null 2>&1; then \
	  echo ""; \
	  echo "  ✗ bump-my-version not installed."; \
	  echo "    install with: pip install -e '.[dev]'"; \
	  echo ""; \
	  exit 1; \
	fi
	@echo "→ Pre-flight: working tree clean?"
	@git diff-index --quiet HEAD -- || (echo "  ✗ uncommitted changes; commit or stash first" && exit 1)
	@echo "→ Pre-flight: gates green?"
	@ruff check jeevesagent tests examples
	@mypy --strict jeevesagent
	@pytest -q
	@echo "→ Bumping version ($(BUMP)) + committing + tagging..."
	bump-my-version bump $(BUMP) --verbose
	@echo "→ Pushing commit + tag to origin..."
	git push origin main
	git push origin --tags
	@echo ""
	@echo "  ✓ Released. PyPI workflow firing on the new tag."
	@echo "    Check: https://github.com/Anurich/JeevesHarness/actions"
	@echo ""

# Show what `make release BUMP=...` would do without changing anything.
check:
	@if [ -z "$(BUMP)" ]; then \
	  echo "usage: make check BUMP=patch|minor|major"; \
	  exit 1; \
	fi
	bump-my-version show-bump

show-version:
	@bump-my-version show current_version 2>/dev/null || \
	  grep '^version = ' pyproject.toml

release-help:
	@echo ""
	@echo "  make release BUMP=patch     # X.Y.Z[.devN]  → X.Y.(Z+1)"
	@echo "  make release BUMP=minor     # X.Y.Z[.devN]  → X.(Y+1).0"
	@echo "  make release BUMP=major     # X.Y.Z[.devN]  → (X+1).0.0"
	@echo "  make release BUMP=dev_n     # X.Y.Z.devN    → X.Y.Z.dev(N+1)"
	@echo ""
	@echo "  make check BUMP=...         # dry-run; show new version"
	@echo "  make show-version           # print current version"
	@echo "  make first-release          # ship in-development version as-is"
	@echo "                              # (drops .devN marker; one-time use)"
	@echo ""

# Special-case: drop the .devN marker on the CURRENT version and
# release it as-is. Use this when shipping an in-development branch
# without bumping past the version it intended.
#
# Example: 0.3.0.dev0 (current) → 0.3.0 (released).
#
# After this, normal `make release BUMP=...` works on a clean version.
first-release:
	@if ! command -v bump-my-version >/dev/null 2>&1; then \
	  echo "  ✗ bump-my-version not installed. pip install -e '.[dev]'"; \
	  exit 1; \
	fi
	@git diff-index --quiet HEAD -- || (echo "  ✗ uncommitted changes; commit or stash first" && exit 1)
	@CURRENT=$$(bump-my-version show current_version 2>/dev/null | tail -1); \
	  case "$$CURRENT" in \
	    *.dev*) ;; \
	    *) echo "  ✗ current version $$CURRENT has no .devN marker; use 'make release BUMP=...'"; exit 1 ;; \
	  esac; \
	  FINAL=$$(echo "$$CURRENT" | sed 's/\.dev[0-9]*$$//'); \
	  echo "  → finalizing $$CURRENT → $$FINAL"; \
	  ruff check jeevesagent tests examples && \
	  mypy --strict jeevesagent && \
	  pytest -q && \
	  bump-my-version replace --new-version "$$FINAL" && \
	  git add pyproject.toml jeevesagent/__init__.py && \
	  git commit -m "Release v$$FINAL" && \
	  git tag -a "v$$FINAL" -m "Release v$$FINAL" && \
	  git push origin main && \
	  git push origin "v$$FINAL" && \
	  echo "  ✓ Released v$$FINAL. PyPI workflow firing on the new tag."

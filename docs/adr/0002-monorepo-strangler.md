# 0002 — Monorepo with strangler migration from `legacy/`

- **Status:** Proposed
- **Date:** 2026-09-18

## Context
Three codebases (`Backend/`, `ML/`, `frontend/`) with no shared contracts, no CI and inconsistent ports/env.
Good parts (auth, licence workflow, UI) must be kept; the inference/training layers must be rebuilt.

## Decision
One repository, one `uv` workspace for Python, `apps/` + `packages/` + `ml/` + `infra/` + `docs/` layout.
The previous code moves to `legacy/` via `git mv` (history preserved) and is deleted directory-by-directory as
each capability is ported. New code never imports from `legacy/`.

## Consequences
- Positive: shared tooling, atomic cross-service changes, one CI, contracts next to their producers.
- Negative: a larger repo; `legacy/` is a temptation to copy-paste — mitigated by review.
- Harder: partial deploys of one service require per-app Dockerfiles (Phase 5).

## Alternatives considered
- Separate repos per service — rejected: contracts drift, four CI setups, cross-cutting changes need 3 PRs.
- Rewrite in a fresh repo — rejected: loses history and the good parts' provenance.

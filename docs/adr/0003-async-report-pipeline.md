# 0003 — Report analysis is an asynchronous job

- **Status:** Proposed
- **Date:** 2026-09-18

## Context
Findings BE-3, LLM-7: OCR + six models + two LLM calls run inside one Django request with no timeouts, no
retry and no status. Slow LLM calls block workers; failures are invisible.

## Decision
`POST /reports` stores the file and enqueues a `ReportJob`. A Celery worker (Redis broker) executes the
pipeline `extract -> (human review) -> evaluate_rules -> assess -> explain`, one idempotent task per step, with
per-step timeouts and retries, updating `ReportJob.status`. The web app polls the job (v1) / SSE (v2).

## Consequences
- Positive: horizontal scaling by adding workers; clear failure attribution; inference and LLM services scale
  independently; no long-held HTTP connections.
- Negative: more moving parts (Redis, worker process); eventual consistency in the UI.
- Harder: synchronous "upload and get result" demos — replaced by a progress UI.

## Alternatives considered
- Keep synchronous with timeouts — rejected: still blocks workers and cannot retry.
- Dramatiq / Django-Q / Postgres-backed queue — viable; Celery chosen for tooling and existing Redis dependency
  (open decision D6 may revisit).

# 0006 — Policy functions and audit log for all PHI access

- **Status:** Proposed
- **Date:** 2026-09-18

## Context
Findings SEC-1, SEC-2, SEC-6, SEC-8, LLM-1: permission checks scattered as `if user.role ==` in views, several
missing; medical files served statically; no record of who accessed what.

## Decision
Each Django app has `policies.py` with pure functions `can_<action>(user, obj) -> bool`. Views and services
call a policy before touching PHI. Every allowed PHI read/download/edit writes an `AccessEvent`. Files live in
object storage and are reachable only through short-lived signed URLs issued after a policy check. A generated
role x resource x action matrix test covers every policy.

## Consequences
- Positive: one place to reason about access; auditability required for medical data; regressions caught by
  the matrix test.
- Negative: a little ceremony per endpoint.
- Harder: ad-hoc admin scripts that read PHI without leaving a trace (intended).

## Alternatives considered
- django-guardian / object permissions in DB — heavier than needed; policies can adopt it later without
  changing call sites.

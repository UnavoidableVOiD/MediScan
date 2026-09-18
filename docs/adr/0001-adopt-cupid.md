# 0001 — Adopt CUPID as the guiding philosophy

- **Status:** Proposed
- **Date:** 2026-09-18
- **Deciders:** MediScan team

## Context
The audit (baseline doc section 3) found that almost every serious defect was a *property* failure rather than
a class-design failure: silent fallbacks (`reindex` -> NaN -> median), mixed responsibilities (one function doing
safety + inference + LLM), unobservable behaviour (constant predictions with no error), and code named after
dataset columns instead of the clinical domain. The system is polyglot (Python/Django, Python/FastAPI,
TypeScript/React, YAML rules), so an OO-specific rule set is a poor fit.

## Decision
Every module, service and PR is judged against CUPID: Composable, Unix philosophy, Predictable, Idiomatic,
Domain-based. Each app's README states its one job, its contract and what it never does. The PR template
carries a CUPID checklist. Dependency inversion is used only at genuine swap points (LLM provider, storage,
payment gateway, OCR engine).

## Consequences
- Positive: language-agnostic review criteria; directly targets the failure modes we observed; small pure
  packages (`clinical`) become the centre of gravity.
- Negative: less prescriptive than SOLID — reviewers must exercise judgement; we mitigate with the README
  contracts and the checklist.
- Harder: "quick hacks" that mutate data or swallow errors are rejected even when convenient.

## Alternatives considered
- SOLID — rejected as primary: class-centric, does not name predictability or domain language.
- No stated philosophy — rejected: that is how the current state arose.

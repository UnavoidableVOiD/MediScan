# 0005 — Model bundles with manifests; NOT_ASSESSABLE over imputation

- **Status:** Proposed
- **Date:** 2026-09-18

## Context
Findings ML-1, ML-3, ML-4, ML-6, ML-7: model, scaler and column lists were separate pickles from different
training runs; thresholds diverged between training and serving; missing real-world features were imputed with
medians or hard-coded defaults, yielding constant predictions no one noticed.

## Decision
A deployed model is a versioned bundle directory with `manifest.json` (features as LOINC + unit + required flag,
classes, threshold, metrics, dataset hash, git SHA). The inference service asserts feature-name equality at
load and validates requests at runtime. If a required analyte is missing the result is
`RiskAssessment(status=NOT_ASSESSABLE, missing_analytes=[...])`. Imputation is allowed only for analytes the
manifest marks optional and records the strategy.

## Consequences
- Positive: constant-output bugs become start-up failures or explicit NOT_ASSESSABLE; report tables are
  generated from manifests; honest UI.
- Negative: fewer conditions will be "assessed" on sparse panels; this is the correct behaviour.
- Harder: shipping a model without an evaluation report (the trainer refuses to write a manifest without it).

## Alternatives considered
- Keep loose pickles with a naming convention — rejected: convention is what failed.
- Impute and warn — rejected: warnings are ignored; a fabricated probability is worse than none.

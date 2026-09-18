# 0004 — Observation/Analyte (LOINC) as the clinical domain model

- **Status:** Proposed
- **Date:** 2026-09-18

## Context
Findings ML-2, ML-5, OCR-2: dataset column headers (`sc`, `Total_Protiens`, `TT4`) were used as the data model;
Free T4 was fed to a Total T4 feature; kidney features became NaN through a name mismatch; substring alias
matching produced wrong values.

## Decision
`packages/clinical` defines `Analyte` (LOINC code, canonical unit, printed aliases) and `Observation`
(value, unit, canonical value, printed reference range, source, confidence, provenance). All services
exchange Observations. Unit conversion is an explicit step. Free and Total hormone measurements are distinct
analytes. Dataset column names appear only in `ml/configs/*.yaml` mappings.

## Consequences
- Positive: structural fix for the misalignment and unit bugs; lab's own reference ranges preserved; FHIR
  R4 `Observation` compatibility for HMS integration.
- Negative: upfront registry work (~40 analytes with Nepali lab aliases); stricter — unknown analytes are
  rejected rather than guessed.
- Harder: adding a new lab test requires a registry entry (this is a feature).

## Alternatives considered
- Keep free-form dict keys — rejected: the root cause of three S1 findings.
- Full FHIR server — rejected for v1: heavy; subset gives the benefit.

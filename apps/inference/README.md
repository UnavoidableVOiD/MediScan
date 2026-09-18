# apps/inference — FastAPI

## One job
Given verified `Observation`s, return a `RiskAssessment` per condition from a **versioned model bundle**, or
`NOT_ASSESSABLE` with the reason. Stateless. No OCR, no LLM, no database.

## Model bundle (the contract that makes ML-1..ML-7 impossible)
```
bundles/<condition>/<version>/
  model.joblib
  manifest.json   { condition, model_version, algorithm,
                    features: [{loinc, unit, required}],   # names + units, not dataset column headers
                    classes: [...], threshold, selection_rule,
                    metrics: {cv_mean, cv_std, test}, dataset_sha256, git_sha, trained_at }
  card.md
```
At load time the service asserts `model.feature_names_in_ == [f.loinc for f in manifest.features]` and refuses
to start otherwise. At request time it validates the payload with pydantic against the manifest.

## API
| Route | Purpose |
|---|---|
| `POST /v1/assess` | `{observations: [...], conditions?: [...]}` -> `[RiskAssessment]` |
| `GET /v1/bundles` | loaded bundles, versions, metrics |
| `GET /health`, `GET /ready` | liveness / bundles loaded |
Auth: `INTERNAL_SERVICE_TOKEN` header; bound to the internal network only.

## CUPID contract
- **Composable**: `Predictor.assess(observations) -> RiskAssessment` — one interface for every condition.
- **Unix**: predicts. Does not extract, explain, store or decide policy.
- **Predictable**: missing required analyte -> `NOT_ASSESSABLE`, never a median. Same input, same output.
  Thresholds and class labels come from the manifest only. No `fillna(0)`, no bare `except`.
- **Idiomatic**: FastAPI + pydantic v2, `mypy --strict`, structured logs with `job_id`.
- **Domain-based**: inputs are `Observation`s with LOINC codes; outputs are `RiskAssessment`s with `probability`,
  `label`, `threshold`, `missing_analytes`.

## Mandatory tests
`test_sensitivity.py` (vary one required analyte -> output must change), `test_bundle_contract.py`
(manifest features are producible from a standard panel; feature-name assertion), `test_not_assessable.py`.

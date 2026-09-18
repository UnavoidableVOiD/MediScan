# MediScan architecture (summary)

Full rationale: `MediScan_Engineering_Baseline.pdf` sections 5–8. Decisions: `adr/`.

## Services and boundaries
```
web (React/TS) --HTTPS--> api (Django) --enqueue--> worker (Celery)
                                                      |--HTTP--> inference (FastAPI)   model bundles + manifests
                                                      `--HTTP--> llm_gateway (FastAPI) providers, RAG, validation
api/worker share: PostgreSQL (domain + audit), Redis (queue/cache), S3/MinIO (PDFs, page images, bundles)
packages/clinical: pure domain (analytes, units, observations, ranges, rules) imported by api, worker, inference
```
| Service | One job | Never |
|---|---|---|
| `api` | own business data, humans' HTTP API, policies, audit | run OCR/model/LLM in a request |
| `worker` | run the report pipeline step by step | implement models, prompts, policies |
| `inference` | RiskAssessment from Observations via versioned bundles, or NOT_ASSESSABLE | impute, extract, explain, store |
| `llm_gateway` | provider adapter + grounding + validation + citations | store conversations, predict risk, see PHI identifiers |
| `web` | render and collect input | compute clinical meaning |

## Report pipeline
`QUEUED -> EXTRACTING -> AWAITING_REVIEW -> EVALUATING -> ASSESSING -> EXPLAINING -> DONE | FAILED(step)`

## Domain vocabulary (use these names everywhere)
Analyte (LOINC) · Observation · Suggestion · CriticalFlag · RiskAssessment (ASSESSED | NOT_ASSESSABLE) ·
Explanation · ReportJob · Report · Conversation · Consultation · AvailabilitySlot · Payment · LicenceReview ·
AccessEvent

## CUPID in one line each
Composable — small typed interfaces, no hidden I/O. Unix — one job per service/module. Predictable — fail
loudly, deterministic, observable, NOT_ASSESSABLE over guessing. Idiomatic — the framework's way, typed and
linted. Domain-based — clinical names, never dataset column headers.

## Non-negotiables
No secrets or PHI in git · every PHI read writes an AccessEvent · files only via signed URLs · every model
loaded from a manifest-validated bundle · every HTTP boundary has a committed OpenAPI spec · every policy has a
matrix test.

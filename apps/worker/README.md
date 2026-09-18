# apps/worker — Celery

## One job
Run the report pipeline asynchronously, one step at a time, recording progress on `ReportJob`.

```
QUEUED -> EXTRACTING -> AWAITING_REVIEW -> EVALUATING -> ASSESSING -> EXPLAINING -> DONE
                                                                              \-> FAILED (step, error)
```
| Step | Calls | Produces |
|---|---|---|
| `extract` | OCR engine (Tesseract TSV) + `clinical.analytes` aliases | `Observation`s (source=OCR, bbox, confidence) + `Suggestion`s |
| *(human review happens in the web app; the job waits)* | | `Observation`s (source=MANUAL) |
| `evaluate_rules` | `clinical.rules` | `CriticalFlag`s |
| `assess` | `apps/inference` over HTTP | `RiskAssessment`s (ASSESSED or NOT_ASSESSABLE) |
| `explain` | `apps/llm_gateway` over HTTP | `Explanation`s (validated, with citations) |

## CUPID contract
- **Composable**: each step is a function `(job_id) -> None` with one input type and one output type; steps
  are chained by Celery, not by importing each other.
- **Unix**: the worker orchestrates; it does not implement models, prompts or policies.
- **Predictable**: idempotent steps (re-running a step overwrites its own outputs only); per-step timeout and
  retry; a failure marks the job FAILED with the step name and a structured error — never a silent partial result.
- **Idiomatic**: Celery tasks with explicit `bind`, `acks_late`, `max_retries`; shared Django ORM via `apps/api`.
- **Domain-based**: task names are domain verbs: `extract_observations`, `evaluate_rules`, `assess_risk`,
  `explain_results`.

# apps/llm_gateway — FastAPI

## One job
Own every call to a language model: choose the provider, ground the prompt, validate the output, cite sources.

## Components
| Component | Responsibility |
|---|---|
| `providers/` | `OllamaProvider`, `GroqProvider`, `AnthropicProvider` behind one `complete()` / `stream()` interface |
| `prompts/<name>/<version>.md` | versioned prompt templates; the version is recorded on every `Explanation` |
| `rag/` | corpus manifest (source, version, licence), chunking, embeddings, retrieval returning `(text, doc, page)` |
| `validators/` | numeric consistency (every number in output exists in input or retrieved text); scope guard |
| `deidentify.py` | strips name, ID, DOB, contact before anything leaves the process |

## API
| Route | Purpose |
|---|---|
| `POST /v1/explain` | `{audience, observations, flags, assessments}` -> `{text, citations, validated, prompt_version, provider}` |
| `POST /v1/chat` (SSE) | `{conversation, observations?, assessments?}` -> streamed tokens + final citations |
| `GET /health`, `GET /ready` | liveness / provider reachable, index loaded |

## CUPID contract
- **Composable**: providers are interchangeable by config; the rest of the service does not know which is active.
- **Unix**: generates and validates text. Does not store conversations (api does), does not predict risk.
- **Predictable**: temperature and prompt version are fixed per route; an output that fails validation is
  regenerated once, then redacted — never returned silently. PHI never leaves un-de-identified.
- **Idiomatic**: FastAPI, pydantic, async httpx, SSE via `StreamingResponse`.
- **Domain-based**: prompts speak in `Observation` / `CriticalFlag` / `RiskAssessment` terms.

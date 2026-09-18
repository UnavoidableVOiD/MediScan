# apps/api — Django

## One job
Own the business data and the HTTP API for humans: who the users are, which reports exist, who may see what,
consultations, payments, administration and the audit trail. It **orchestrates nothing clinical itself** — it
enqueues a `ReportJob` and reads the results the worker wrote.

## Domain apps (one Django app per bounded context)
| App | Owns | Never does |
|---|---|---|
| `identity` | users, roles, OTP, JWT cookies, OAuth, doctor licence review | business rules about reports |
| `reports` | `Report`, `ReportJob`, `Observation`, `Suggestion`, `CriticalFlag`, `RiskAssessment`, `Explanation` | run OCR, models or LLM calls in a request |
| `consultation` | `AvailabilitySlot`, `Consultation`, doctor-patient episodes, comments | touch payment gateways |
| `billing` | `Payment`, gateway adapters (Khalti), refunds, revenue split | change consultation status except via events |
| `administration` | admin dashboards, licence approval, user management | bypass policies |
| `audit` | `AccessEvent` — every read/download/edit of PHI | anything else |

## Layout inside every app
```
<app>/
  models.py        domain entities (names from docs/architecture.md, nothing else)
  services.py      use cases; the only place that writes to models
  policies.py      can_<action>(user, obj) -> bool; the only place that decides permission
  serializers.py   API shapes (generated into packages/contracts/openapi/api.yaml)
  views.py         thin: parse -> policy -> service -> respond
  tests/           test_policies.py is mandatory when policies.py changes
```

## CUPID contract
- **Composable**: views call one service function; services return domain objects, not `Response`s.
- **Unix**: no OCR, model or LLM code here — that is `worker`, `inference`, `llm_gateway`.
- **Predictable**: every PHI access passes a policy and writes an `AccessEvent`; no `print`; no bare `except`;
  files are served only via signed URLs from object storage.
- **Idiomatic**: Django apps, managers, migrations; DRF viewsets kept thin; settings from environment via
  `pydantic-settings`; `structlog` JSON logging with `request_id`.
- **Domain-based**: model and field names come from the domain vocabulary. No `final_data` JSON blobs.

## Migrated from `legacy/Backend`
Keep: OTP flow, JWT cookie auth, Google login, licence verification state machine, specialist routing,
revenue-split rules. Rewrite: `reports`, `chatbot`. Drop: `is_premium`, duplicate Khalti v1 flow.

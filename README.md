# MediScan v2

**Medical report translator and triage platform.** A patient uploads a laboratory report (PDF or photo);
MediScan extracts the values, lets the patient verify them, checks for critical values, assesses disease
risk with versioned models, explains the result in plain language with cited clinical guidelines, and
connects the patient with a verified doctor for a paid consultation.

> **Version note.** This is **v2**, a ground-up re-architecture. The previous implementation (v1, the
> final-year project) lives read-only in [`legacy/`](legacy/) and is being ported into this structure
> capability by capability. If you are looking for the code described in the original project report,
> see [Previous version](#previous-version-v1).

| | |
|---|---|
| Base document | [`docs/MediScan_Engineering_Baseline.pdf`](docs/MediScan_Engineering_Baseline.pdf) — audit of v1, architecture, standards, roadmap, open decisions |
| Architecture summary | [`docs/architecture.md`](docs/architecture.md) |
| Decisions | [`docs/adr/`](docs/adr/) |
| Philosophy | [CUPID](docs/adr/0001-adopt-cupid.md) — Composable, Unix, Predictable, Idiomatic, Domain-based |
| Status | Phase 1 (foundations). Not deployable yet. See [Roadmap](#roadmap). |

---

## What MediScan does

```
upload PDF/photo ──▶ extract values ──▶ patient verifies ──▶ critical-value rules ──▶ risk assessment
                     (OCR + analyte      (review screen,     (deterministic,        (versioned model
                      registry)           suggestions)        guideline-sourced)     bundles, or
                                                                                      NOT_ASSESSABLE)
        ──▶ plain-language explanation ──▶ doctor booking & payment ──▶ doctor comments on the report
            (LLM, grounded in guidelines,   (verified doctors,           (patient sees both)
             every number validated)         Khalti, revenue split)
```

Conditions in scope: anemia, chronic kidney disease, liver dysfunction, thyroid dysfunction, glycaemic
risk (diabetes), and — pending decision D4 — heart disease. Target users: patients in Nepal with low health
literacy, and the doctors who review their results.

---

## Architecture at a glance

```
web (React + TypeScript)
   │  HTTPS · client generated from OpenAPI
api (Django + DRF) ── identity · reports · consultation · billing · administration · audit
   │  enqueue ReportJob
worker (Celery) ──── extract → (review) → evaluate_rules → assess → explain
   ├──▶ inference (FastAPI)    model bundles + manifests → RiskAssessment | NOT_ASSESSABLE
   └──▶ llm_gateway (FastAPI)  provider adapter · RAG with citations · output validation

shared: PostgreSQL · Redis · S3/MinIO          packages/clinical: pure domain (analytes, units, rules)
```

Each service has **one job** and a README that states what it never does — that boundary is the review
criterion for every PR. Full rationale in the baseline document, sections 5–8.

---

## Repository layout

```
apps/
  api/            Django — business data, policies, audit, HTTP API for humans
  worker/         Celery — the report pipeline, one idempotent task per step
  inference/      FastAPI — risk assessment from versioned model bundles
  llm_gateway/    FastAPI — every LLM call: providers, prompts, RAG, validation
  web/            React + TypeScript — renders what the API says, computes nothing clinical
packages/
  clinical/       pure Python domain: Analyte (LOINC), units, Observation, reference ranges, rules
  contracts/      committed OpenAPI specs for every HTTP boundary (generated, CI-checked)
ml/               training pipelines, evaluation, OCR golden set → produces bundles; never runs in prod
infra/            docker-compose (dev), production compose, ops scripts
docs/             baseline PDF, architecture summary, ADRs
legacy/           v1 — read-only reference during migration (see below)
```

---

## Getting started (developers)

Prerequisites: [uv](https://docs.astral.sh/uv/), Node 22, Docker.

```bash
git clone https://github.com/UnavoidableVOiD/MediScan.git && cd MediScan
cp .env.example .env         # fill in local secrets; .env is never committed
make setup                   # uv workspace (Python 3.12) + pre-commit hooks
make infra-up                # Postgres 16, Redis 7, MinIO, Ollama
make test                    # unit tests
make help                    # all targets: lint, typecheck, contracts, doc, ...
```

Per-service instructions live in each `apps/*/README.md`. Application services run from source in dev;
only infrastructure runs in Docker.

---

## How we work

- **CUPID** over SOLID — every module is judged on being Composable, Unix-like, Predictable, Idiomatic and
  Domain-based. The PR template carries the checklist.
- **Contracts first** — HTTP boundaries have committed OpenAPI specs; the web client is generated; CI fails
  on drift.
- **Fail loudly** — `NOT_ASSESSABLE` instead of imputed guesses; typed errors instead of `None`; no bare
  `except`, no `print` (enforced by ruff).
- **Nothing sensitive in git** — no secrets, no patient documents, no model artefacts or datasets. Enforced
  by gitleaks and a large-file guard in pre-commit and CI. Anything that was ever committed is treated as
  leaked and rotated.
- **Every PHI access is audited and policy-checked** — `policies.py` per app, `AccessEvent` per read.
- **Decisions are written down** — a change to a boundary, contract, domain model or non-negotiable needs
  an [ADR](docs/adr/).

---

## Roadmap

| Phase | Outcome | Status |
|---|---|---|
| 0 — Stop the bleeding (on v1) | keys rotated, PHI purged from history, authz holes closed, broken models disabled | in progress |
| 1 — Foundations | monorepo, CI, contracts, `packages/clinical`, Postgres/MinIO, ADRs | **current** |
| 2 — Clinical core | Observation model, extraction v2, rule engine, model bundles, async pipeline, review + result UI | |
| 3 — Explanation & chat | LLM gateway, grounded + validated explanations, persistent conversations | |
| 4 — Consultation & billing | consultation state machine, single Khalti flow, audit log, notifications | |
| 5 — Scale & ship | load tests, backups, single-VPS deploy, runbook, privacy policy | |

Open decisions (D1–D11) are listed in the baseline document, section 12, and tracked in `docs/adr/README.md`.

---

## Previous version (v1)

v1 was built as a Bachelor of Engineering (IT) major project at Pokhara University (2025–26) and is
described in the original project report. Its code is preserved unchanged under [`legacy/`](legacy/):

| Path | v1 component | Fate in v2 |
|---|---|---|
| `legacy/Backend/` | Django API: auth (OTP, JWT cookies, Google), reports, doctors, appointments, Khalti, admin | auth, licence flow, consultation and revenue rules are being **ported**; reports and chatbot **rewritten** |
| `legacy/ML/` | FastAPI: Tesseract OCR, six disease models, SafetyGuard, RAG chatbot | OCR aliases, safety thresholds and training approach **ported** into `packages/clinical` and `ml/`; inference service **rewritten** (see baseline findings ML-1…ML-10) |
| `legacy/frontend/` | React (JavaScript) UI | screens **ported** one by one onto TypeScript and the generated API client |

v1 is not maintained and should not be deployed: the baseline audit found broken model inference,
authorization gaps and unredacted documents in the repository history. New code never imports from
`legacy/`. Directories are deleted from `legacy/` as their capabilities land in `apps/`.

If you need to run v1 for reference, follow `legacy/Backend/README.md` and `legacy/ML/README.md`.

---

## Team

Aashish Sharma · Prabhat Acharya · Pratik Chapagain · Supreme Badal — supervised by Er. Himal Acharya
(Pokhara University). Ownership per area is in [`CODEOWNERS`](CODEOWNERS).

## Licence

To be decided (open item). Until a licence file exists, all rights reserved by the authors.

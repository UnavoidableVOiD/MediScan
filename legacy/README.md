# MediScan v1 (legacy) — read-only reference

This directory holds the original implementation of MediScan, built as a BE (IT) major project at Pokhara
University (2025–26) and described in the project's final report. It was moved here unchanged when the v2
re-architecture began (see [`../README.md`](../README.md) and ADR 0002).

| Directory | Contents | How to run (reference only) |
|---|---|---|
| `Backend/` | Django 4.2 + DRF API — authentication (OTP, JWT cookies, Google), reports/OCR orchestration, doctors, appointments, Khalti, admin panel, chatbot proxy | `Backend/README.md` |
| `ML/` | FastAPI AI service — Tesseract OCR, six disease-risk models, SafetyGuard, RAG chatbot (Ollama/Groq), Dockerfile | `ML/README.md` |
| `frontend/` | React 19 + Vite + Redux (JavaScript) UI | `cd frontend && npm install && npm run dev` |

## Status

- **Not maintained. Do not deploy.** The v2 baseline audit (`../docs/MediScan_Engineering_Baseline.pdf`,
  section 3) documents broken model inference, a safety layer that can mask critical values, authorization
  gaps and other issues.
- **Do not import from here** in v2 code. Copy, adapt to the v2 domain model and contracts, add tests, then
  delete the v1 directory once its capability has landed in `../apps/`.
- Secrets and patient documents that were ever in this tree are treated as leaked; keys have to be rotated
  and history purged (baseline findings SEC-4, SEC-5, REP-1).

## Port map (what moves where)

| v1 | v2 destination |
|---|---|
| `Backend/authentication`, `admin_panel` | `apps/api/identity`, `apps/api/administration` |
| `Backend/doctor` (availability, appointments, comments, revenue split) | `apps/api/consultation`, `apps/api/billing` |
| `Backend/reports` (two-step OCR review) | `apps/api/reports` on the `Observation` model |
| `ML/src/ocr_engine` (aliases, preprocessing) | `packages/clinical/analytes.py`, `apps/worker` extract step |
| `ML/src/models/safety.py` (thresholds) | `packages/clinical/rules/*.yaml` |
| `ML/src/training/*` (ensembles, SMOTE, thresholds) | `ml/pipelines/train.py`, `ml/configs/` |
| `ML/src/chatbot`, `generation` (prompts, RAG, providers) | `apps/llm_gateway` |
| `ML/Dockerfile` | template for all v2 service images |
| `frontend/src` | `apps/web` (TypeScript, generated client) |

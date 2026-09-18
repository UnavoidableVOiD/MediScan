# contracts

Committed OpenAPI specifications for every HTTP boundary. Generated, never hand-edited.

| Spec | Producer | Consumers |
|---|---|---|
| `openapi/api.yaml` | `apps/api` (drf-spectacular) | `apps/web` (generated TS client) |
| `openapi/inference.yaml` | `apps/inference` (FastAPI) | `apps/worker` |
| `openapi/llm.yaml` | `apps/llm_gateway` (FastAPI) | `apps/worker`, `apps/api` (chat) |

CI regenerates each spec and fails the build if it differs from the committed file (`make contracts`).
A breaking change to a spec requires an ADR.

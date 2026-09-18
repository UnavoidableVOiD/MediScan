# apps/web — React + TypeScript (Vite)

## One job

Let patients, doctors and admins see and act on their data. It renders what the API says; it never computes
clinical meaning (no thresholds, no risk maths, no unit conversion in the browser).

## Structure

```
src/
  api/          generated from packages/contracts/openapi/api.yaml (openapi-typescript). Never hand-edit.
  features/     one folder per domain: reports/ review/ results/ consultation/ billing/ identity/ admin/
  components/   shared, dumb UI
  app/          router, providers (TanStack Query, auth), layout
```

## Key screens (Phase 2)

- **Review**: page image with bbox highlights, one row per `Observation`, `Suggestion`s to accept/reject,
  confidence colouring.
- **Result**: `CriticalFlag` banner first -> per-condition cards (`ASSESSED` with probability, or
  `NOT_ASSESSABLE` with what is missing) -> `Explanation` -> doctor comment. No fabricated confidence bar.
- **Job progress**: step-wise status from `ReportJob`.

## CUPID contract

- **Composable**: features depend on `api/` types, not on each other.
- **Unix**: display and input only.
- **Predictable**: every server state (loading/error/empty/not-assessable) has a rendering; no optimistic
  clinical results.
- **Idiomatic**: TypeScript strict, TanStack Query for server state, React Router, feature folders.
- **Domain-based**: component names mirror the domain (`ObservationRow`, `CriticalFlagBanner`, `RiskCard`).

## Migration from `legacy/frontend`

Keep the visual design and auth/booking screens; port feature by feature onto the generated client. Scaffold
with `npm create vite@latest . -- --template react-ts` when Phase 1 starts.

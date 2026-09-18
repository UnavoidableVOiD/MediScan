## What
<!-- one paragraph; link the finding ID (e.g. ML-2) or ADR if relevant -->

## CUPID check
- [ ] **Composable** — new code exposes small typed interfaces; no hidden I/O or globals
- [ ] **Unix** — this change stays inside one service/module's stated job (see its README)
- [ ] **Predictable** — errors are raised or typed, never swallowed; no `print`; deterministic
- [ ] **Idiomatic** — passes ruff/mypy/tsc; follows the framework's conventions
- [ ] **Domain-based** — names come from `docs/architecture.md` vocabulary

## Safety
- [ ] No secrets, PHI, model artefacts or datasets in the diff
- [ ] If permissions changed: `policies.py` + matrix test updated
- [ ] If an HTTP boundary changed: OpenAPI regenerated and committed

## Tests
<!-- what was added; include at least one negative case -->

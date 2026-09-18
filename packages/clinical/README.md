# clinical

The heart of MediScan. **Pure Python, zero framework dependencies** (pydantic only). Imported by `apps/api`,
`apps/worker` and `apps/inference`. If a concept about lab data has to be true everywhere, it lives here and
nowhere else.

## One job
Represent laboratory observations correctly: which analyte, what value, in what unit, against which reference
range, from which source — and evaluate deterministic rules over them.

## Modules (Phase 2 fills these in)
| Module | Responsibility |
|---|---|
| `analytes.py` | `Analyte` registry keyed by LOINC code: canonical unit, aliases as printed by Nepali labs |
| `units.py` | Unit parsing and conversion to canonical units (mg/dL <-> mmol/L, g/dL <-> g/L, ...) |
| `observation.py` | `Observation` value object; `source`, `ocr_confidence`, provenance |
| `ranges.py` | Reference-range parsing (`"0.4 - 1.4"`, `"< 140"`, `"4000-11000"`) and in/out-of-range evaluation |
| `rules/` | Rule engine: YAML rule catalogue -> `CriticalFlag`s. Pure; never mutates input |

## CUPID contract
- **Composable**: functions take and return plain value objects. No I/O, no logging, no globals, no env vars.
- **Unix**: each module does one thing; the package does not know what a report, a model or a user is.
- **Predictable**: unknown analyte, unparseable unit or malformed range -> a typed error, never `None` or a guess.
- **Idiomatic**: pydantic models, `Decimal` for values, `Enum` for closed sets, full type hints, `mypy --strict`.
- **Domain-based**: names come from the lab domain (`Analyte`, `Observation`, `ReferenceRange`), never from
  dataset column headers (`Total_Protiens`, `sc`, `hemo`).

## Testing
`uv run pytest packages/clinical` — every public function has a positive and a negative test; conversions have
round-trip tests; rules have one passing and one failing case each.

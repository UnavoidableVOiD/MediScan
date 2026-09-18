# ml — training, evaluation, golden set

Training is **separate from serving**. Nothing here runs in production; it produces versioned bundles that
`apps/inference` loads.

| Dir | Purpose | Rules |
|---|---|---|
| `data/` | public datasets, hash-pinned (`data/manifest.json`). Never PHI. | gitignored; `make data` downloads and verifies hashes |
| `configs/` | one YAML per condition: dataset, feature -> LOINC mapping, class mapping, target metric, selection rule | the only place a dataset column header may appear |
| `pipelines/` | `train.py --config configs/<condition>.yaml` -> `dist/<condition>/<version>/` bundle | deterministic (seeded); writes manifest + card + eval artefacts |
| `evaluation/` | stratified 5-fold CV, calibration, threshold selection, `report_tables.py` for the academic report | one metric definition per task type |
| `golden_set/` | OCR benchmark: `pdfs/` (private/consented, gitignored) + `truth/*.json` (hand-transcribed, committed) | truth is transcribed by one person and verified by a second; never copied from OCR output |

## Bundle manifest schema
Defined once in `apps/inference/src/bundles/manifest.py` (pydantic) and imported here — the trainer cannot emit
a manifest the server cannot load.

## Conditions and v1 stance (see baseline doc section 10.4)
| Condition | v1 |
|---|---|
| Anemia | ML + WHO sex-specific rule shown side by side |
| CKD | retrain on panel analytes only + deterministic eGFR (CKD-EPI 2021) |
| Liver | keep ensemble, honest probability |
| Thyroid | deliberate 3-class mapping; requires TSH + matching T4 analyte |
| Diabetes | rules-first (fasting glucose, HbA1c); ML secondary — decision D4 |
| Heart | drop or retrain on panel analytes — decision D4 |

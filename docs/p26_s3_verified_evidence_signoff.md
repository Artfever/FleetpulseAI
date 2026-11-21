# P26-S3 verified evidence and open sign-off items

Issue: [P26-S3 #237](https://github.com/Artfever/AI_PROJECT/issues/237)

This is a bounded verification record. “Verified” below means checked against
the local tracked artifacts and commands listed here. It does not imply that a
teammate approved the result, that the final ZIP was tested, or that the current
hosted app revision was identified.

## Evidence verified in this workspace

| Area | Check and observed result | Evidence |
| --- | --- | --- |
| Frozen model loading | Five saved bundles loaded in separate fresh processes against the same FD001/train engine 4 prefix through cycle 164; no fit occurred. Raw outputs were 29.79411765, 28.02408951, 26.14615822, 107.80786196, and 17.24176598 cycles for Decision Tree, Random Forest, XGBoost, constant baseline, and Dense MLP. | `scripts/smoke_p12_final_bundles_clean_process.py`; [`P26-A1 inventory`](p26_a1_runnable_package.md) |
| Final test rows and metrics | 500 rows (five frozen versions × 100 engines); independent metric audit recomputed all rows and all ten model metrics with zero MAE/RMSE delta. No tuning was performed. | `artifacts/final/p13/final_test_results.json`; `scripts/audit_p13_metrics.py`; [`P13 result report`](p13_s3_final_test_results.md) |
| Report, slide, and UI agreement | Five versions and values agree. Local Model Laboratory AppTest had no page exception and its five rows matched the saved JSON/loader exactly. Both slide tables now list explicit model versions. | `docs/p26_s2_claim_consistency.md`; `app/pages/5_Model_Laboratory.py`; `src/fleetpulse/evaluation/final_test_display.py` |
| Diagnostic figures and studies | P22 audit checked 60 claims, three SVG figures, and 24 local links. A1/A2 scheduler comparisons are saved in tracked JSON with declared assumptions. | `scripts/audit_p22_report_results.py`; `docs/p22_report_results.md`; `docs/p22_a1_fixed_fleet_scenarios.json`; `docs/p22_a2_seed_comparison.json` |
| Data and feature provenance | Manifests specify the source/split and selected configuration; the feature contract is `past-only-v1`, 30 cycles, 96 features. FD001 is simulated data. Raw train/test/RUL files are not tracked; source and SHA-256 instructions are documented. | `artifacts/data_quality.json`; `artifacts/split_manifest.json`; `artifacts/selection_manifest.json`; `docs/fd001_provenance.md`; `docs/artifact_inventory.md` |
| Rebuild consistency limitation | P24 recorded a small saved-MLP reproduction drift in one environment comparison; a repeat in the current environment was exact. The cause of the tiny drift is unproven and was not retuned away. | `docs/p24_s2_frozen_evaluation.md`; `docs/p24_gate_verification.md` |

## Commands run

```powershell
.venv/Scripts/python.exe scripts/smoke_p12_final_bundles_clean_process.py
.venv/Scripts/python.exe scripts/audit_p13_metrics.py
.venv/Scripts/python.exe scripts/audit_p22_report_results.py
.venv/Scripts/python.exe -m pytest -q tests/test_model_laboratory_page.py tests/test_final_test_display.py
.venv/Scripts/python.exe -m ruff check .
```

Observed results: five-model fresh-process load passed; P13 metrics passed with
zero deltas; P22 audit passed (60 claims, 3 SVGs, 24 links); focused UI/loader
tests passed (8); Ruff passed. The actual local AppTest comparison also passed
for all five visible rows. These checks do not establish physical fault
diagnosis, maintenance suitability, future-engine accuracy, or official
reproduction on another machine.

## Items deliberately left unsigned

- The candidate ZIP was audited and run from a fresh extraction for P26-A3.
  `docs/p26_a3_archive_run.md` records the extraction and checks; P26-T3 records
  the exact SHA-256 and package path. The final phase gate still needs real
  three-member review/sign-off of that exact archive.
- The deployed Streamlit Cloud SHA is not independently identified in this
  workspace. Local AppTest is not remote deployment evidence.
- No teammate approval or three-person archive sign-off is claimed. Those
  remain human checks for the final phase gate after the nine task deliverables
  and exact archive are ready for review.

The evidence listed above can be relied on within its stated scope. The open
items must be resolved or explicitly accepted before closing P26 gate #31.

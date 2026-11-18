# P13-T3: Report-ready official test CSV exports

Model Laboratory provides two downloads from the checksum-verified P13 final result report:

- `p13_official_test_summary.csv` — one row for each of the five frozen model versions, with FD001/test partition, engine sample count, MAE/RMSE cycle units, and source/result hashes.
- `p13_official_test_per_engine.csv` — one row per model and engine (500 rows total), with engine ID, cutoff, answer line, true RUL, raw prediction, residual, absolute/squared error, model version, and provenance hashes.

The builder validates the sealed report, exact model inventory, all 100 engine IDs per model, answer-line mapping, every per-engine error field, and whether the per-engine rows reproduce the reported MAE/RMSE. It rejects missing values, duplicates, mismatched errors, incomplete engine sets, or inconsistent metrics. It never inserts placeholders or fills gaps. Both app downloads are hidden if the report is missing or fails verification.

## Actual local export

Command: `.venv/Scripts/python.exe -m fleetpulse.evaluation.final_test_export --output-directory artifacts/final/p13/exports`.

| Local ignored file | Rows | Blank cells | SHA-256 |
| --- | ---: | ---: | --- |
| `artifacts/final/p13/exports/p13_official_test_summary.csv` | 5 | 0 | `16f5b9cd9efa09dd7c015ea73da633c0c059474a21509c3b21c2e83808b6efa6` |
| `artifacts/final/p13/exports/p13_official_test_per_engine.csv` | 500 | 0 | `7a165046be15fb6f943c83e444bf887e9118c3f42d6d4f174f995800b4c6926b` |

The files preserve the five observed MAE/RMSE scores and all 500 saved engine rows from the P13 report. Values were not recomputed into placeholders or rounded to display precision. The source report payload is `2a0686eaa3d369c435938cf6bf6256b047b3265f8ac8893e08e040adc798a6a6`; the test-history, RUL-answer and pairing-payload hashes are included in each export.

The CSV files and source report are Git-ignored under `artifacts/final`. A hosted checkout without the local report therefore displays the explicit unavailable state and no download buttons; public deployment of these exports is not claimed.

## Verification

- `.venv/Scripts/python.exe -m pytest tests/test_final_test_export.py tests/test_final_test_display.py tests/test_model_laboratory_page.py -q` — 14 passed, including missing-value, inconsistent-error, and summary/per-engine mismatch rejection.
- Actual export command — 5 summary rows and 500 per-engine rows; zero blank cells; file hashes above.
- Local Model Laboratory AppTest — showed both download buttons and the five-row official test table.
- `.venv/Scripts/ruff.exe check .` — passed.
- `git diff --check` — passed.

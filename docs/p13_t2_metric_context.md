# P13-T2: Units, samples, model versions, and score meaning

The official FD001/test table in Model Laboratory now shows model name, frozen model version, sample count in engines, MAE in cycles, and RMSE in cycles. The checksum-verified saved report is still the only source; the page does not recompute scores.

## Local verified output

The pinned report contained five models and exactly 100 engine rows per model.

| Model | Version | Engines | MAE (cycles) | RMSE (cycles) |
| --- | --- | ---: | ---: | ---: |
| Decision Tree | `decision-tree-rul-v1` | 100 | 24.513306424988755 | 33.95219459227001 |
| Random Forest | `random-forest-rul-v1` | 100 | 20.34970822712025 | 27.51996165541398 |
| XGBoost | `xgboost-rul-v1` | 100 | 18.44893268585205 | 24.16644813405696 |
| Constant mean baseline | `constant-rul-v1` | 100 | 40.87930202123019 | 52.62485752612688 |
| Dense MLP | `dense-rul-mlp-final-v1` | 100 | 18.160531787872316 | 25.573537464442705 |

The page defines MAE as mean absolute error and RMSE as the square root of mean squared error. Both express error magnitudes in operating cycles; lower values mean smaller errors on these 100 test engines. Neither is an accuracy percentage, and neither guarantees results for future engines. These held-out scores were used for final reporting, not model selection, tuning, or fitting.

The source file `artifacts/final/p13/final_test_results.json` is Git-ignored. On hosted clones without that artifact, the table remains unavailable with the P13-T1 explanation; it never fabricates, fills, or substitutes values.

## Verification

- `.venv/Scripts/python.exe -m pytest tests/test_final_test_display.py tests/test_model_laboratory_page.py -q` — 8 passed.
- `.venv/Scripts/python.exe -m pytest -q` — 510 passed.
- `.venv/Scripts/ruff.exe check .` — passed.
- `git diff --check` — passed.
- Actual local AppTest loaded the verified report and showed all five versions, 100-engine sample counts, cycle units, and interpretation text.

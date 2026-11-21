# P26-S2 report, slides, and app consistency check

Issue: [P26-S2 #183](https://github.com/Artfever/AI_PROJECT/issues/183)

## Source of truth

The saved P13 result envelope is
`artifacts/final/p13/final_test_results.json` (payload SHA-256
`2a0686eaa3d369c435938cf6bf6256b047b3265f8ac8893e08e040adc798a6a6`). It
contains five frozen model versions and 100 per-engine rows per model. The UI
loader in `src/fleetpulse/evaluation/final_test_display.py` validates its
envelope, FD001/test source hashes, model/version inventory, and row counts.
The Model Laboratory page at `app/pages/5_Model_Laboratory.py` reads that
loader; it does not hard-code the results.

## Values checked across artifacts

| Model | Version | MAE (cycles) | RMSE (cycles) | Result |
| --- | --- | ---: | ---: | --- |
| Decision Tree | `decision-tree-rul-v1` | 24.513306424988755 | 33.95219459227001 | Match |
| Random Forest | `random-forest-rul-v1` | 20.34970822712025 | 27.51996165541398 | Match |
| XGBoost | `xgboost-rul-v1` | 18.44893268585205 | 24.16644813405696 | Match |
| Constant mean baseline | `constant-rul-v1` | 40.87930202123019 | 52.62485752612688 | Match |
| Dense RUL MLP | `dense-rul-mlp-final-v1` | 18.160531787872316 | 25.573537464442705 | Match |

The exact values match `docs/report/p25_s1_model_evaluation_scheduling.md` and
the saved final JSON. Both slide tables
(`docs/slides/p25_ai_algorithms_results.md` and
`docs/slides/p25_final_deck.md`) show all five versions and the same metrics
rounded to four decimal places. The summary claims also match the saved table:
Dense MLP has the lowest MAE; XGBoost has the lowest RMSE. The final result is
for 100 simulated FD001/test engines and is an error measure in cycles, not an
accuracy percentage or a guarantee for future engines.

## UI check

The actual local Model Laboratory page was executed with Streamlit `AppTest`
against the tracked report. It rendered one five-row official-results table
without page exceptions. Each displayed model name, version, 100-engine count,
MAE, and RMSE exactly matched `load_final_test_display()` and the saved JSON.
The focused existing UI and loader checks also passed:

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_model_laboratory_page.py tests/test_final_test_display.py
```

Result: 8 passed in 6.82 seconds. The report/slides/app audit also checked the
five versions and metrics programmatically; all assertions passed. No app code
changed, so a separate Streamlit server smoke was not needed for this task.

## Correction and follow-up

The previous slide tables showed names and metrics but omitted model version
strings. Added explicit version columns to both slide tables so readers can
identify the exact frozen artifacts being compared. P26-T2 must re-export and
inspect the PDF after this change. This audit covers local source and saved
artifacts; it does not independently identify the currently deployed cloud SHA.

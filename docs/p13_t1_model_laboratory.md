# P13-T1: Separate development and official test results

The Model Laboratory now labels development validation as `FD001/train` and gives the final `FD001/test` results a separate section. Development validation and selection are used during model work; the official test section is reporting only and is loaded from the checksum-verified P13 result report. The page does not calculate new scores or load final model weights.

The reader rejects a bad report checksum, wrong dataset/source hashes, incomplete model set, non-100-engine result tables, and invalid metrics. If the ignored local report is missing (as it is in a fresh public source checkout), the app says official test results are unavailable instead of showing development scores in their place.

## Actual local display

The local ignored report loaded successfully and populated five official rows in Model Laboratory:

- Decision Tree: MAE 24.513306424988755 cycles; RMSE 33.95219459227001 cycles.
- Random Forest: MAE 20.34970822712025 cycles; RMSE 27.51996165541398 cycles.
- XGBoost: MAE 18.44893268585205 cycles; RMSE 24.16644813405696 cycles.
- Constant mean baseline: MAE 40.87930202123019 cycles; RMSE 52.62485752612688 cycles.
- Dense MLP: MAE 18.160531787872316 cycles; RMSE 25.573537464442705 cycles.

The page labels these as official held-out test error measures in cycles, says they were not used for fitting, selection, or tuning, and explicitly says they are not accuracy percentages. These values are from the saved P13 final report, not from a new evaluation in the page.

The report path is `artifacts/final/p13/final_test_results.json`, which is Git-ignored. Therefore a public Streamlit clone without secure artifact provisioning shows the explicit unavailable state. P13-T3's export path can provide a safe, traceable way to make approved summary results available to the hosted app; this task does not claim the public deployment displays final scores.

## Verification

- `.venv/Scripts/python.exe -m pytest tests/test_final_test_display.py tests/test_model_laboratory_page.py -q` — 8 passed, including valid official display and missing-report behavior.
- `.venv/Scripts/python.exe -m pytest -q` — 510 passed.
- `.venv/Scripts/ruff.exe check .` — passed.
- `git diff --check` — passed.
- Actual AppTest with local artifacts — passed; official table displayed five model rows and the actual saved P13 metrics.
- `.venv/Scripts/streamlit.exe run app/Home.py --server.headless true --server.port 8502 --server.address 127.0.0.1`; `curl.exe -i http://127.0.0.1:8502/_stcore/health` — HTTP 200, body `ok`.

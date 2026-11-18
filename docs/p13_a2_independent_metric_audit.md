# P13-A2: Independent final score recomputation

This audit independently joins saved P13-S1 prediction rows to the pinned FD001/test RUL vector by engine ID, recalculates MAE and RMSE, and compares every engine row and aggregate with the P13-S3 result report. It does not import the P13 prediction, pairing, or scoring implementation and does not load or refit models.

## Inputs and method

Run `.venv/Scripts/python.exe scripts/audit_p13_metrics.py` with the local ignored FD001 inputs and P13 artifacts present. The script reads `artifacts/final/p13/official_test_predictions.json`, `data/raw/RUL_FD001.txt`, `data/raw/test_FD001.txt`, and `artifacts/final/p13/final_test_results.json`. The available saved prediction artifact is JSON rather than CSV; this audit uses those actual per-engine prediction rows without converting or rewriting the source. It verifies both report checksums and the pinned source hashes, maps answer line *i* to sorted `unit_id` *i*, joins each model's predictions by ID, checks the 100 cutoffs/labels/raw predictions against all reported rows, and independently computes raw MAE/RMSE.

The independent audit is a separate implementation in `scripts/audit_p13_metrics.py`; no functions from `src/fleetpulse/evaluation/` are called. It requires exactly 100 engines and the five saved final model versions, and rejects missing/duplicate identities, non-finite values, changed input hashes, count errors, or score differences greater than `1e-12` cycles. Reported test scores remain descriptive and were not used to select, tune, or refit any model.

## Observed result

All five model tables contained 100 unique engine predictions. All 500 engine rows matched the reported raw prediction, answer line, observed cutoff, and label. The independently recalculated aggregates matched the report exactly at full stored precision; each MAE and RMSE delta was `0.0` cycles.

| Model version | Independent MAE (cycles) | Reported MAE | Independent RMSE (cycles) | Reported RMSE |
| --- | ---: | ---: | ---: | ---: |
| `decision-tree-rul-v1` | 24.513306424988755 | 24.513306424988755 | 33.95219459227001 | 33.95219459227001 |
| `random-forest-rul-v1` | 20.34970822712025 | 20.34970822712025 | 27.51996165541398 | 27.51996165541398 |
| `xgboost-rul-v1` | 18.44893268585205 | 18.44893268585205 | 24.16644813405696 | 24.16644813405696 |
| `constant-rul-v1` | 40.87930202123019 | 40.87930202123019 | 52.62485752612688 | 52.62485752612688 |
| `dense-rul-mlp-final-v1` | 18.160531787872316 | 18.160531787872316 | 25.573537464442705 | 25.573537464442705 |

This is an independent arithmetic and saved-row consistency audit, not a separate model inference run. Input hashes: test histories `3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851`; RUL vector `a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca`; prediction payload `34a92a60d2a6af6d279f929210c29d52ef43172c2d2a7122bb249d35538ffc2d`; reported result payload `2a0686eaa3d369c435938cf6bf6256b047b3265f8ac8893e08e040adc798a6a6`.

## Local output and verification

The generated JSON is ignored by Git: `artifacts/final/p13/a2_independent_metric_audit.json`. Its payload SHA-256 is `efb9683e8af70abe368242ba499b2b5c032eb932b08076ca7b9b6a80385abac8`; its whole-file SHA-256 is `b391728543a2dfbaaf3a8b1736d60f2449fb45da5d8d78a19e8889a1b72aa713`.

- `.venv/Scripts/python.exe -m pytest tests/test_p13_metric_audit.py -q` — 4 passed, including deliberately shuffled IDs `(30, 10, 20)`, a reported-score mismatch, answer-count mismatch, and source-hash mismatch.
- `.venv/Scripts/ruff.exe check scripts/audit_p13_metrics.py tests/test_p13_metric_audit.py` — passed.
- `.venv/Scripts/python.exe scripts/audit_p13_metrics.py` — passed on the pinned actual files; 500 per-engine rows verified, all ten aggregate values matched with zero deltas.

The original RUL vector contains no engine IDs. The sorted-ID convention and its archive provenance limitation are documented in [P13-A1](p13_a1_label_alignment_audit.md). No teammate inspection or approval is claimed here.

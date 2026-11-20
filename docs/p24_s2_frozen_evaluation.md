# P24-S2 frozen final evaluation recheck

Task: [P24-S2 #177](https://github.com/Artfever/FleetpulseAI/issues/177)

## Method and safeguards

Reran inference and scoring for the five checksum-verified P12 final bundles on the pinned simulated `FD001/test` histories. The evaluator loads existing model files; it does not fit or tune a model. Predictions, pairings, result JSON and figures were written to a new directory under `%TEMP%`, so the repository's existing P13 report and ignored source artifacts were not overwritten.

The test history SHA-256 was `3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851`; the separate RUL-answer file SHA-256 was `a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca`. The output reports `test_results_used_for_tuning: false`. All estimates and metrics below use operating cycles.

## Commands

Run from the repository root in PowerShell:

```powershell
$s2Dir = Join-Path $env:TEMP 'fleetpulse-p24-s2-rerun-20261121'
.\.venv\Scripts\python.exe -m fleetpulse.evaluation.test --test-data data/raw/test_FD001.txt --output (Join-Path $s2Dir 'official_test_predictions.json')
.\.venv\Scripts\python.exe -m fleetpulse.evaluation.official_test_pairing --predictions (Join-Path $s2Dir 'official_test_predictions.json') --rul-answers data/raw/RUL_FD001.txt --output (Join-Path $s2Dir 'official_test_pairings.json')
.\.venv\Scripts\python.exe -m fleetpulse.evaluation.final_test_results --pairings (Join-Path $s2Dir 'official_test_pairings.json') --output (Join-Path $s2Dir 'final_test_results.json') --residual-figure (Join-Path $s2Dir 'test_residual_histogram.html') --prediction-figure (Join-Path $s2Dir 'test_prediction_errors.html')
.\.venv\Scripts\python.exe scripts/audit_p13_metrics.py --predictions (Join-Path $s2Dir 'official_test_predictions.json') --reported (Join-Path $s2Dir 'final_test_results.json') --output (Join-Path $s2Dir 'independent_metric_audit.json')
.\.venv\Scripts\python.exe scripts/audit_p13_bundles.py --output (Join-Path $s2Dir 'bundle_identity_audit.json')
```

## Observed comparison

The reloaded frozen bundle identities, test history, engine IDs and cutoffs matched. Decision Tree, Random Forest, XGBoost and the constant baseline reproduced every saved prediction and both aggregate scores exactly. The final MLP's weights and manifest also matched their frozen hashes, but 43 of its 100 raw predictions differed slightly from the previous P13 output. The maximum absolute per-engine change was `0.0000152587890625` cycles; mean absolute change was `0.00000335693359375` cycles.

| Final bundle version | Saved MAE | Rerun MAE | Difference | Saved RMSE | Rerun RMSE | Difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `decision-tree-rul-v1` | 24.513306424988755 | 24.513306424988755 | 0 | 33.95219459227001 | 33.95219459227001 | 0 |
| `random-forest-rul-v1` | 20.34970822712025 | 20.34970822712025 | 0 | 27.51996165541398 | 27.51996165541398 | 0 |
| `xgboost-rul-v1` | 18.44893268585205 | 18.44893268585205 | 0 | 24.16644813405696 | 24.16644813405696 | 0 |
| `constant-rul-v1` | 40.87930202123019 | 40.87930202123019 | 0 | 52.62485752612688 | 52.62485752612688 | 0 |
| `dense-rul-mlp-final-v1` | 18.160531787872316 | 18.16053291320801 | +0.000001125335693075158 | 25.573537464442705 | 25.573538664123753 | +0.0000011996810478365205 |

The independent audit checked all 500 prediction rows against the rerun result and the sorted-ID RUL pairing; it recomputed all ten aggregate metrics with zero delta against the rerun JSON. The P13 bundle audit checked 10 files and confirmed all five registered final model identities and artifact hashes match the frozen inventory. A second prediction-only run on the same environment produced the exact same payload hash and all 500 same values as the first rerun.

The rerun prediction payload SHA-256 is `077fd47ec4c588fc0d4697583df37670b1fab1d5ac5f331c181ad5ce9ea59a36`; the pairing payload SHA-256 is `27ac8ae6ba954b9801a873f90516d834bef507c0315cb030502c6808018dc8f6`; and the rerun result payload SHA-256 is `24a69ceee7dac33730a5cf9468e5296a7a7948b9bfa2245064e2c4edb65ce6b9`. Their raw file hashes and temporary output location are recorded in the handoff. The existing tracked P13 result and ignored prediction/pairing files were left unchanged.

## Runtime and interpretation

Current runtime: Python 3.12.16, NumPy 2.5.4, scikit-learn 1.9.1, joblib 1.6.0, and PyTorch 2.15.0+cpu. The final MLP manifest records Python 3.12.10 and PyTorch 2.15.0+cpu. Since the saved MLP weights, scaler, source histories, and feature identities match, the small MLP output difference is recorded as an unresolved numerical reproducibility difference; its cause was not established. The rerun is stable within the current environment. No model or evaluation code was changed to force a match, and no tuning was performed.

These results are final reporting on a simulated held-out test set, not a new model selection step or real-fleet evidence. The MLP difference is small in cycles but should remain visible in the audit trail rather than being rounded away. P23 hosted checks remain owner-waived/unverified; this local evaluation does not establish public deployment.

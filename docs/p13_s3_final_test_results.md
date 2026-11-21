# P13-S3: Final FD001/test scores and residual figures

This report summarizes the frozen P12 model outputs on the original FD001/test histories. Each model has one raw prediction at each of the 100 engines' last observed cycles. The test answer vector was paired by sorted engine ID in P13-S2; every result row carries engine ID, observed cycle, answer line, true RUL, raw prediction, residual, absolute error, and squared error.

## Metric results

Residual is `raw_prediction_cycles - true_rul_cycles`. MAE is the mean absolute residual; RMSE is the square root of mean squared residual. Predictions were not clipped. All values are in operating cycles.

| Final model | Version | Engines | MAE | RMSE |
| --- | --- | ---: | ---: | ---: |
| Decision Tree | `decision-tree-rul-v1` | 100 | 24.51330642 | 33.95219459 |
| Random Forest | `random-forest-rul-v1` | 100 | 20.34970823 | 27.51996166 |
| XGBoost | `xgboost-rul-v1` | 100 | 18.44893269 | 24.16644813 |
| Constant mean baseline | `constant-rul-v1` | 100 | 40.87930202 | 52.62485753 |
| Dense MLP | `dense-rul-mlp-final-v1` | 100 | 18.16053179 | 25.57353746 |

The MLP has the lowest MAE; XGBoost has the lowest RMSE. These are descriptive held-out test results, not a new selection step. They were not used to choose, tune, or refit any model. P13-A2 #143 independently recalculates these aggregate metrics from the saved per-engine rows.

## Input and model provenance

- Test histories SHA-256: `3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851`.
- Separate RUL answer file SHA-256: `a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca`.
- Paired input payload SHA-256: `2f5a05e3015602405857950824d71399f8b9352d1ab7012129208304b4f41730`.
- Common ordered 96-feature `past-only-v1` config SHA-256: `19dd0331ee3e0c3a90217446a4481507b7da94fe1424b1c251c4605a6b168dfe`.
- The runtime reloaded each bundle through its checksum-verifying P12 loader before attributing results to it.

| Model | Manifest payload SHA-256 | Weight SHA-256 |
| --- | --- | --- |
| Decision Tree | `ed434d11ce7ae22cd6445aa18fc4fb7e5db5f3438dded681b596ef54d6d37e89` | `c33021cb9cf5931aba545b6a327d0547f14be1363cbd0c24d57f0ad923dd7bb3` |
| Random Forest | `f52ff4dbcb9c3876ff045bebfaef791652d996a151e987815cf37a6afed69b1a` | `712a8313271565091fb3a7c9b4fce5e5d8bf2caf64b77e8da9c942a477605d81` |
| XGBoost | `e40f4a9142caf5dc6d8c4880a2b0e526ef0e9e097e472ab70d44aa79169c2e11` | `719a5553ba938fb40dd4196d081d7d07f6c7490be0dbbd4d0cf653e776a4816e` |
| Constant mean baseline | `dff249bdfef9426b731a4d8fb8f74bd298571d38e86eb277f0a2a8b27d724c96` | No weight file; baseline statistic is stored in its manifest |
| Dense MLP | `7a3e56ff80df0ba194067956842dab1e9d4148f85f5dc1ad9378bca0b234549c` | `fcbd5a67ca8bf307f049c7131fee08ee484f8f35e6cb4e8966aa3c45839b4650` |

## Reproduction artifacts

Run `.venv/Scripts/python.exe -m fleetpulse.evaluation.final_test_results` after P13-S1 and P13-S2 artifacts and the local verified P12 bundles exist. It writes an immutable JSON report with all five 100-row error tables and the model/source/config hashes, plus two standalone Plotly HTML figures:

- `artifacts/final/p13/final_test_results.json` — payload SHA-256 `2a0686eaa3d369c435938cf6bf6256b047b3265f8ac8893e08e040adc798a6a6`; current tracked whole-file SHA-256 `12f63b73f763cac0b70d133a3b1dd63c06422adf81e4900698383c0037b7925f`.
- `artifacts/final/p13/test_residual_histogram.html` — self-contained residual distribution plot for all five models; SHA-256 `d2732d92c1279c31672b594a126ff03f997676cbfa184126cced7508b493ff2f`.
- `artifacts/final/p13/test_prediction_errors.html` — residual vs. true RUL scatter with engine IDs on hover; SHA-256 `cfe8dc2bf193bdea388aa95bfe5b76e340c6328040a3822756026d7aa944584b`.

The verified result JSON is tracked and contains all five 100-row prediction/error tables; standalone HTML plots and intermediate prediction/pairing files remain local and are not included. Tracked P22 SVGs provide the packaged official-test diagnostic figures. The tracked generator and this report preserve exact commands, observed metrics, checksums, and the evidence boundary; a fresh clone needs the original data and locally provisioned P12 bundles to reproduce final scores. The repository's [FD001 summary](fd001_summary.md) also records a historical incidental first-line exposure during archive inspection; it states that no answer values were used for training/model selection. This P13 step is the explicit final scoring use and does not feed back into model choices.

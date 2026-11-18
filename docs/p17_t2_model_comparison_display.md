# P17-T2: frozen model comparison and raw disagreement display

The RUL Prediction page has an explicit **Compare frozen final models** action.
It passes the selected engine's observed prefix through the current cutoff to
the P12 final-model comparison service. Successful models appear in one table
with model/version and unclipped raw RUL estimates in cycles. Failed or
unavailable models appear in a separate table with their reason and do not
enter the range. If every model fails, the page reports that no spread is
available.

The spread is `max(raw RUL) - min(raw RUL)` across successful models, shown in
cycles and labelled **model disagreement**. The page states that this is not a
confidence interval or prediction uncertainty. No model is retrained and no
display clipping is applied to comparison values.

## Local saved-bundle output

The real UI audit command is:

```powershell
.venv/Scripts/python.exe scripts/check_p17_t2_model_spread_display.py
```

It selects the packaged FD001/train engine 4 prefix through cycle 164, invokes
the actual RUL page and captures the five comparison rows and spread metric.
The ignored full UI payload is `reports/P17_T2_model_spread_display.json`.
The same history has a shared raw feature configuration and model lineage as
documented in `docs/p17_final_model_comparison.md`.

Observed model outputs in cycles:

| Final model | Version | Raw RUL |
| --- | --- | ---: |
| Decision Tree | `decision-tree-rul-v1` | 29.794118 |
| Random Forest | `random-forest-rul-v1` | 28.024090 |
| XGBoost | `xgboost-rul-v1` | 26.146158 |
| Constant mean | `constant-rul-v1` | 107.807862 |
| Dense MLP | `dense-rul-mlp-final-v1` | 17.241766 |

Raw minimum is 17.241766 cycles (Dense MLP), maximum is 107.807862 cycles
(constant mean), and spread is 90.566096 cycles. These are individual outputs
and model-to-model disagreement on one packaged training prefix. They do not
show accuracy, probability, calibrated confidence, or fleet-wide behavior.
The real frozen bundles are ignored local artifacts, not committed files.

## Synthetic UI coverage

`tests/test_prediction_page.py` verifies that the selected observed prefix is
used, a negative raw output remains unclipped, raw range is displayed in cycle
units, a failed model is listed separately, and all-model failure has no
fabricated spread. The full frozen-service schema/integrity tests are in
`tests/test_final_comparison.py`.

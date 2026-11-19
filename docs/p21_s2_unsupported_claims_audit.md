# P21-S2: unsupported claims audit

Audit date: 2026-11-19. Scope: visible copy, numeric outputs, and downloads in
Home, Fleet Overview, Engine Explorer, RUL Prediction, Maintenance Planner, and
Model Laboratory. The requirement is to remove unsupported failure
probabilities, health percentages, fault diagnoses, and savings claims.

## Findings

No affirmative failure probability, health percentage, fault diagnosis, or
financial savings claim was found in the six screens. Existing matches for
`fault` and `diagnosis` explicitly deny diagnostic meaning. The visible `%`
matches are Plotly hover-format placeholders and wording that says MAE/RMSE are
not accuracy percentages. Planner numeric inputs and outputs are identified as
scenario assumptions or cycle-based scheduling measures. Since the current
copy already makes these limits clear, this task changes no application code
and does not remove valid metrics.

The source scan was:

```powershell
rg -n -i "probab|health|fault|diagnos|savings|cost|downtime|%|accuracy percentage" app --glob '*.py'
```

## Screen-by-screen result

| Screen | User-visible claims and evidence | Unsupported-claim result |
| --- | --- | --- |
| Home | Describes simulated turbofan planning and says verified model/evaluation artifacts appear only when installed. | No health score, probability, diagnosis, or savings promise. |
| Fleet Overview | Shows model/version, observed cycle, fit membership, raw/display RUL in cycles, and a review threshold. Its copy says estimates do not confirm faults or maintenance needs; the threshold is explicitly not an alarm or rule. | The threshold is a display filter, not an implied failure probability or service recommendation. |
| Engine Explorer | Sensor summaries describe observed ranges and reversals. The page says these do not identify a physical fault. Training RUL explicitly uses the complete engine's future failure cycle and is not a model prediction/input. | No sensor-based fault diagnosis or health percentage. The future-derived label is confined to analysis. |
| RUL Prediction | Displays cycle estimates with model/version and membership. Synthetic input is marked artificial and unvalidated. SHAP values are relative model attribution, not causation; raw model spread is called disagreement, not confidence. | No probability, health percentage, fault label, or maintenance diagnosis. |
| Maintenance Planner | Shows completion, due cycle, tardiness, weighted tardiness, makespan, and soft-horizon overflow. It says synthetic RUL/duration/priority/team/buffer/horizon values are assumptions; selected-fleet durations/weights are not FD001 measurements. | These are schedule measures under stated assumptions, not cost/savings estimates or reliability claims. |
| Model Laboratory | Labels development selection, validation, and held-out FD001/test separately. MAE/RMSE and MSE units are cycles/cycles-squared; captions say these are errors, not accuracy percentages or future guarantees. | No unsupported percentage, probability, diagnosis, or savings claim. |

## Traced numerical outputs

### P11 development model selection

The app loads `artifacts/selection_manifest.json` with schema
`p11-selection-manifest-v1`. The local artifact file SHA-256 is
`7496b3dc535879d5d97fa3214d99474753cd8f90458faf2f85109cdbd5403e15`. The
loader verifies the payload hash, `FD001/train` partition, 20 engine-prefix
sample unit/count, source/split hashes, six candidate rows, and owner-recorded
selection decision. These values were used during selection; they are not an
untouched performance estimate.

| Candidate | MAE (cycles) | RMSE (cycles) |
| --- | ---: | ---: |
| Mean constant baseline | 65.728002 | 74.258987 |
| Decision Tree, 30-cycle window | 22.494270 | 29.449574 |
| Random Forest, depth 16 | 18.360242 | 24.563262 |
| XGBoost, depth 4 | 15.108697 | 21.092113 |
| MLP, hidden (64, 32) | 15.511258 | 22.937993 |
| MLP, hidden (128, 64) | 15.080604 | 22.066152 |

The manifest records selected candidate XGBoost depth 4, model
`xgboost-rul-v1`, and matched checkpoint SHA
`20801e897816f879c0f93f56dc199c49735c3afc3a544b595247f5c62d419612`.
The report is visible only when the verified local manifest is present.

### P13 held-out FD001/test report

The Model Laboratory loads `artifacts/final/p13/final_test_results.json` using
the `p13-final-test-results-v1` loader. Local file SHA-256:
`db420935dc14221c50f71f0ca3349b0667bea6b37df45b330026df3d3e8d40d4`. The
loader verifies the canonical payload checksum, FD001/test partition, intended
final-report-only use, pinned original test-history and RUL-answer hashes, five
expected model versions, and 100 per-engine rows per model. The CSV export
builder consumes this same verified report and refuses inconsistent fields.

| Model version | MAE (cycles) | RMSE (cycles) |
| --- | ---: | ---: |
| `decision-tree-rul-v1` | 24.513306 | 33.952195 |
| `random-forest-rul-v1` | 20.349708 | 27.519962 |
| `xgboost-rul-v1` | 18.448933 | 24.166449 |
| `constant-rul-v1` | 40.879302 | 52.624858 |
| `dense-rul-mlp-final-v1` | 18.160532 | 25.573537 |

Independent local P13 audits passed: `scripts/audit_p13_label_alignment.py`
checked 100 test engines, 100 answer rows, and 500 model/engine rows;
`scripts/audit_p13_metrics.py` recomputed all five MAE/RMSE pairs with zero
delta from the saved report. The report artifacts are local/ignored and are
not represented as deployed in every checkout.

### RUL explanation and disagreement examples

Existing reproducible UI audits passed on packaged simulated FD001/train
engine 4 through cycle 164. `scripts/check_p17_t1_contribution_display.py`
confirmed signed SHAP contributions in cycles; its report and source/model
provenance are documented in `docs/p17_t1_feature_contribution_display.md`.
`scripts/check_p17_t2_model_spread_display.py` confirmed five raw model
estimates and a spread of 90.566096 cycles; `docs/p17_t2_model_comparison_display.md`
records model versions and says this one-prefix spread is disagreement, not
accuracy, probability, confidence, or fleet-wide evidence. These are model
outputs for one training prefix, not diagnosis or savings evidence.

### Planner examples

`scripts/check_p20_s3_paired_artifact.py` passed against the saved development
Decision Tree `decision-tree-rul-v1` and packaged FD001/train prefixes. The
three raw estimates were 34.196078 cycles (engine 4/cycle 164), 158.16 (engine
5/cycle 58), and 184.357143 (engine 12/cycle 7). The paired comparison used
one team, duration/weight 1 per engine, buffer 2, and soft horizon 6; the
artifact records zero weighted tardiness for both methods under these
illustrative assumptions. Source, model, and scenario hashes are in the output
and `docs/p20_s3_paired_comparison_artifact.md`. This is a wiring example, not
maintenance evidence, schedule-superiority evidence, or savings.

## Verification and review status

Local saved-artifact commands run for this audit:

- `.venv/Scripts/python.exe scripts/check_p17_t1_contribution_display.py` PASS.
- `.venv/Scripts/python.exe scripts/check_p17_t2_model_spread_display.py` PASS.
- `.venv/Scripts/python.exe scripts/check_p20_s3_paired_artifact.py` PASS; artifact round-trip PASS.
- `.venv/Scripts/python.exe scripts/audit_p13_label_alignment.py` PASS.
- `.venv/Scripts/python.exe scripts/audit_p13_metrics.py` PASS; all five metrics independently reproduced with zero delta.
- Direct calls to `load_selection_manifest` and `load_final_test_display` PASS for the local P11/P13 reports above.

The local audit artifacts are under ignored `reports/` or `artifacts/`; this
tracked note records the exact outputs and source paths without adding raw
datasets or model bundles. No application code or data was changed. The issue's
named teammate review of scientific wording, references, and current audit PR
remains required before closure.

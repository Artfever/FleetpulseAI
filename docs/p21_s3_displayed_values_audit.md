# P21-S3: trace displayed values to real outputs

Audit date: 2026-11-19. Task: P21-S3, issue #224. Audit baseline: `dev`
commit `473d13c1d91e7eef633c34b35f523689909573a3`. The audit checked Home
and all five Streamlit pages against the functions, source files, manifests,
and saved reports that supply their visible values.

## Finding

No displayed numeric result was found without a source. Results come from one
of four paths: validated FD001 records, a checksum-verified saved metric
report, a saved model called by the prediction service with an observed
history prefix, or a visibly labeled scheduling simulation using explicit
assumptions. When optional local artifacts are absent or invalid, the app
shows an unavailable/error state instead of substituting invented numbers.

The local audit artifacts and raw FD001 data are ignored by Git. Their hashes
below identify this workspace's evidence; they are not claims that these files
are bundled in a clean checkout or deployed remotely.

## Source inventory

| Source | Local SHA-256 | How the app uses it |
| --- | --- | --- |
| `data/raw/train_FD001.txt` (20,631 rows, 100 engines; valid, 0 quality warnings) | `963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8` | Home and Engine Explorer observed records; FD001/train local history and source for displayed training-only ground truth. |
| `data/demo/fd001_training_prefixes.txt` (229 rows; engines 4/5/12 ending at 164/58/7) | `953489b722a33815ee08f360c1a677d314f318b7d74d8bef88cd31dde0a8f9a7` | The small parity/demo source for Fleet Overview, feature preview, and saved-model examples. |
| `artifacts/selection_manifest.json` (`p11-selection-manifest-v1`) | `7496b3dc535879d5d97fa3214d99474753cd8f90458faf2f85109cdbd5403e15` | Model Laboratory P11 candidate comparison; loader verifies the payload, partition, sample count, source/split provenance, and six candidate entries. |
| `artifacts/development/decision_tree_validation.json` (`decision-tree-rul-v1`) | `41f16e1397e1e2780133fd8a6aae8e8990d4af260055111b595c9831c8cb13b3` | Development Decision Tree prediction manifest used by prediction and fleet services; the model and manifest are validated before use. |
| `artifacts/split_manifest.json` | `47eeae52d9099035f1db2780f2e11e8ec7718eec807aa248d11141dd67072c78` | Fit/validation membership and cutoff provenance; unknown membership remains unknown when verification is unavailable. |
| `artifacts/final/p13/final_test_results.json` (`p13-final-test-results-v1`) | `db420935dc14221c50f71f0ca3349b0667bea6b37df45b330026df3d3e8d40d4` | Model Laboratory held-out test table and CSV exports; loader verifies the canonical payload, pinned FD001/test sources, five versions, and 100 rows per model. |

The local raw training file loaded as 20,631 valid records across 100 engines
with no quality warnings. The demo loader separately checks engine IDs,
contiguous cycle prefixes, and the expected 229 rows.

## Screen-by-screen trace

| Screen and displayed result | Source and verification | Traceability result |
| --- | --- | --- |
| **Home:** record/engine counts and first 10 rows | `fleetpulse.ui.render_training_data` loads `data/raw/train_FD001.txt`, runs `validate_engine_history`, then passes the actual record/engine counts and file rows to `render_dataset_preview`. | Direct record inspection: 20,631 rows; 100 engines; valid; zero warnings. If missing/invalid, the page shows a warning/error and no count table. |
| **Fleet Overview:** engine ID, observed cycle, fit membership, displayed/raw RUL, model/version, feature rows, warnings/errors | Loads the validated local or fixed packaged FD001/train rows, verifies the saved Decision Tree manifest, calls `predict_fleet`, joins results by dataset-qualified engine identity, and only then builds the visible rows. `artifacts/split_manifest.json` supplies membership. | Saved-model examples below reproduce the page output. Missing models or invalid rows remain errors with no estimate; membership is not inferred when the split manifest is absent. |
| **Engine Explorer:** observed sensor history/plots, sensor range and reversal summaries, training RUL, ordered feature preview | Observed values come from `load_engine_history`; summaries use `summarize_sensor_history`; training RUL uses `calculate_training_rul` on a complete FD001/train lifetime and is labeled analysis-only; preview calls `build_engine_feature_preview` with the selected prefix and saved feature configuration. | P21-A2's real feature audit verified engine 4 through cycle 164, 30 observed rows, 96 `past-only-v1` features, and source hash above. It reproduced sensor_1 latest/mean/std/slope `518.67/518.67/0/0` and sensor_4 `1411.81/1411.7286667/4.3979455/0.2859310`. No future rows are passed to the preview. |
| **RUL Prediction:** visible history through cutoff, raw/display estimate, model/version, warning list; optional SHAP contributions and five-model disagreement | Local histories are truncated at the chosen cycle; upload uses validated CSV records; the prediction service receives that history and a checksum-verified model manifest. Display-only clipping is separate from the raw estimate. SHAP and comparison tables load the corresponding frozen final model bundles and explain/display service outputs. | P17 and P21 actual UI examples below pass. Synthetic input is explicitly identified as artificial; missing bundles show unavailable messages. Attribution is relative to a reference, not cause; model spread is disagreement, not confidence. |
| **Maintenance Planner:** due/completion cycles, tardiness, weighted tardiness, makespan, overflow, schedules and timelines | Restores one selected-fleet snapshot with engine/model provenance or uses the labeled Appendix B synthetic scenario; `build_planning_scenario` validates estimates and explicit duration/weight/team/buffer/horizon assumptions; EDD and GA run on the same `SchedulingScenario`; UI reads the paired comparison result. | P20-S3's saved artifact round-trip below verifies the real prediction-to-planner boundary. Durations, weights and limits are called simulation assumptions; overflow is visible and not silently scored or dropped. |
| **Model Laboratory:** P11 selection scores, P06 development validation values/plots, P13 held-out test metrics and exports | P11 is loaded by `load_selection_manifest`; development validation is loaded with matching feature datasets, manifests, model bundle, and checksums; P13 is loaded by `load_final_test_display`, and CSV export is built from the same verified report. | P11/P13 source hashes and metric checks below pass. Missing or incompatible optional files produce unavailable messages rather than fallback numbers. |

## Reproduced numeric examples

### Saved model through feature preview and planner

`python scripts/check_p21_a2_feature_preview.py` reproduced the preview for
`FD001/train` engine 4 at cycle 164 from the 30 observed rows in the fixed
parity subset. It emitted 96 ordered `past-only-v1` features. This is feature
inspection, not fault diagnosis or a prediction.

`python scripts/check_p20_s3_paired_artifact.py` loaded the saved development
Decision Tree (`decision-tree-rul-v1`; model SHA-256
`2a693e6ecc09de5d9549e287a6798fdb339c7e945c700231444f0c5bcd425018`) and
reproduced these live service outputs:

| Dataset/engine/cutoff | Raw RUL (cycles) | Input feature rows | Warning |
| --- | ---: | ---: | --- |
| FD001/train / 4 / 164 | 34.1960784314 | 30 | none |
| FD001/train / 5 / 58 | 158.16 | 30 | none |
| FD001/train / 12 / 7 | 184.3571428571 | 7 | short window |

The paired planner artifact has schema `p20-paired-schedule-comparison-v1`,
scenario SHA-256 `3c58edea62bf88da3eb79b66e14779273e523a77cd7cc84b2f58b7560ac93049`,
and an artifact round-trip PASS. EDD and GA each produced 0 weighted tardiness
cycles. That is a schedule output for one illustrative team, duration/weight
1, buffer 2 and soft horizon 6; it is not a model-quality or maintenance
result.

### Explanation and model disagreement shown on RUL Prediction

The local UI check `scripts/check_p17_t1_contribution_display.py` verified the
final Decision Tree at `FD001/train` engine 4/cycle 164: raw RUL 29.7941176471
cycles, SHAP reference 126.9360964740 cycles, `sensor_13_mean` +6.5855725670
cycles, and `sensor_4_mean` -29.4441008377 cycles. The contributions sum to
the prediction within the reported additivity tolerance. They describe model
attribution for this prefix, not a physical cause or fault.

`scripts/check_p17_t2_model_spread_display.py` verified five raw predictions
on the same prefix: Decision Tree 29.7941176471, Random Forest 28.0240895095,
XGBoost 26.1461582184, constant baseline 107.8078619553, and Dense MLP
17.2417659760 cycles. The visible raw max-minus-min spread is 90.5660959794
cycles (displayed 90.57). This single simulated training prefix is not accuracy
or confidence evidence.

### Saved metric tables in Model Laboratory

P11's verified selection manifest contains six candidate rows over the same
20 held-out development engine prefixes. The selected row is XGBoost depth 4,
version `xgboost-rul-v1`; the complete candidate metrics are recorded in
`docs/p21_s2_unsupported_claims_audit.md`. The numbers are development model
selection evidence, not an untouched final estimate.

P13's verified final report is partition `FD001/test`, final-report-only, five
model versions, 100 engines per model. `scripts/audit_p13_label_alignment.py`
verified the 100 answer-to-engine mapping and all 500 prediction/engine pairs.
`scripts/audit_p13_metrics.py` independently recomputed all five MAE/RMSE pairs
with zero delta:

| Model | MAE (cycles) | RMSE (cycles) |
| --- | ---: | ---: |
| Decision Tree v1 | 24.5133064250 | 33.9521945923 |
| Random Forest v1 | 20.3497082271 | 27.5199616554 |
| XGBoost v1 | 18.4489326859 | 24.1664481341 |
| Constant baseline v1 | 40.8793020212 | 52.6248575261 |
| Dense MLP final v1 | 18.1605317879 | 25.5735374644 |

These are aggregate error measures in cycles on the named 100-engine test
sample. They are not accuracy percentages or per-engine guarantees.

## Traceability issue list

No unmatched numeric output or unexplained result label was found in the six
screen review. Optional missing-artifact states and their wording are
purposeful: for example, Model Laboratory says verified P13 results are
unavailable in a public checkout, and Prediction/Fleet Overview do not show
RUL estimates without a trusted bundle. No application wording or calculation
needed a correction for this task. The report's source-hash table and UI checks
are the evidence for that finding; it does not claim the ignored artifacts
exist outside this workspace.

## Checks run

- `.venv/Scripts/python.exe scripts/check_p21_a2_feature_preview.py` PASS; 96 features and source/engine/cutoff printed above.
- `.venv/Scripts/python.exe scripts/check_p21_a1_cache_audit.py` PASS; same-source replay returned 39.6842105263 cycles at cycle 162 and 34.1960784314 at cycle 164, using the saved Decision Tree and engine 4 history.
- `.venv/Scripts/python.exe scripts/check_p17_t1_contribution_display.py` PASS; signed contribution values above were found in the rendered UI.
- `.venv/Scripts/python.exe scripts/check_p17_t2_model_spread_display.py` PASS; five rendered estimates and 90.566096-cycle spread verified.
- `.venv/Scripts/python.exe scripts/check_p20_s3_paired_artifact.py` PASS; source/model/scenario hashes, engine identities, raw RUL values and artifact round-trip verified.
- `.venv/Scripts/python.exe scripts/audit_p13_label_alignment.py` PASS; 100 engine/answer rows and 500 model/engine pairings.
- `.venv/Scripts/python.exe scripts/audit_p13_metrics.py` PASS; five MAE/RMSE pairs independently reproduced with zero delta.
- Direct `load_selection_manifest` / `load_final_test_display` verification PASS: P11 schema `p11-selection-manifest-v1`, 6 candidates and 20 validation engines; P13 loader returned all 5 expected versions with 100 rows per model.
- `.venv/Scripts/python.exe -m ruff check .` PASS (`All checks passed!`).
- `.venv/Scripts/python.exe -m pytest -q` PASS (688 passed in 195.86s).
- `git diff --check` PASS before the final handoff update; rerun before commit.
- Streamlit smoke NOT RUN as a separate local command because this audit changes no app code. PR CI's Streamlit smoke check will run; the unchanged six-screen UI had already been live-walked at laptop and narrow widths for P21-T1.

No model was trained, dataset was changed, or ignored artifact was added to Git.
The issue asks for a teammate to independently inspect a small example,
explain units and provenance, and approve the current PR revision; that human
review remains a separate acceptance check.

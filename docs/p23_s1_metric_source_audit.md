# P23-S1 local Model Laboratory metric-source audit

Status: the local source/display audit passed, and the owner-supplied public-app
screenshot shows the P22-S1 official FD001/test metric table. Its five model
versions, counts, MAE, and RMSE values match the verified local P13 report and
frozen model audit below. The screenshot's deployment revision predates this
branch's packaging fix; the dedicated official-results panel was previously
observed unavailable in the hosted app. Verify that panel again after deployment.

## What was checked

The Model Laboratory's **Official held-out test results (FD001/test)** section
loads `artifacts/final/p13/final_test_results.json` through
`load_final_test_display`. The loader validates the report checksum, schema,
partition, test-history and RUL-source hashes, intended final-report-only use,
five expected model versions, and exactly 100 engine rows per model.

The local AppTest-rendered table was compared row-for-row with those validated
loader records. Then `scripts/audit_p13_bundles.py` checked the saved P13
evaluation metadata against the frozen P12 manifests, model versions, weight
hashes, feature-config hash, and tree index. It checked ten files; all five
evaluated bundles matched.

| Final model | Version | Engines | MAE (cycles) | RMSE (cycles) |
| --- | --- | ---: | ---: | ---: |
| Decision Tree | `decision-tree-rul-v1` | 100 | 24.513306424988755 | 33.95219459227001 |
| Random Forest | `random-forest-rul-v1` | 100 | 20.34970822712025 | 27.51996165541398 |
| XGBoost | `xgboost-rul-v1` | 100 | 18.44893268585205 | 24.16644813405696 |
| Constant mean baseline | `constant-rul-v1` | 100 | 40.87930202123019 | 52.62485752612688 |
| Dense RUL MLP | `dense-rul-mlp-final-v1` | 100 | 18.160531787872316 | 25.573537464442705 |

The scores are error magnitudes in operating cycles, not accuracy percentages.
They summarize the 100 simulated FD001/test engines and were not used for model
selection or tuning. Example row: FD001/test engine 1 is observed through cycle
31 and has true RUL 112 cycles. The frozen Decision Tree reports raw RUL
164.96629213483146 cycles, absolute error 52.96629213483146 cycles. This is an
example audit row, not a future-engine guarantee.

## Frozen source identity

- P13 result payload SHA-256:
  `2a0686eaa3d369c435938cf6bf6256b047b3265f8ac8893e08e040adc798a6a6`.
- P12 tree-index payload SHA-256:
  `33bc4051835a5125f46ae31fc9f8a812325eb900559671de65dbe2e0273f8c89`.
- FD001/test history SHA-256:
  `3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851`.
- Separate RUL answer file SHA-256:
  `a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca`.
- Ordered feature config SHA-256: `19dd0331ee3e0c3a90217446a4481507b7da94fe1424b1c251c4605a6b168dfe`.
- Exact per-model manifest payload and weight hashes are recorded in
  `docs/p13_a3_frozen_bundle_audit.md` and checked by the P13-A3 audit.

## Cross-platform audit correction

The first Windows run of the existing P13-A3 audit failed on the whole-file
manifest hash. Investigation found that `core.autocrlf=true` converted tracked
JSON from repository LF to working-tree CRLF. Every manifest's Git blob hash
matched the corresponding P13 recorded file hash, and all canonical payload
hashes matched. The audit now normalizes CRLF to LF before comparing manifest
file identity. A regression test covers a Windows-style CRLF checkout. This
change does not weaken the payload checksum or serialized model-weight checks.

## Commands and results

- `.venv/Scripts/python.exe -m scripts.audit_p23_s1_metric_display` — PASS;
  no AppTest page exceptions, five rendered rows equal the verified report,
  five bundle identities match, ten files checked.
- `.venv/Scripts/python.exe -m pytest -q tests/test_p13_bundle_audit.py` — PASS,
  4 passed, including the CRLF regression.
- `.venv/Scripts/python.exe scripts/audit_p13_bundles.py` — PASS after the
  newline correction; all five final bundles matched the P13 report.

The local audit verifies the page and frozen artifacts. The owner-supplied
screenshot independently shows the P22-S1 table with matching rounded values;
the dedicated final-results panel was still absent on the previously deployed
revision. A fresh browser check after this report is deployed must confirm the
dedicated panel itself.

## Owner-supplied public screenshot and deployment fix

The Model Laboratory screenshot supplied in the project conversation shows the
P22-S1 table with the five frozen FD001/test model versions, 100 engines each,
and the same metrics listed above (rounded to four decimals). The values are
consistent with the local checksum-sealed report. The screenshot also shows the
P22 selection and study sections; those are separate development/diagnostic
views, not a substitute for the dedicated official-results panel.

The signed-in remote audit recorded in `docs/p23_a3_remote_parity.md` found the
dedicated **Official held-out test results (FD001/test)** panel unavailable on
the then-deployed revision because the report was Git-ignored. This change
packages only `artifacts/final/p13/final_test_results.json`, a 140,609-byte
checksum-sealed report with per-engine evaluation rows and source/model hashes.
It does not add raw sensor data, test history files, answer files, or model
weights. After this change is deployed, a fresh remote check is still needed to
confirm that the panel renders the same five rows.

## Post-deployment verification (2026-11-21)

The owner supplied a fresh signed-in screenshot of the deployed Model Laboratory at `https://fleetpulseai.streamlit.app/Model_Laboratory`, checked November 21, 2026 at 12:54:58 AM PKT. It shows the dedicated **Official held-out test results (FD001/test)** section, separate from the P22 study table. The screenshot's accessible table text exposes full precision; all five rows match the checksum-verified report exactly:

| Model | Version | Engines | MAE (cycles) | RMSE (cycles) |
| --- | --- | ---: | ---: | ---: |
| Decision Tree | `decision-tree-rul-v1` | 100 | 24.513306424988755 | 33.95219459227001 |
| Random Forest | `random-forest-rul-v1` | 100 | 20.34970822712025 | 27.51996165541398 |
| XGBoost | `xgboost-rul-v1` | 100 | 18.44893268585205 | 24.16644813405696 |
| Constant mean baseline | `constant-rul-v1` | 100 | 40.87930202123019 | 52.62485752612688 |
| Dense RUL MLP | `dense-rul-mlp-final-v1` | 100 | 18.160531787872316 | 25.573537464442705 |

The visible explanatory note identifies the values as error measures in operating cycles, not accuracy percentages or guarantees about future engines. The hosted commit was not displayed. The screenshot is attached to the owner's conversation; no local image path was supplied. This is a remote rendering check plus the exact match to the locally audited report and frozen manifests, not a claim that the hosting platform exposed its deployed SHA.

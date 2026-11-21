# P08-T2 model switching verification

**Issue:** [P08-T2 #130](https://github.com/Artfever/FleetpulseAI/issues/130)

**Observed:** 2026-11-16, Streamlit `AppTest`, Python 3.12.16, scikit-learn 1.9.1.

## Verification

The RUL Prediction page was run against the locally available saved development
bundles and the packaged FD001/train parity prefix for engine 4 through cycle
164. The same engine and cycle were kept selected while switching Decision Tree
to Random Forest and back. Each click used the manifest for the selected model
and showed that model's version:

| Selection | Displayed estimate | Model version | Engine / cycle |
| --- | ---: | --- | --- |
| Decision Tree | 34.20 cycles | `decision-tree-rul-v1` | 4 / 164 |
| Random Forest | 34.04 cycles | `random-forest-rul-v1` | 4 / 164 |
| Decision Tree again | 34.20 cycles | `decision-tree-rul-v1` | 4 / 164 |

Both models used the same past-only, 30-cycle input window. AppTest confirmed
that the selected identity and cutoff stayed at engine 4 / cycle 164 through
both switches and that the displayed model version followed the selected
manifest. The generic page caption now says “saved development model” because
the page can select either the P06 tree or the P08 forest.

## Interpretation and limits

The 0.16-cycle difference in this one example reflects two different saved
regressors; it does not show that either model is better. The forest's separate
20-engine FD001/train development-validation audit reports MAE 18.833479230754712
cycles and RMSE 25.24302308237894 cycles. This UI replay does not recompute or
replace that audit. Engine 4 is held out from fitting in the saved development
split, but this is still development data, not official-test evidence. The
packaged file is a three-engine prefix subset for parity examples, not the full
FD001 dataset. No maintenance guarantee is implied.

## Reproduction

The tracked regression test uses isolated fake artifact files and a stubbed
service to check selection state, manifest routing, and version labels without
requiring ignored model binaries in CI:

```powershell
.venv\Scripts\python.exe -m pytest -q tests\test_prediction_page.py
```

The real-bundle AppTest used ignored files in `artifacts/development/` and
`data/demo/fd001_training_prefixes.txt`; those files remain Git-ignored.

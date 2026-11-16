# P09-A2: XGBoost reload parity and same-batch runtime

## Saved candidate and prediction parity

Inspected the saved `xgboost-rul-v1` candidate generated for P09-T3. The
artifact verifier checked the model/manifest hashes, runtime version, saved
feature schema/config, source/split/table provenance, metrics, and every saved
validation identity and target. A fresh in-memory estimator was rebuilt from
the same frozen training configuration and fitting table.

The same ordered validation batch was then sent to the in-memory estimator, the
joblib-reloaded candidate, and the candidate's saved run record. Each contains
20 held-out FD001/train engine prefixes, each represented by 96 features.
Predictions matched exactly for **20/20** rows; maximum raw difference was
**0.0 cycles**. Model version: `xgboost-rul-v1`; XGBoost 3.4.1; saved model
SHA-256 `20801e897816f879c0f93f56dc199c49735c3afc3a544b595247f5c62d419612`.

## Same-batch timing method and observed result

Both estimators received the identical in-memory `validation.X` matrix in the
same process. Five warmup calls per estimator were excluded. Then each model
predicted that 20-row batch 200 times, with calls alternated in order. Timed
scope is `predict()` only; it excludes model fitting, feature engineering,
deserialization, raw-history service overhead, and app rendering.

| Estimator | Total for 200 batches | Mean per 20-engine batch |
|---|---:|---:|
| Fresh in-memory fit | 0.128937100 seconds | 0.000644686 seconds |
| Joblib-reloaded candidate | 0.128992000 seconds | 0.000644960 seconds |

Both measurements used Python 3.12.10 and XGBoost 3.4.1 on this local machine.
The roughly 0.000000275-second per-batch difference is smaller than a useful
practical distinction here; this single-process microbenchmark does not show
that either estimator is meaningfully faster. It demonstrates that the same
batch and same operation were timed. Results are machine-specific and should
not be used to rank models.

## Reproduction and limits

```powershell
.venv/Scripts/python.exe scripts/audit_xgboost_reload_runtime.py
```

The exact machine-local record, including hashes and full timing values, is
`reports/P09_A2_xgboost_reload_runtime.json` (Git-ignored). The underlying
split is development validation from FD001/train, not official test data. No
official test histories or RUL answers were opened. This task verifies
serialization/reload and controlled prediction timing; it does not claim
final-model quality or production latency.

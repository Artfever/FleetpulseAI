# P09-T3: saved XGBoost candidate and service registration

## Candidate and provenance

`scripts/train_xgboost_candidate.py` rebuilds the fixed P09-T2 estimator from
the shared fitting/validation tables, saves the estimator and hashed manifest,
verifies a reload against every saved prediction and metric, then launches the
shared-service replay. When the ignored P09-T2 run JSON is present, the script
also requires the new fit to reproduce each recorded validation identity and
raw prediction exactly.

The generated local bundle is in `artifacts/development/`:

- `xgboost_model.joblib`, SHA-256
  `20801e897816f879c0f93f56dc199c49735c3afc3a544b595247f5c62d419612`;
- `xgboost_validation.json`, schema `xgboost-validation-v1`, model version
  `xgboost-rul-v1`;
- `xgboost_metrics.json`, using the shared metrics-record format.

The manifest stores the XGBoost and scikit-learn versions, exact estimator
parameters, feature order/configuration, fitting/validation table digests,
source and semantic split hashes, fit runtime, all engine/cutoff identities,
validation targets, raw predictions, and recomputed errors. The joblib file and
metrics are ignored local artifacts; this document does not claim they were
published to GitHub.

## Verification observed

The fresh fit reproduced all 20 P09-T2 validation predictions exactly. The
artifact verifier reloaded the saved estimator and reproduced all 20 values
and the saved MAE/RMSE. The shared `predict_rul` service replay used each raw
FD001/train history only through its saved cutoff: **20/20 exact matches**,
maximum absolute difference **0.0 cycles**. The service returned
`xgboost-rul` / `xgboost-rul-v1`.

The Prediction page AppTest used the packaged training prefix for engine 4 at
cycle 164. It displayed **28.69 cycles** from the registered service result;
raw service output was **28.693811416625977 cycles**. The page exposed XGBoost
only when both the manifest and model file existed. Existing decision-tree
inference remains the default. If the XGBoost bundle is missing, the selector
does not offer it and the page explains the missing bundle.

XGBoost stays an optional package extra. The model registry resolves its
estimator type only when an XGBoost manifest is loaded, so importing the app or
using the tree/forest candidates does not require XGBoost. Loading the
candidate requires the matching pinned `xgboost==3.4.1` extra.

These checks establish artifact integrity, reload, and service/page wiring for
the held-out FD001/train development split. They do not establish final-test
quality, maintenance readiness, or remote deployment. The bundle is local and
Git-ignored; the hosted app has not been provisioned with this candidate.

# P21-A1: prediction cache and application state audit

Audit date: 2026-11-18. Scope: model deserialization, historical replay, uploads,
and the selected-fleet state handed from Fleet Overview to Maintenance Planner.

## Cache boundaries

| Area | Cached state | Identity/invalidation behavior |
| --- | --- | --- |
| Saved sklearn/XGBoost estimator (`services/predictor.py`) | Deserialized estimator, at most two entries | Keyed by exact model bytes and registered model name. Each prediction still rereads and verifies manifest/model integrity and checks saved feature/config/parameter compatibility before inference. |
| Saved MLP estimator (`services/predictor.py`) | Deserialized inference adapter, at most two entries | Keyed by checkpoint bytes, input dimension, hidden dimensions, and serialized scaler. Bundle integrity and manifest checks run on each prediction. |
| Historical replay (`services/history.py`) | At most 128 prediction points, least recently used first | Key contains dataset partition, engine ID, cutoff, canonical full observed prefix SHA-256, saved model version, feature config SHA-256, and a fingerprint of exact manifest plus model bytes. Unreadable/unrecognized bundle metadata bypasses this cache and runs the normal inference path. |
| Uploaded CSV | No application prediction cache | The upload parser and prediction service validate and process the current upload directly. A changed upload is not served from the replay cache. |
| Fleet-to-planner UI state | Streamlit session state snapshot, not a prediction cache | A saved fleet includes model and source provenance. Saving a new fleet clears dependent P20 scenario, comparison, and comparison-artifact entries before the planner can reuse them. |

`load_engine_history` is called from page rendering without a Streamlit data
cache. No `st.cache_data` or `st.cache_resource` decorators were found under
`app/` or `src/fleetpulse/`.

## Evidence

Focused regression tests in `tests/test_history_service.py` confirm that
identical history/model/config reuses a cached replay result, while changing
history, dataset partition, engine identity, model version, feature config, or
model bytes causes a cache miss. `tests/test_predictor.py` verifies exact-byte
model deserialization reuse and rejection of changed/corrupt/missing artifacts.
`tests/test_planner_selection.py` confirms saving a different selected fleet
clears the dependent planner snapshot; the P20 workflow tests confirm the saved
identity and model version reach the planner.

Actual local integration check used the saved development Decision Tree bundle
and packaged FD001/train engine 4 history. Model version
`decision-tree-rul-v1`; training seed 42; tree parameters `max_depth=8`,
`min_samples_leaf=5`; feature config `past-only-v1`, 30-cycle window, 24
channels and 96 ordered features. Source SHA-256:
`963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8`; model
SHA-256:
`2a693e6ecc09de5d9549e287a6798fdb339c7e945c700231444f0c5bcd425018`.

Replay at cutoff 162 returned raw/display RUL `39.68421052631579` cycles;
cutoff 164 returned `34.19607843137255` cycles. Both report the same model
version, and two distinct replay entries were present after these two cutoffs.
These are saved-model outputs for two prefixes of one packaged training
history, not accuracy or fleet-performance evidence. No model was retrained.

## Limits and next consumer

The current replay identity checks cover data, feature config, model version,
and artifact contents. Uploads are intentionally uncached. This task does not
change shared UI state design or remove page-local feature calculations; those
belong to P21-A2. P21-A3 consumes the audited state behavior in the built-in and
uploaded end-to-end journey.

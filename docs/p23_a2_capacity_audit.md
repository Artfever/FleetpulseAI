# P23-A2 upload, replay, and model-capacity audit

Status: local implementation and measurements only. P23-A2 issue #173 remains
blocked on P23-A1 #96's deployed-release verification. This audit does not claim
that the hosted service has been checked.

## Existing bounds inspected

- `src/fleetpulse/data/input_validation.py` limits uploaded CSV input to
  `MAX_UPLOAD_BYTES = 1_048_576` bytes (1 MiB), and rejects oversize input.
- `src/fleetpulse/services/history.py` limits replay requests to
  `MAX_REPLAY_POINTS = 10` and retains no more than
  `MAX_REPLAY_CACHE_POINTS = 128` points in its bounded cache.
- The upload widget in `app/pages/3_RUL_Prediction.py` previously had no
  widget-level size limit. Streamlit's default upload limit is 200 MB, which
  meant an oversized file could be received before the parser rejected it.
- The upload widget now sets `max_upload_size=1` (MiB), matching the parser's
  1 MiB contract. `tests/test_upload_page.py` asserts the configured widget
  protocol value.
- `compare_final_models` validates history before accessing artifacts. Invalid
  histories raise `PredictionInputError` before a model is called. Per-model
  load/validation/prediction errors are returned with a model name, code, and
  reason; one failed model does not prevent attempts on the remaining models.
  The service distinguishes `artifact_unavailable`, `dependency_unavailable`,
  `incompatible_artifact`, and `prediction_failed`. If none succeed, it returns
  no synthetic estimates. The service documentation states there is no model
  cache, training, or development-model fallback.
- Existing tests cover invalid input before artifact access, all bundles
  missing, incompatible manifests/weights, one estimator failure with continued
  per-model handling, and no fake estimate when all models fail.

## Local model-load profile

Profiled locally on Windows by loading all five packaged final models and
running `compare_final_models` on the tracked demo history for FD001/train
engine 4 through cycle 164. The profile process already had the application
imports and demo rows loaded.

- Total five-bundle asset size: 25,700,858 bytes.
- One comparison call elapsed: 11.2325 seconds.
- Process working set: 170,254,336 bytes before model load and
  339,136,512 bytes after; measured increase 168,882,176 bytes (about 161 MiB).
- All five models loaded; zero model failures.
- Raw outputs: Decision Tree 29.7941176471, Random Forest 28.0240895095,
  XGBoost 26.1461582184, constant mean 107.8078619553, and Dense MLP
  17.2417659760 cycles.

These are local process measurements, not clean cold-start, Streamlit Cloud,
concurrency, latency-SLO, or memory-limit evidence. The example outputs are a
single simulated FD001/train history and are not accuracy or maintenance
evidence. No model was retrained.

## Verification

- `.venv/Scripts/python.exe -m ruff check app/pages/3_RUL_Prediction.py tests/test_upload_page.py` — PASS.
- `.venv/Scripts/python.exe -m pytest -q tests/test_upload_page.py tests/test_input_validation.py tests/test_history_service.py tests/test_final_comparison.py` — PASS, 59 passed.
- `.venv/Scripts/python.exe -m ruff check .` — PASS.
- `.venv/Scripts/python.exe -m pytest -q` — PASS, 731 passed in 205.96 seconds.
- The focused final-comparison tests are included in the 59-test focused run and
  the full 731-test run; they exercise the error cases listed above.
- `.venv/Scripts/python.exe -m streamlit run app/Home.py --server.port 18631 --server.headless true` — first launch reported port already in use; the existing local app health endpoint was checked with `curl.exe --fail --silent --show-error http://localhost:18631/_stcore/health` and returned `ok` (HTTP 200). The existing app process was stopped after the check.

## Remaining evidence

Issue #173 has native blocker #96. A1 still needs an exact deployed commit and
bundle inventory plus an independent check of the public app. The public app
currently redirects anonymous requests to Streamlit sign-in; local profile
data cannot replace deployed memory/loading or remote error-behavior evidence.

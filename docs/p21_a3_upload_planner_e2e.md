# P21-A3: Built-in and uploaded histories through prediction and planning

P21-A3 (#223) verifies that a built-in history and a validated CSV upload can
reach the Maintenance Planner without losing engine identity, prediction
values, or model/source provenance.

## Uploaded route

On the RUL Prediction page, a successful upload prediction can be saved as a
single `user/upload` planner row. The row retains engine ID, observed cycle,
displayed and raw RUL, model/version, feature-row count, warnings, and the
error field expected by the planner. The provenance snapshot includes a hash
of the selected normalized records, the original upload bytes, and the verified
model manifest/artifact identity. The planner then uses that saved row for its
shared scenario and paired EDD/GA comparison.

The pending prediction is keyed by upload-byte SHA-256, selected engine, model
choice, and local manifest/artifact SHA-256 values. Changing those inputs,
clearing/rejecting the upload, or switching away from upload clears the pending
result. If the saved planner fleet came from an upload, changing the upload also
clears that fleet and its dependent scenario/comparison artifacts. A successful
prediction remains visible if provenance cannot be verified, but planner save
is disabled.

## Verification evidence

- `tests/test_upload_planner_workflow.py` covers the full upload-page to
  planner-page flow with a clearly synthetic mocked prediction (12.5 cycles),
  checking the handoff row, provenance, and both schedulers.
- The same test file includes an integration test using the ignored local
  development Decision Tree bundle and the packaged simulated FD001/train
  engine-4 prefix. It uploads the 164 observed rows through the UI, predicts,
  saves the row, and builds the paired planner artifact. The observed cutoff is
  cycle 164; model version is `decision-tree-rul-v1`; raw RUL is
  34.19607843137255 cycles; the planner uses the same nonnegative value. This
  is one simulated training-prefix prediction, not accuracy, official-test, or
  maintenance evidence. The real-bundle test skips in clean environments where
  that intentionally ignored artifact is absent.
- The local manifest identifies the model's training seed as 42,
  `max_depth=8`, and `min_samples_leaf=5`; it records 16,738 fitting rows from
  80 engines. The feature config is `past-only-v1` with a 30-cycle window and
  96 ordered features. These identify the saved local artifact used by the
  integration test; no training or artifact changes were made in this task.
- Existing `tests/test_p20_workflow_gate.py` covers the built-in packaged
  FD001/train flow for engines 4, 5, and 12 through Fleet Overview and both
  schedulers using the same local bundle. It asserts raw RUL values
  34.19607843137255, 158.16, and 184.35714285714286 cycles respectively.
- A synthetic cache-invalidation test verifies that changing the selected
  uploaded engine clears the pending result and any upload-backed planner
  snapshot and scenario.

Checks on the implementation branch:

- `.venv/Scripts/python.exe -m ruff check .` — PASS.
- `.venv/Scripts/python.exe -m pytest -q tests/test_upload_page.py tests/test_upload_planner_workflow.py tests/test_p20_workflow_gate.py` — PASS, 8 tests.
- `.venv/Scripts/python.exe -m pytest -q` — PASS, 687 tests in 66.16 seconds.
- `.venv/Scripts/python.exe -m streamlit run app/Home.py --server.port 18603 --server.headless true` — started; `curl.exe --fail --silent --show-error http://localhost:18603/_stcore/health` returned `ok`; server stopped.
- `git diff --check` — PASS.

Abdullah confirmed that a teammate inspected and discussed the exact uploaded
output and its limits. No raw data, model, or trained artifact was added to
Git.

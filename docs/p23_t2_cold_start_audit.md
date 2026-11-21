# P23-T2 local cold-start and full-flow audit

Status: local fresh-process, page-route, and prediction-to-planner checks passed.
The hosted cold start and a fresh remote browser flow are not independently
observable from this environment: anonymous requests to the app redirect to
Streamlit sign-in (HTTP 303). Abdullah reports that he can open the deployed
URL. That owner check does not establish remote startup time or every screen's
displayed values.

## Fresh process startup

Command: `.venv/Scripts/python.exe scripts/smoke_p23_t2_cold_start.py`.

The script starts a new headless Streamlit process on a free local port, waits
for `/_stcore/health`, records process-start-to-ready duration, and shuts down
the child process. Two clean runs took **3.395 seconds** and **4.612 seconds**;
both returned HTTP 200 with `ok` on Python 3.12.16. Streamlit is pinned to
1.67.0. This is local startup timing, not Streamlit Cloud cold-start latency.

## Screens and demo flow

- `tests/test_page_shells.py` — all five app pages render in a fresh AppTest
  session when data/artifacts are missing; each shows its explicit unavailable
  or empty state without an exception.
- `tests/test_p21_t1_navigation.py` — one fresh Home session follows links
  through Fleet Overview, Engine Explorer, RUL Prediction, Maintenance Planner,
  and Model Laboratory; no page raises an exception.
- `tests/test_upload_planner_workflow.py` — the real packaged FD001/train
  engine-4 history (164 observed rows, ending cycle 164) was uploaded and
  predicted with the saved development Decision Tree. It produced raw RUL
  **34.19607843137255 cycles**, model version `decision-tree-rul-v1`; the same
  `user/upload` engine, cycle, prediction, and source/model provenance reached
  the planner scenario. Synthetic-path checks also exercise the paired EDD/GA
  workflow, but are not real-model schedule results.
- The bundled final serving files and 229-row demo subset are tracked. The
  local development Decision Tree and P13 official score report are ignored
  files; they are present in this working environment but absent from the
  tracked deployment tree. The app has explicit unavailable states when those
  local-only reports/models are absent. This audit does not claim they are
  provisioned in the hosted app.

This prediction is a smoke example from one simulated training history, not an
accuracy result or maintenance recommendation.

## Commands and results

- `.venv/Scripts/python.exe -m pytest -q tests/test_page_shells.py tests/test_p21_t1_navigation.py tests/test_upload_planner_workflow.py` — PASS, 9 passed.
- `.venv/Scripts/python.exe scripts/smoke_p23_t2_cold_start.py` — PASS twice, 3.395s and 4.612s to local health HTTP 200/`ok`.
- `.venv/Scripts/python.exe -m ruff check .` — run and result recorded in the task handoff.
- `.venv/Scripts/python.exe -m pytest -q` — PASS, 732 passed in 97.05s.

Remaining T2 evidence: test the fresh browser flow against the deployed app and
compare its screen output with service results; inspect hosting cold-start logs.
Those checks require access beyond the anonymous sign-in redirect observed
here.

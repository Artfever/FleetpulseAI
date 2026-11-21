# P21 connected application gate evidence

Parent issue: [#26](https://github.com/Artfever/FleetpulseAI/issues/26)

P21's outcome is that the principal app screens work as one connected product. All nine child issues (#90, #167, #223, #91, #168, #224, #92, #169 and #225) are closed. Predecessor gates P14 #16, P16 #18, P17 #24 and P20 #25 are closed.

## Connected screens and selection state

The app navigation exposes Home, Fleet Overview, Engine Explorer, RUL Prediction, Maintenance Planner and Model Laboratory. PR #494 connects the six screens; `tests/test_p21_t1_navigation.py` checks the page destinations. PR #496 adds explicit persistent session state for history source, engine, model and observed cycle. `tests/test_p21_t2_session_state.py` exercises engine/cycle transfer between Engine Explorer and RUL Prediction and model restoration after returning to Prediction.

The P21-T3 browser walk used the locally running Streamlit app at the laptop viewport 1365×900 and narrow viewport 390×844. It visited all six routes at each size. Each page rendered its expected title; the browser reported no page errors; `document.documentElement.scrollWidth` equaled `window.innerWidth` for all twelve route/viewport combinations. At 390px, planner input columns computed to `flex-direction: column` with 302px-wide fields; at 1365px, they remained in rows with 280px-wide fields. Dark-theme screenshots were captured outside the repository under the system temporary directory.

## Saved-model fleet path

`tests/test_p20_workflow_gate.py::test_real_saved_model_fleet_selection_reaches_both_schedulers` follows Fleet Overview selection into Maintenance Planner with the local saved Decision Tree v1. The packaged simulated FD001/train prefixes yielded raw estimates of 34.19607843137255 cycles for engine 4/cycle 164, 158.16 cycles for engine 5/cycle 58, and 184.35714285714286 cycles for engine 12/cycle 7. Both schedulers used the same scenario hash and scheduled all three engines. With illustrative one-cycle durations and priority weights, one team, a two-cycle buffer and six-cycle soft horizon, EDD and GA both produced zero weighted tardiness, three-cycle makespan and zero horizon overflow. These settings are demonstration assumptions, not maintenance recommendations or model-quality evidence.

The saved-model bundle and source histories are local ignored artifacts, not committed raw data. The real-model test is guarded and may skip in clean CI without that bundle. CI still runs the rest of the automated contract and synthetic journey checks.

## Uploaded-history path and limits

P21-A3's [#223 evidence](https://github.com/Artfever/FleetpulseAI/issues/223) records the uploaded FD001/train engine-4 prefix (164 rows through cycle 164), the saved Decision Tree v1 estimate of 34.19607843137255 raw cycles, and its arrival in the paired EDD/GA planning scenario with provenance attached. `tests/test_upload_planner_workflow.py` covers the real saved-bundle path when artifacts are available and the synthetic upload-to-planner transfer. This one simulated training-prefix result is not official-test performance, accuracy, or maintenance evidence.

## Verification

Combined gate-focused command:

```powershell
.venv\Scripts\python.exe -m pytest -q tests/test_p20_workflow_gate.py tests/test_upload_planner_workflow.py tests/test_p21_t1_navigation.py tests/test_p21_t2_session_state.py tests/test_p21_t3_responsive_layout.py
```

Result: **7 passed in 16.56s** with the local saved Decision Tree bundle present.

Latest complete checks on the P21-T3 revision: `.venv\Scripts\python.exe -m ruff check .` passed; `.venv\Scripts\python.exe -m pytest -q` passed (**691 passed in 92.23s**); Streamlit health on port 18607 returned HTTP 200 and `ok`; `git diff --check` passed. PR #497's Development CI run `44580977121` passed branch-policy, quality, unit-and-contract-tests, Streamlit smoke and ci-gate.

## Limits

The examples use simulated FD001/train data. The planner assumptions are illustrative. No official-test estimate, accuracy claim, fault diagnosis, safety result or maintenance benefit is inferred. The source/model bundle is ignored and unavailable to a clean clone until provisioned.

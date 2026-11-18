# P17-T3: explain explanation limits in plain language

The RUL Prediction page now places a plain-language note beside the selected
history and estimate. It explains that a feature contribution describes how a
sensor statistic shifted a model output relative to a reference; it does not
show that a sensor physically caused wear or identify an FD001 fault type.
FD001 is simulated, and the displayed cycle estimate is not a maintenance
diagnosis.

The comparison panel separately states that model spread is disagreement
between raw outputs, not a confidence interval or probability. Both the local
SHAP contribution view and five-model comparison remain tied to one selected,
past-only engine prefix. Neither supplies fleet-wide accuracy or evidence about
real-world failure mechanisms.

## Verification

`tests/test_prediction_page.py::test_selected_engine_and_cutoff_reach_service_without_future_or_target`
checks that the limit note appears in the page with the selected history. Run:

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_prediction_page.py
```

No model or data changes are required for this explanation; the actual saved
model output and model-disagreement values remain documented in the P17-T1 and
P17-T2 reports.

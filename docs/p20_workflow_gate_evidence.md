# P20 shared workflow gate evidence

The P20 shared gate is exercised from the actual Streamlit pages with the tracked development Decision Tree bundle and packaged FD001/train prefixes. The AppTest user flow opens Fleet Overview, explicitly selects engines 4/5/12, saves the selected fleet, switches to Maintenance Planner, and submits the planning scenario. The saved model and raw histories are real project files; all scheduling settings remain illustrative simulation assumptions.

Command:

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_p20_workflow_gate.py
```

The observed Fleet Overview inputs were:

| Engine | Observed cycle | Saved model | Raw RUL (cycles) |
| --- | ---: | --- | ---: |
| FD001/train 4 | 164 | decision-tree-rul-v1 | 34.19607843137255 |
| FD001/train 5 | 58 | decision-tree-rul-v1 | 158.16 |
| FD001/train 12 | 7 | decision-tree-rul-v1 | 184.35714285714286 |

The Planner received those same RUL values and engine identities. Its chosen defaults were one team, a two-cycle buffer and a six-cycle soft horizon, with duration 1 and priority weight 1 for each engine. These are illustrative assumptions, not measured maintenance values. Both schedules were feasible, shared scenario hash `3c58edea62bf88da3eb79b66e14779273e523a77cd7cc84b2f58b7560ac93049`, and reported zero weighted tardiness, three-cycle makespan and zero horizon overflow. The paired JSON download was available and retained the model and scenario provenance.

This verifies the connected application path for this small packaged subset. It is not a full fleet run, not prediction-quality evidence, and not an operational maintenance recommendation. The test is UI automation; it does not claim a human teammate review.

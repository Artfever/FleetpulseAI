# P20-T2 comparable schedule results

The Maintenance Planner now evaluates EDD and the seeded Genetic Algorithm from the exact same immutable scenario snapshot. It presents both methods side by side with the same summary measures, per-engine schedule columns, and relative-cycle timelines. Both charts use the same cycle range and show the same soft-horizon marker. The GA seed, population, generation count and distinct order count are shown with its result. A comparison is saved when the user submits the scenario form; changing form inputs without submitting does not silently rerun or change the saved comparison.

## Actual saved-model wiring run

Command:

```powershell
.venv/Scripts/python.exe scripts/check_p20_t2_real_comparison.py
```

The script loaded the existing development Decision Tree bundle and the packaged `FD001/train` prefix subset. It did not train a model or use labels for prediction. The output identified model `decision-tree-rul-v1`, model SHA-256 `2a693e6ecc09de5d9549e287a6798fdb339c7e945c700231444f0c5bcd425018`, manifest SHA-256 `41f16e1397e1e2780133fd8a6aae8e8990d4af260055111b595c9831c8cb13b3`, and the common scenario SHA-256 `3c58edea62bf88da3eb79b66e14779273e523a77cd7cc84b2f58b7560ac93049` for both reports.

| Engine | Observed cycle | Raw/planner RUL (cycles) | Due completion (cycles) |
| --- | ---: | ---: | ---: |
| FD001/train 4 | 164 | 34.19607843137255 | 32.19607843137255 |
| FD001/train 5 | 58 | 158.16 | 156.16 |
| FD001/train 12 | 7 | 184.35714285714286 | 182.35714285714286 |

The same single-team scenario was passed to EDD and GA: duration 1 and weight 1 per job, buffer 2 cycles, and soft horizon 6 cycles. These are illustrative simulation assumptions. Both methods ordered engines 4, 5, 12, produced weighted and total tardiness 0, makespan 3 cycles, no late jobs, and no horizon overflow. This shows the service and comparison path working; it is not evidence about prediction quality or schedule superiority. The GA used seed 42, population 16 and 20 generations, evaluating six distinct orders.

The automated Streamlit comparison tests use mocked fleet predictions for deterministic UI checks and verify that both reports retain the same engine set and deadlines. The separate script above is the real saved-model integration check. The checked-in plot and tables label all time values in cycles and retain engine partition/ID in the selected-fleet view.

# P20-S2 urgent and capacity-pressure checks

This check runs both schedulers on the same fixed, explicitly synthetic four-job scenario in `tests/fixtures/p20_s2_capacity_pressure.json`. It is not an FD001 result or operational maintenance recommendation. Predicted RUL values are illustrative; service durations, weights, team capacity, buffer, and horizon are simulation assumptions.

The scenario has one team, a two-cycle RUL buffer, and a six-cycle soft horizon. Engine 1 has RUL 1, so its completion deadline clamps to cycle 0; the four jobs have durations 3, 1, 4, and 2 cycles. Engines 3 and 4 have tied due dates and tied priority weights. The 10-cycle total duration exceeds the six-cycle horizon, creating capacity pressure.

Run it with:

```powershell
.venv/Scripts/python.exe scripts/check_p20_s2_scheduler_cases.py
```

Observed output:

| Measure | EDD | Genetic Algorithm |
| --- | ---: | ---: |
| Order | 1, 2, 3, 4 | 2, 4, 1, 3 |
| Weighted tardiness (weighted cycles) | 23 | 12 |
| Late jobs | 4 | 2 |
| Makespan (cycles) | 10 | 10 |
| Soft-horizon overflow (cycles) | 4 | 4 |
| Jobs still visible | 4 | 4 |

The EDD order follows due date and then qualified engine identity; it does not use duration or priority to choose order. The seeded GA run used 16 population members, 20 generations, and seed 42, evaluating 20 distinct orders. It found a lower weighted tardiness score for this one case; this is a synthetic illustration, not evidence that GA always beats EDD. Both methods preserve all jobs, and the horizon reports overflow separately instead of dropping work.

Automated assertions are in `tests/test_p20_s2_scheduler_cases.py`. They cover the zero-deadline job, late work under one-team capacity pressure, all rows remaining present after the soft horizon, varied durations, tied priorities, and repeatability of the seeded GA schedule.

# P20-T3 planner assumptions and overflow

The Maintenance Planner separates the RUL used to calculate deadlines from scenario assumptions. In the Fleet Overview flow, the planner RUL is derived from a saved-model prediction. In the synthetic scenario, RUL is an example input. Service durations and priority weights are never inferred from FD001; team count, buffer, and horizon are also explicit scenario choices.

For each job, due completion is `max(0, planner RUL - buffer)`. A job is late when its finish exceeds that due completion. Per-job `Late by (cycles)` is the positive difference. Weighted tardiness sums late cycles multiplied by the explicitly assumed priority weight.

The horizon is a soft reporting limit. It does not remove a job, cancel work, or change the weighted tardiness objective. Every selected job remains in both schedules; each row shows `Beyond horizon (cycles)`, and the method summary reports aggregate horizon overflow separately.

## Verified UI example

The AppTest submits the synthetic Appendix B scenario with its reviewed inputs: RUL 4/5/8 cycles, durations 3/1/2 cycles, priority weights 1/5/1, one team, and a two-cycle buffer. With the soft horizon set to 2 cycles:

| Method | Order | Late by per job (cycles) | Beyond horizon per job (cycles) | Weighted tardiness (weighted cycles) | Aggregate overflow (cycles) |
| --- | --- | --- | --- | ---: | ---: |
| EDD | A, B, C | 1, 1, 0 | 1, 2, 4 | 6 | 4 |
| Seeded GA | B, A, C | 0, 2, 0 | 0, 2, 4 | 2 | 4 |

All three jobs appear in both tables. Horizon overflow is calculated independently from tardiness and is not a hidden score penalty. The run is synthetic and illustrates the UI/scoring contract; it is not evidence of model quality or maintenance policy.

## Saved-model path example

The integrated fleet path is exercised by:

```powershell
.venv/Scripts/python.exe scripts/check_p20_s3_paired_artifact.py
```

It loads the existing development Decision Tree and packaged FD001/train prefixes for engines 4, 5 and 12, predicts their observed histories, builds one planner scenario, and passes that same scenario to EDD and GA. Raw/planner RUL values were 34.19607843137255, 158.16, and 184.35714285714286 cycles. The check used one team, duration/priority 1 per engine, buffer 2, and horizon 6; those are illustrative assumptions. Both methods reported zero weighted tardiness, makespan 3 cycles, and no overflow. This is a three-prefix integration check, not a complete-fleet or model-quality result. UI interaction tests use mocked model predictions and the synthetic case above.

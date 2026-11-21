# P20-A3 paired scheduler service

`fleetpulse.services.compare_schedules(scenario, *, population_size, generations, seed)` runs EDD and the genetic algorithm with the same immutable `SchedulingScenario`. It returns a `ScheduleComparisonResult` with one common scenario SHA-256 and two `ScheduleMethodResult` values. Each result includes the checked JSON-safe schedule report, algorithm runtime, feasibility, weighted tardiness, qualified identities of late jobs, makespan, horizon overflow, and (for GA) search settings/history.

Both reports are revalidated by the shared schedule feasibility checker before the service returns. If a schedule fails validation, or GA settings are invalid, the call raises and returns no partial pair. Runtime measures the scheduler call only; it excludes report validation/serialization and is machine-dependent. It is diagnostic output, not a runtime guarantee. Raw model predictions remain on their original prediction results; the scheduling scenario and report contain planner RUL and explicitly assumed schedule inputs.

## Hand-calculated example

Command:

```powershell
.venv/Scripts/python.exe scripts/check_p20_a3_schedule_service.py
```

On the synthetic Appendix B jobs (RUL 4/5/8, durations 3/1/2, weights 1/5/1, one team, buffer 2, horizon 6), the service returned the same scenario hash for both reports. EDD was feasible with weighted tardiness 6 cycles, late jobs A and B, makespan 6 cycles and zero overflow. Seeded GA (16 members, 20 generations, seed 42) was feasible with weighted tardiness 2, late job A, makespan 6 cycles and zero overflow. One observed run measured about 0.00025s for EDD and 0.00948s for GA; runtimes vary by environment.

## Saved-model wiring check

The same script also loads the existing development Decision Tree v1 and packaged FD001/train history prefixes. On that three-engine subset, observed raw RUL was 34.19607843137255, 158.16, and 184.35714285714286 cycles for engines 4, 5, and 12. The service produced matching feasible EDD/GA reports with scenario hash `3c58edea62bf88da3eb79b66e14779273e523a77cd7cc84b2f58b7560ac93049`. Under illustrative duration/weight 1, one team, buffer 2 and horizon 6 assumptions, both scores were 0, makespan was 3 cycles, and no job was late or beyond horizon. One observed run took about 0.00015s for EDD and 0.00950s for GA. This is integration evidence only, not model-quality or scheduling-superiority evidence.

Automated checks in `tests/test_schedule_comparison.py` verify the hand-calculated scores, same-scenario identity, feasibility, late jobs, makespan, overflow, runtime fields, GA reproducibility, and invalid-input errors. The selected-fleet and synthetic UI flows now use this common service.

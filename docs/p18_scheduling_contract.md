# P18 synthetic job input contract

P18-S1 defines the inputs consumed by the shared maintenance-scheduling
objective. `fleetpulse.optimization.jobs.MaintenanceJob` is immutable and stores
one dataset-qualified engine identity, predicted remaining useful life (RUL) in
cycles, a positive integer service duration in cycles, and a positive integer
priority weight. RUL must be finite and nonnegative. Duration and weight are
synthetic scenario assumptions, not measured maintenance records.

The existing `EngineIdentity(dataset_partition, unit_id)` type preserves
partition context: numeric unit 7 in `FD001/train` is different from unit 7 in
`FD001/test`. The Appendix B fixture uses the explicit namespace
`synthetic/appendix-b-v1`; its RUL inputs are illustrative too and are not
predictions from the model or NASA observations.

## Tiny input example

`appendix_b_jobs()` returns:

| Dataset-qualified engine | Predicted RUL (cycles) | Duration (cycles, synthetic) | Priority weight (synthetic) |
| --- | ---: | ---: | ---: |
| `synthetic/appendix-b-v1:1` | 4 | 3 | 1 |
| `synthetic/appendix-b-v1:2` | 5 | 1 | 5 |
| `synthetic/appendix-b-v1:3` | 8 | 2 | 1 |

This task defines the input records only. Due dates, completion-time tardiness,
the weighted objective, team capacity, and horizon behavior remain later P18
tasks. The plan's worked example uses a buffer of two cycles and completion as
the deadline event; teammate agreement on that scheduling convention is still
required at phase acceptance. No real fleet maintenance durations or priorities
are inferred here.

## Completion deadline and tardiness

P18-S2's `calculate_job_timing(job, buffer_cycles=..., start_cycle=0)` applies
the plan's convention: due completion cycle is
`max(0, predicted_rul_cycles - buffer_cycles)`, completion is start plus service
duration, and tardiness is `max(0, completion - due)`. RUL and buffer may be
fractional cycles; start and service duration are integer cycles. The default
start is relative cycle zero. A decoder can pass a later nonnegative integer
start. Buffer must be finite and nonnegative. This function does not score jobs,
assign teams, enforce capacity, or drop horizon overflow.

For Appendix B's one-team inputs and two-cycle buffer:

| Order | Engine | Due completion | Start | Completion | Tardiness (cycles) |
| --- | --- | ---: | ---: | ---: | ---: |
| A/B/C | 1 | 2 | 0 | 3 | 1 |
| A/B/C | 2 | 3 | 3 | 4 | 1 |
| A/B/C | 3 | 6 | 4 | 6 | 0 |
| B/A/C | 2 | 3 | 0 | 1 | 0 |
| B/A/C | 1 | 2 | 1 | 4 | 2 |
| B/A/C | 3 | 6 | 4 | 6 | 0 |

These are hand-calculated outputs for a synthetic example, not observed fleet
outcomes. P18-S2's required teammate confirmation that the deadline event is
completion rather than service start remains an acceptance check.

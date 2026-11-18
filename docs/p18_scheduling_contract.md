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
the deadline event; Abdullah directly accepted that scheduling convention on 18 November 2026
after inspecting the worked example and its explanation. The shared phase gate
still requires the remaining scheduling outputs. No real fleet maintenance durations or priorities
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
outcomes. On 18 November 2026, Abdullah independently reviewed the executed example,
requested a plain-English explanation, and explicitly confirmed: deadlines
apply to completion; late and horizon-overflow work remains visible; duration
and priority weights are labelled simulation assumptions. This is his direct
confirmation, not a claimed GitHub approving review or a report of a private
message. It supersedes the earlier pending inspection status for #159.


## P18-S2 closeout verification (18 November 2026)

The implementation was already merged by PR #372 as
`1fa3895bda77d3a7bf06b937ae40f2fd9337f84f`, with all five required CI checks
passing. Abdullah authorized taking over remaining team work, including this
Shaheer-assigned closeout; original implementation authorship is preserved.
No duplicate timing code or new model result was created.

- `.venv/Scripts/python.exe -m pytest -q tests/test_scheduling_jobs.py`: **22 passed in0.31s**.
- `.venv/Scripts/python.exe -m ruff check .`: PASS.
- `git diff --check`: PASS.
- Full pytest NOT RERUN for this documentation-only closeout; the unchanged
  implementation tree passed 531 tests with one missing-P06-artifact skip in
  P17 PR #446. Required PR CI still applies.
- Local Streamlit smoke NOT RUN:no app change.

This completes the timing convention/evidence for #159 after closeout merge.
The weighted objective/hard feasibility rules (#217), decoder (#81), EDD
(#158), scenario UI and shared P18 gate are separate remaining work.


## P18-S3 shared objective and hard feasibility

`fleetpulse.optimization.scoring` defines immutable `SchedulingScenario`,
`JobAssignment`, `ScheduledJob`, and `ScheduleEvaluation` records. A scenario
owns the reviewed jobs, a positive integer `team_count`, nonnegative finite
`buffer_cycles`, and a positive finite soft `horizon_cycles`. Job identities
are dataset-qualified and unique. An assignment supplies an engine identity,
a one-based team ID and a nonnegative integer start cycle. Completion and
lateness are derived through the existing reviewed `calculate_job_timing`.

`evaluate_schedule(scenario, assignments)` requires every selected job exactly
once, no unknown jobs, every team ID within capacity, and non-overlapping
half-open service intervals `[start, completion)` on each team. Touching
intervals are allowed; different identical teams work concurrently. Violations
raise `ScheduleFeasibilityError` instead of receiving a comparable score.
No ordering/decoder/EDD/GA is implemented by this scoring component.

The sole objective is **sum(priority_weight * tardiness_cycles)**. Lower is
better. Weights are dimensionless simulation assumptions; the cost has weighted
cycle units. There is no hidden horizon penalty or dropped job. The result
records weighted and unweighted tardiness, late job count, makespan, overflow
job count, and every scheduled job with its input/timing/team/per-job overflow.
Aggregate `horizon_overflow_cycles` means `max(0, makespan - horizon)`, not a sum
of per-job overflow. Fractional RUL/deadlines retain precision. Nonfinite
objectives are rejected.

Executed Appendix B results:

| Scenario | Weighted tardiness | Makespan | Horizon | Aggregate overflow | Jobs kept |
| --- | ---: | ---: | ---: | ---: | ---: |
| One team, A/B/C | 6 | 6 | 6 | 0 | 3 |
| One team, B/A/C | 2 | 6 | 6 | 0 | 3 |
| One team, A/B/C | 6 | 6 | 2 | 4 | 3 |
| Two teams: A[0,3), B[0,1), C[1,3) | 1 | 3 | 2 | 1 | 3 |

Weights A/B/C are 1/5/1. Thus A/B/C costs `1*1 + 5*1 + 1*0 = 6`;
B/A/C costs `5*0 + 1*2 + 1*0 = 2`. Abdullah explicitly independently checked
and accepted this objective/example and the hard/soft rule distinction on
18 November 2026 under the all-owner takeover. No another-person approval or
real maintenance improvement is claimed.

Checks: `.venv/Scripts/python.exe -m ruff check .` PASS after replacing a
successive-pair zip with `itertools.pairwise`; focused
`.venv/Scripts/python.exe -m pytest -q tests/test_schedule_scoring.py tests/test_scheduling_jobs.py`
**47 passed in0.28s**; full `.venv/Scripts/python.exe -m pytest -q`
**556 passed, 1 skipped in106.28s**. The skip needs the absent P06 development
bundle. `git diff --check` PASS. Local Streamlit NOT RUN:no app changes.


## P18-A1 common deterministic decoder

`fleetpulse.optimization.decoder.decode_order(scenario, ordering)` accepts a
`SchedulingScenario` and an ordering of every selected `EngineIdentity` exactly
once. Numeric unit IDs alone are rejected because they lose dataset identity.
It assigns the next job to the earliest available team; availability ties use
the lowest one-based team ID. Every team starts at relative cycle zero. Jobs
are contiguous/non-preemptive and none are dropped for deadline/horizon misses.

The result is the shared `ScheduleEvaluation`: the generated assignments pass
`evaluate_schedule` before returning. Decoder code does not duplicate due-date,
objective, overflow, or feasibility formulas. It does not choose an EDD/GA
ordering. Input/scenario are unchanged. Ordering reads are bounded at n+1
values to reject oversized/infinite iterables; idle teams above the number of
jobs are not allocated in memory. Heap dispatch is O(n log min(n, team_count)).

```python
from fleetpulse.optimization.decoder import decode_order
from fleetpulse.optimization.jobs import appendix_b_jobs
from fleetpulse.optimization.scoring import SchedulingScenario

scenario = SchedulingScenario(appendix_b_jobs(), 1, 2, 6)
result = decode_order(scenario, tuple(job.engine for job in scenario.jobs))
assert result.weighted_tardiness_cycles == 6
```

Actual generated one-team A/B/C completions are3/4/6 with cost6; B/A/C
completions are1/4/6 with cost2. These reproduce the synthetic hand example
Abdullah checked/accepted in the timing/objective review on18Nov. With two
teams and A/B/C ordering, assignments are A team1[0,3), B team2[0,1),
C team2[1,3); cost1. At H2 all3jobs remain, twooverflow and aggregateoverflow1.
Independent tie tests dispatch equal-duration jobs to teams1,2,1,2. Invalid,
duplicate, missing, extra or wrong-partition orderings are rejected; repeated
calls are identical. These are software simulation outputs, not fleet results.

Verification: RuffPASS; focused
`.venv/Scripts/python.exe -m pytest -q tests/test_schedule_decoder.py tests/test_schedule_scoring.py tests/test_scheduling_jobs.py`
**60passed in0.46s**; full `.venv/Scripts/python.exe -m pytest -q`
**569passed,1skipped in118.00s**. The skip requires the absent P06development
bundle. `git diff --check`PASS; localStreamlit NOT RUN:noappchange.

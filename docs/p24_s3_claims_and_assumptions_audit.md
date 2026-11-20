# P24-S3 claims, assumptions, and shared understanding audit

Issue: [P24-S3 #231](https://github.com/Artfever/FleetpulseAI/issues/231)

This audit brings the P24 claims and their limits together with the scheduling assumptions used by earlier phases. It records observed evidence; it does not certify a production release, real-fleet performance, or maintenance readiness.

## Evidence and claims

| Claim | Source and check | Scope and limitation |
| --- | --- | --- |
| P24-A1 provides a reproducible clean setup. | [P24-A3 inventory](p24_a3_release_candidate.md) and the [P24-A1 handoff](../handoff_Abdullah.md) link the merged clean-install work. The reported fresh setup used Python 3.12.16, passed Ruff and 727 tests (7 skipped), and returned Streamlit health `ok`. | Local clean-clone evidence; it does not establish a hosted deployment check. |
| P24-A2 reproduced the declared Random Forest validation predictions and scores. | [P24-A2 reproduction](p24_a2_experiment_reproduction.md) records matching source/features/split hashes, all 20 validation predictions, MAE and RMSE. | Validation uses 20 held-out FD001/train engines. Runtime and serialized model bytes differed; neither byte-identical reproduction nor official-test evaluation is claimed. |
| P24-A3 identifies the frozen final candidate bundles. | [P24-A3 inventory](p24_a3_release_candidate.md) and `scripts/audit_p13_bundles.py` record 10 manifest/weight files across five model identities and their checksums. | `p24-rc1` is a candidate snapshot, not a completed P24 gate or production release. The audit checks identity, not full evaluation regeneration. |
| P24-S1 traces reported results to source rows, splits, and versions. | [P24-S1 result trace](p24_s1_result_trace.md) maps P13/P17/P20/P22/P24 outputs to reports or machine records. Its independent P13 audit joined 500 rows and recomputed all 10 MAE/RMSE aggregates with zero delta; the P22 report audit checked 60 claims, three figures, and 24 local links. | Some row-level artifacts are ignored local files and require source histories/bundles to regenerate. P23 hosted checks were waived as `NOT_PLANNED`, so public deployment is not established. |
| P24-S2 is a consistency recheck, not tuning. | [P24-S2 frozen evaluation](p24_s2_frozen_evaluation.md) reran the five frozen final bundles on FD001/test. Four models matched all predictions and scores. The final MLP differed on 43/100 rows by at most `0.0000152587890625` cycles; its MAE/RMSE moved by about `0.00000113`/`0.00000120` cycles. A second run in the current environment was exact. | No fitting or tuning occurred. The small difference against the prior saved MLP outputs has no established cause. FD001 is simulated data; it is not real-fleet or maintenance evidence. |

## Scheduler contract in plain language

- A prediction is an estimated number of remaining operating cycles. The planner uses that estimate as one input; it does not receive true RUL labels, residuals, or model-quality scores as planning inputs.
- A job's due time is its **completion** deadline: `max(0, planner RUL - buffer)`. A job is late only for the cycles it completes after that due time.
- Weighted tardiness is the sum of each job's late cycles multiplied by its priority weight. The weight is a scenario assumption, not a measured property of an engine.
- EDD orders jobs by earliest due date. The genetic algorithm (GA) searches candidate job orders using a fixed seed/configuration. A lower score in one run is an observed heuristic result, not proof of a global optimum or general superiority.
- The horizon is a soft reporting limit. Jobs beyond it remain in the schedule; overflow is reported separately and is not silently dropped or added to tardiness.
- Cycle-valued buffers, durations, horizon, team capacity, and dimensionless priority weights are simulation controls. In particular, the 120-cycle buffer is a stress case, not a recommendation.

The shared planner boundary and example outputs are documented in [P20 workflow gate evidence](p20_workflow_gate_evidence.md). The full fixed-fleet and seed results and their provenance are in [P22-A1](p22_a1_fixed_fleet_scenarios.md) and [P22-A2](p22_a2_seed_comparison.md). Their scoring constraints are also covered by the [course evidence note](COURSE_COMPLIANCE.md).

## Hand-check and schedule checks

P22-A2's small synthetic Appendix B case has one team, buffer 2 cycles, and horizon 6. Its three jobs have RULs 4/5/8 cycles, durations 3/1/2 cycles, and weights 1/5/1. The saved runner reproduced:

| Scheduler | Order | Weighted tardiness | Late jobs | Makespan | Overflow |
| --- | --- | ---: | ---: | ---: | ---: |
| EDD | 1, 2, 3 | 6 weighted cycles | 2 | 6 cycles | 0 cycles |
| GA, seed 42 | 2, 1, 3 | 2 weighted cycles | 1 | 6 cycles | 0 cycles |

Both schedules were accepted by the shared hard-feasibility validator. This is deliberately small, synthetic arithmetic; it does not show that GA will always improve a schedule.

The P22-A2 rerun also exercised all 18 declared fleet/team/buffer cases under five GA seeds (90 comparisons). It reported 60 ties, 30 improvements, zero regressions, all schedules feasible, and exact repeated results for each seed. The assumptions and raw predictions are held fixed; this check does not turn synthetic settings into operational inputs. P22-A1's separate 18-case run reported 12 ties and 6 improvements with seed 42.

## Shared understanding and review provenance

This document states the key distinctions for each member to use when describing the work: model estimates are not fault diagnoses; offline scores describe their named FD001 split; GA comparisons are limited to their declared scenarios and seeds; and scheduler settings are assumed inputs. The exact result lineage is in [P24-S1](p24_s1_result_trace.md), and the frozen evaluation method/limitation is in [P24-S2](p24_s2_frozen_evaluation.md).

Abdullah reported in the project conversation that Shaheer explicitly agreed to the P22 scenario list and benchmark machine, and that teammates checked the listed P24 result outputs. The P24-S2 acceptance reply did not identify which teammate, so this report does not assign that review to a named person or represent it as a GitHub approval. No teammate messages were sent for this audit. The scheduled task PR is subject to the repository's normal CI and conversation-resolution policy.

## Reproduction commands and observed results

Run from the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe scripts\run_p22_a1_fleet_scenarios.py
.\.venv\Scripts\python.exe scripts\run_p22_a2_seed_comparison.py
.\.venv\Scripts\python.exe scripts\check_p20_s2_scheduler_cases.py
.\.venv\Scripts\python.exe -m pytest -q tests\test_p22_a1_scenarios.py tests\test_p22_a2_seed_comparison.py
```

On 2026-11-21, both scenario runners exited successfully. A1 reported 18 cases (12 equal, 6 better). A2 reported 18 cases, 90 seeded comparisons (60 equal, 30 better), all schedules feasible, all repeated runs matching, and the Appendix B scores 6/2. The P20-S2 synthetic scenario check exited successfully and retained all four jobs with makespan 10 and horizon overflow 4 under both schedules. The focused pytest command passed: 4 tests.

Ruff and the full pytest suite were not run for this documentation-only audit. The relevant scenario runners and focused tests above were run; regular required CI applies to the PR. Streamlit smoke is not applicable because no application code changed.

## Remaining caveats

- NASA C-MAPSS FD001 is simulated engine data, not aircraft operating data.
- The P23 hosted explanation and upload checks were waived, not verified. P24 documentation does not change that status.
- P24 gate #29 remains open until its complete phase checklist is satisfied. This task does not close the gate or promote `main`.
- Human review provenance is reported as supplied by Abdullah; no unnamed teammate is presented as a named approver.

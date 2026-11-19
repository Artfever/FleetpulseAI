# P22-A2 EDD versus GA across fixed seeds

**Run date:** 2026-11-19

**Reproduction command:** `.venv/Scripts/python.exe scripts/run_p22_a2_seed_comparison.py`

**Result:** PASS; the script runs 18 fixed scenarios over five GA seeds (90 seeded comparisons), repeats every GA run to verify determinism, revalidates each returned schedule, and prints the complete JSON report.

## Scenario and provenance

This extends the P22-A1 matrix without changing its inputs: first N ascending engine IDs from the frozen FD001/test Decision Tree v1 raw prediction rows; fleet sizes 5, 10, and 20; team counts 1, 2, and 3; buffers 2 and 120 cycles; 30-cycle soft horizon; synthetic duration `1 + (engine_id mod 3)` cycles; synthetic weight `1 + ((engine_id * 7) mod 5)`; GA population 16 and 20 generations. GA seeds are fixed at **0, 7, 42, 2026, 2027**. EDD is deterministic and runs once for each case.

The loader reads only `unit_id` and `raw_prediction_cycles` from ignored local `artifacts/final/p13/final_test_results.json`. It never uses true RUL, residuals, or model quality metrics for planning. Report SHA-256 is `db420935dc14221c50f71f0ca3349b0667bea6b37df45b330026df3d3e8d40d4`; Decision Tree v1 manifest SHA-256 is `928428a1cd81e24922f55518aacc6dfb25b98d74fbc6df542ce72c5c179247cf`; weights SHA-256 is `c33021cb9cf5931aba545b6a327d0547f14be1363cbd0c24d57f0ad923dd7bb3`.

Duration, priority, buffer, team count and horizon are simulation assumptions. The 120-cycle buffer is a deliberate stress setting; it is not a maintenance recommendation. RUL predictions are from NASA's simulated FD001 dataset, not evidence about aircraft or real maintenance outcomes.

## Aggregate results

Every EDD and GA schedule passed the shared hard-feasibility validator: all selected engines scheduled once, no overlapping jobs per team. Every GA result repeated exactly for the same seed (order, schedule and generation score history). Across 90 comparisons, GA tied EDD **60** times, improved weighted tardiness **30** times, and was worse **0** times. All 30 improvements occurred under the declared 120-cycle stress buffer for the 10- and 20-engine cases. All 45 runs with the 2-cycle buffer tied at zero tardiness; all 15 five-engine stress-buffer runs tied as well.

The complete per-seed weighted tardiness, late-job counts, feasibility, overflow, order, runtime and evaluation counts are emitted by the command. The summary below records each observed EDD score and every GA seed's weighted tardiness/late-job count; all schedules were feasible. GA overflow is shown as a range across seeds, and runtime as the observed minimum-to-maximum seconds on the benchmark machine.

| Buffer | Engines | Teams | EDD: weighted / late / feasible / overflow / runtime s | GA: seed -> weighted / late (all feasible) | Better / equal | GA overflow range | GA runtime range s |
|---:|---:|---:|---|---|---:|---:|---:|
| 2 | 5 | 1 | 0.000 / 0 / True / 0.0 / 0.000089 | 0 -> 0.000/0; 7 -> 0.000/0; 42 -> 0.000/0; 2026 -> 0.000/0; 2027 -> 0.000/0 | 0 / 5 | 0.0-0.0 | 0.0055-0.0075 |
| 2 | 5 | 2 | 0.000 / 0 / True / 0.0 / 0.000052 | 0 -> 0.000/0; 7 -> 0.000/0; 42 -> 0.000/0; 2026 -> 0.000/0; 2027 -> 0.000/0 | 0 / 5 | 0.0-0.0 | 0.0056-0.0072 |
| 2 | 5 | 3 | 0.000 / 0 / True / 0.0 / 0.000049 | 0 -> 0.000/0; 7 -> 0.000/0; 42 -> 0.000/0; 2026 -> 0.000/0; 2027 -> 0.000/0 | 0 / 5 | 0.0-0.0 | 0.0057-0.0065 |
| 120 | 5 | 1 | 20.000 / 3 / True / 0.0 / 0.000048 | 0 -> 20.000/3; 7 -> 20.000/3; 42 -> 20.000/3; 2026 -> 20.000/3; 2027 -> 20.000/3 | 0 / 5 | 0.0-0.0 | 0.0042-0.0052 |
| 120 | 5 | 2 | 14.000 / 3 / True / 0.0 / 0.000049 | 0 -> 14.000/3; 7 -> 14.000/3; 42 -> 14.000/3; 2026 -> 14.000/3; 2027 -> 14.000/3 | 0 / 5 | 0.0-0.0 | 0.0043-0.0046 |
| 120 | 5 | 3 | 13.000 / 3 / True / 0.0 / 0.000048 | 0 -> 13.000/3; 7 -> 13.000/3; 42 -> 13.000/3; 2026 -> 13.000/3; 2027 -> 13.000/3 | 0 / 5 | 0.0-0.0 | 0.0045-0.0052 |
| 2 | 10 | 1 | 0.000 / 0 / True / 0.0 / 0.000084 | 0 -> 0.000/0; 7 -> 0.000/0; 42 -> 0.000/0; 2026 -> 0.000/0; 2027 -> 0.000/0 | 0 / 5 | 0.0-0.0 | 0.0181-0.0258 |
| 2 | 10 | 2 | 0.000 / 0 / True / 0.0 / 0.000145 | 0 -> 0.000/0; 7 -> 0.000/0; 42 -> 0.000/0; 2026 -> 0.000/0; 2027 -> 0.000/0 | 0 / 5 | 0.0-0.0 | 0.0196-0.0287 |
| 2 | 10 | 3 | 0.000 / 0 / True / 0.0 / 0.000083 | 0 -> 0.000/0; 7 -> 0.000/0; 42 -> 0.000/0; 2026 -> 0.000/0; 2027 -> 0.000/0 | 0 / 5 | 0.0-0.0 | 0.0209-0.0226 |
| 120 | 10 | 1 | 85.102 / 7 / True / 0.0 / 0.000085 | 0 -> 61.600/6; 7 -> 61.600/6; 42 -> 61.600/6; 2026 -> 65.600/6; 2027 -> 67.000/5 | 5 / 0 | 0.0-0.0 | 0.0097-0.0143 |
| 120 | 10 | 2 | 40.000 / 5 / True / 0.0 / 0.000169 | 0 -> 34.000/5; 7 -> 34.000/5; 42 -> 35.000/5; 2026 -> 34.000/5; 2027 -> 34.000/5 | 5 / 0 | 0.0-0.0 | 0.0114-0.0147 |
| 120 | 10 | 3 | 32.000 / 5 / True / 0.0 / 0.000089 | 0 -> 28.000/5; 7 -> 28.000/5; 42 -> 28.000/5; 2026 -> 28.000/5; 2027 -> 28.000/5 | 5 / 0 | 0.0-0.0 | 0.0111-0.0125 |
| 2 | 20 | 1 | 0.000 / 0 / True / 11.0 / 0.000152 | 0 -> 0.000/0; 7 -> 0.000/0; 42 -> 0.000/0; 2026 -> 0.000/0; 2027 -> 0.000/0 | 0 / 5 | 11.0-11.0 | 0.0404-0.0465 |
| 2 | 20 | 2 | 0.000 / 0 / True / 0.0 / 0.000154 | 0 -> 0.000/0; 7 -> 0.000/0; 42 -> 0.000/0; 2026 -> 0.000/0; 2027 -> 0.000/0 | 0 / 5 | 0.0-0.0 | 0.0386-0.0462 |
| 2 | 20 | 3 | 0.000 / 0 / True / 0.0 / 0.000153 | 0 -> 0.000/0; 7 -> 0.000/0; 42 -> 0.000/0; 2026 -> 0.000/0; 2027 -> 0.000/0 | 0 / 5 | 0.0-0.0 | 0.0429-0.0469 |
| 120 | 20 | 1 | 736.729 / 16 / True / 11.0 / 0.000304 | 0 -> 541.501/14; 7 -> 512.729/16; 42 -> 525.002/15; 2026 -> 537.002/15; 2027 -> 552.729/16 | 5 / 0 | 11.0-11.0 | 0.0253-0.0285 |
| 120 | 20 | 2 | 311.002 / 15 / True / 0.0 / 0.000157 | 0 -> 236.002/15; 7 -> 225.102/14; 42 -> 239.002/15; 2026 -> 251.002/15; 2027 -> 246.002/15 | 5 / 0 | 0.0-0.0 | 0.0247-0.0286 |
| 120 | 20 | 3 | 189.501 / 14 / True / 0.0 / 0.000153 | 0 -> 151.501/14; 7 -> 152.501/14; 42 -> 156.501/14; 2026 -> 156.000/12; 2027 -> 156.000/12 | 5 / 0 | 0.0-0.0 | 0.0212-0.0257 |
## Hand-calculated reference and limitations

The synthetic Appendix B check used three jobs with RUL 4/5/8 cycles, durations 3/1/2, weights 1/5/1, one team, buffer 2 and horizon 6. It reproduced EDD order 1/2/3 with weighted tardiness 6, two late jobs, makespan 6 and zero overflow. GA seed 42 returned order 2/1/3 with weighted tardiness 2, one late job, makespan 6 and zero overflow. This is a toy scoring check, not a real fleet result.

The reported GA search is a bounded heuristic. Lower weighted tardiness here does not establish global optimality or operational value. The 2-cycle baseline produces zero tardiness for these predictions, while the deliberately high 120-cycle buffer creates comparison pressure. In this run, observed EDD runtimes were 0.000048-0.000304 seconds and GA runtimes ranged 0.004205-0.046862 seconds; these machine-specific micro-benchmarks are not a throughput claim. The scenarios use a selected subset of FD001/test predictions and do not measure model performance; no truth labels entered either scheduler.

Benchmark machine is Windows 11 Pro build 26200, Python 3.12.16, Intel Core i5-1335U, 12 logical processors, 16 GiB RAM. Runs are serial. Shaheer's agreement to this scenario list and benchmark machine was reported by Abdullah for P22-A1; this P22-A2 analysis reuses that exact setup. Abdullah also reported that a teammate checked the P22-A1 full results.

## Changed files and checks

- `scripts/run_p22_a2_seed_comparison.py`: fixed-seed runner with feasibility revalidation and repeat checks.
- `tests/test_p22_a2_seed_comparison.py`: synthetic fixture covers all 90 comparisons and the hand-score example.
- `docs/p22_a2_seed_comparison.md`: full per-case/per-seed outcome summary, assumptions, provenance and limitations.
- `handoff_Abdullah.md`: current task and evidence.

Checks on this revision:

- `.venv/Scripts/python.exe -m ruff check .` — PASS.
- `.venv/Scripts/python.exe -m pytest -q` — PASS (695 tests in 103.37s).
- `.venv/Scripts/python.exe scripts/run_p22_a2_seed_comparison.py` — PASS (18 cases, 90 GA comparisons, 60 equal/30 better/0 worse; all feasible, all repeats matched; Appendix B 6/2).
- Streamlit smoke — NOT RUN; no app code changed.
- `git diff --check` — PASS.

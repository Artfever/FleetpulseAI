# P22-A1 fixed fleet and team-count scenarios

**Run date:** 2026-11-19

**Reproduction command:** `.venv/Scripts/python.exe scripts/run_p22_a1_fleet_scenarios.py`
**Result:** PASS; the command runs all 18 cases and prints a JSON report including every schedule order and provenance hash.

## Purpose and inputs

This experiment exercises the existing EDD and seeded genetic scheduler on fixed 5-, 10-, and 20-engine fleets, each with one, two, and three teams. It uses the final P13 `decision-tree-rul-v1` raw predictions for the first N ascending engine IDs in `FD001/test`. The experiment code reads only each selected `unit_id` and `raw_prediction_cycles`; it never uses true RUL, residuals, or evaluation metrics as scheduling inputs.

The actual input report is `artifacts/final/p13/final_test_results.json` (ignored local evidence), SHA-256 `db420935dc14221c50f71f0ca3349b0667bea6b37df45b330026df3d3e8d40d4`. The selected Decision Tree manifest SHA-256 is `928428a1cd81e24922f55518aacc6dfb25b98d74fbc6df542ce72c5c179247cf`; weights SHA-256 is `c33021cb9cf5931aba545b6a327d0547f14be1363cbd0c24d57f0ad923dd7bb3`.

Each case uses the same saved prediction for an engine. Durations and priority weights are deterministic synthetic assumptions:

- Duration in cycles: `1 + (engine_id mod 3)`.
- Dimensionless priority weight: `1 + ((engine_id * 7) mod 5)`.
- Buffer levels: 2 cycles (baseline matching the existing planner example) and 120 cycles (deliberately high stress case to expose scheduling differences).
- Soft horizon: 30 cycles. Work beyond it stays scheduled and is reported as overflow.
- GA configuration: population 16, 20 generations, seed 42. EDD seeds the GA population, so its final score cannot be worse than EDD for the same case.

The 120-cycle buffer is a stress assumption, not a maintenance recommendation. All durations, weights, buffers, team counts, and horizon are scenario controls, not values supplied by NASA C-MAPSS or evidence about real maintenance operations.

## Observed paired results

Weighted tardiness is measured in weighted cycles. The makespan and horizon overflow columns show `EDD/GA`, in cycles.

| Buffer | Engines | Teams | EDD weighted tardiness | GA weighted tardiness | GA outcome | EDD/GA makespan | EDD/GA horizon overflow |
|---:|---:|---:|---:|---:|---|---:|---:|
| 2 | 5 | 1 | 0.000 | 0.000 | equal | 11/11 | 0.0/0.0 |
| 2 | 5 | 2 | 0.000 | 0.000 | equal | 6/6 | 0.0/0.0 |
| 2 | 5 | 3 | 0.000 | 0.000 | equal | 4/4 | 0.0/0.0 |
| 120 | 5 | 1 | 20.000 | 20.000 | equal | 11/11 | 0.0/0.0 |
| 120 | 5 | 2 | 14.000 | 14.000 | equal | 6/6 | 0.0/0.0 |
| 120 | 5 | 3 | 13.000 | 13.000 | equal | 4/4 | 0.0/0.0 |
| 2 | 10 | 1 | 0.000 | 0.000 | equal | 20/20 | 0.0/0.0 |
| 2 | 10 | 2 | 0.000 | 0.000 | equal | 11/11 | 0.0/0.0 |
| 2 | 10 | 3 | 0.000 | 0.000 | equal | 8/8 | 0.0/0.0 |
| 120 | 10 | 1 | 85.102 | 61.600 | better | 20/20 | 0.0/0.0 |
| 120 | 10 | 2 | 40.000 | 35.000 | better | 11/11 | 0.0/0.0 |
| 120 | 10 | 3 | 32.000 | 28.000 | better | 8/8 | 0.0/0.0 |
| 2 | 20 | 1 | 0.000 | 0.000 | equal | 41/41 | 11.0/11.0 |
| 2 | 20 | 2 | 0.000 | 0.000 | equal | 21/21 | 0.0/0.0 |
| 2 | 20 | 3 | 0.000 | 0.000 | equal | 14/14 | 0.0/0.0 |
| 120 | 20 | 1 | 736.729 | 525.002 | better | 41/41 | 11.0/11.0 |
| 120 | 20 | 2 | 311.002 | 239.002 | better | 21/21 | 0.0/0.0 |
| 120 | 20 | 3 | 189.501 | 156.501 | better | 14/14 | 0.0/0.0 |

Across all 18 cases, GA tied EDD 12 times and improved weighted tardiness 6 times; it was worse zero times. At the 2-cycle buffer all nine results tied at zero tardiness. The stress buffer produced three ties and six improvements. The results show that this seeded GA can help under some declared synthetic cases; they do not establish general superiority. Increasing teams reduced makespan, but did not always reduce weighted tardiness. The 20-engine, one-team case exceeded the 30-cycle soft horizon by 11 cycles under both schedulers; no jobs were removed.

## Benchmark environment

- Windows 11 Pro, build 26200.
- Python 3.12.16.
- Intel Core i5-1335U, 12 logical processors, 16 GiB installed RAM.
- All cases ran serially on this machine. GA elapsed times and exact orders are included in the command's JSON output; elapsed times are machine-specific and are not used as a quality claim.

The proposed scenario list and this benchmark machine are documented for Shaheer's review; no direct teammate confirmation has been recorded in this handoff yet. The PR's CI can verify the synthetic fixture and code, but it cannot reproduce the ignored P13 artifact in a clean checkout. A teammate should inspect this report and the original task requirement before #93 is closed.

## Changed files and checks

- `scripts/run_p22_a1_fleet_scenarios.py`: reproducible matrix runner and source-hash report.
- `tests/test_p22_a1_scenarios.py`: synthetic fixture verifies all fleet/team/buffer combinations, no-GA-regression invariant, and that true labels are not required by the input loader.
- `docs/p22_a1_fixed_fleet_scenarios.md`: assumptions, provenance, full observed results, and limits.

Checks run on this revision:

- `.venv/Scripts/python.exe -m ruff check scripts/run_p22_a1_fleet_scenarios.py tests/test_p22_a1_scenarios.py` — PASS.
- `.venv/Scripts/python.exe -m pytest -q tests/test_p22_a1_scenarios.py` — PASS (2 tests).
- `.venv/Scripts/python.exe scripts/run_p22_a1_fleet_scenarios.py` — PASS (18 scenarios; 12 ties, 6 GA improvements, 0 regressions).
- `.venv/Scripts/python.exe -m ruff check .` — PASS.
- `.venv/Scripts/python.exe -m pytest -q` — PASS (693 tests in 112.30s).
- Streamlit smoke — NOT RUN; no application code changed.

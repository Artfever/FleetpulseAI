# P22 saved study views

For a concise, report-ready set of observed P22 results with labeled tables, the three S1 figures, and claim-to-artifact traceability, see [P22 report-ready results](p22_report_results.md).

The Model Laboratory and Maintenance Planner load the committed P22 study outputs as read-only evidence. They do not run training, change model inputs or recalculate a saved result. The Planner's saved-study section is visually separated from its current selected-fleet and synthetic-scenario controls.

## Sources displayed

- **P22-S1:** P13 frozen FD001/test model summaries and the actual-vs-predicted, residual, and true-RUL-bin SVG figures under `docs/figures/`. Each row is one of 100 engines at its final observed cycle. The bins are descriptive analysis ranges.
- **P22-S2:** the five development candidates' validation MAE values and whole-engine bootstrap intervals from `docs/p22_s2_validation_bootstrap.json`. The intervals describe sampling variability of aggregate MAE across the 20 validation engines conditional on saved fits; they are not individual RUL intervals.
- **P22-S3:** aggregate raw prediction movement by saved model and ±1%/±5% sensor change from `docs/p22_s3_sensor_sensitivity.json`. Changes are artificial and the resulting movement is not accuracy, noise, fault, or causal evidence.
- **P22-A1:** 18 EDD/seeded-GA cases for 5/10/20-engine fleets and 1/2/3 teams at 2- and 120-cycle buffers. Source predictions are frozen P13 Decision Tree raw outputs. Durations, weights, buffers, team counts and horizon are simulation assumptions.
- **P22-A2:** the same 18 paired scenarios and five GA seeds (90 comparisons). The UI shows the seed-specific weighted tardiness values and feasibility. It does not call the heuristic globally optimal or generally superior.

The scheduler snapshots are checksummed envelopes in `docs/p22_a1_fixed_fleet_scenarios.json` and `docs/p22_a2_seed_comparison.json`. Their reproducible materialization command is:

```powershell
.venv/Scripts/python.exe scripts/export_p22_dashboard_studies.py --output-dir docs
```

It requires the local, ignored P13 report at `artifacts/final/p13/final_test_results.json`, verifies its frozen SHA-256, and consumes only engine IDs and raw Decision Tree predictions. No official test labels or error metrics enter scheduling. The application validates each envelope, source report hash, model version, schema and expected case counts before display.

When a saved study file or figure is missing or incompatible, its view says the evidence is unavailable. It does not substitute an estimate. All displayed values remain labeled with their split, model version, units, and study assumptions.

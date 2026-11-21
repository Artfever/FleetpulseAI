# P26-S1 AI evidence inventory

Issue: [P26-S1 #106](https://github.com/Artfever/AI_PROJECT/issues/106)

This inventory checks the tracked files that support the final model, evaluation,
data provenance, and scheduling claims. Every file in the tables below is tracked
by Git and therefore included in a source archive made from the tracked tree.

## Prediction outputs and model bundles

| Evidence | Tracked files | What is present |
| --- | --- | --- |
| Final test prediction/error table | `artifacts/final/p13/final_test_results.json` | Five models × 100 FD001/test engines; per-engine observed cycle, raw prediction, true RUL, residual, absolute/squared error, and aggregate MAE/RMSE |
| Tree and constant bundles | All files in `artifacts/final/tree/bundles/` | Three model payloads, four manifests (including constant baseline), and bundle index |
| Final MLP bundle | `artifacts/final/mlp/bundle/final_mlp_manifest.json`, `artifacts/final/mlp/bundle/final_mlp_state.pt` | Saved architecture/configuration and model state |
| Bundle checksums and feature contract | `docs/artifact_inventory.md`, the five manifests, and `final_tree_bundle_index.json` | Model versions, payload/weight hashes, training/configuration provenance, and `past-only-v1` 30-cycle/96-feature contract |

The tracked final-result file has whole-file SHA-256
`12f63b73f763cac0b70d133a3b1dd63c06422adf81e4900698383c0037b7925f` and
verified payload SHA-256 `2a0686eaa3d369c435938cf6bf6256b047b3265f8ac8893e08e040adc798a6a6`.
`docs/p13_s3_final_test_results.md` now distinguishes this tracked JSON from
local-only intermediate prediction/pairing files and standalone HTML figures.

## Figures and robustness studies

| Evidence | Tracked files | Scope |
| --- | --- | --- |
| Official FD001/test diagnostics | `docs/figures/p22_s1_actual_vs_predicted.svg`, `p22_s1_residuals.svg`, `p22_s1_rul_bins.svg`, `p22_s1_error_analysis.json` | Five frozen final models; 100 simulated test engines; predictions/errors and analysis bins |
| Validation MAE variability | `docs/p22_s2_validation_bootstrap.json` | Whole-engine bootstrap over the 20 held-out FD001/train validation engines |
| Artificial sensor sensitivity | `docs/p22_s3_sensor_sensitivity.json` | One-channel perturbation study on held-out FD001/train histories |
| Reported results and captions | `docs/p22_report_results.md`, `docs/report/p25_s2_evidence_captions.md` | Figures, source outputs, units, assumptions, and interpretation limits |

## Data and model-selection provenance

| Evidence | Tracked files | Boundary |
| --- | --- | --- |
| Raw source and checksums | `docs/fd001_provenance.md` | NASA C-MAPSS FD001 origin and expected train/test/RUL SHA-256 values; raw files are not included |
| Data-quality and split records | `artifacts/data_quality.json`, `artifacts/split_manifest.json`, `docs/split_manifest.md` | Engine-level fitting/validation split and source provenance |
| Model-selection decision | `artifacts/selection_manifest.json`, `docs/p11_t2_selection_manifest.md` | Frozen candidate/configuration rationale; official test results remain final-reporting-only |
| Result lineage | `docs/p24_s1_result_trace.md`, `docs/p13_s1_test_predictions.md`, `docs/p13_s2_test_label_pairing.md`, `docs/p13_s3_final_test_results.md` | Source/model hashes, sorted engine-ID pairing, metric definitions, and limits |

## Scheduler scenario evidence

- `docs/p22_a1_fixed_fleet_scenarios.json` contains the 18 paired EDD/GA
  scenarios; `docs/p22_a2_seed_comparison.json` records the 90 seeded paired
  comparisons. Their reports identify duration, priority, team, buffer, horizon,
  and GA settings as simulation assumptions.
- `tests/fixtures/p20_s2_capacity_pressure.json` and
  `docs/p20_s2_scheduler_edge_cases.md` preserve the small synthetic capacity
  example. `docs/p20_s3_paired_comparison_artifact.md` documents the provenance
  fields and checks for saved-model planner comparisons.
- The scheduler inputs and outputs do not use true RUL labels. These comparisons
  demonstrate software behavior under declared assumptions, not real maintenance
  policy or model quality.

## Checks performed

- `.venv/Scripts/python.exe scripts/audit_p13_metrics.py` recomputed all five
  models' metrics from saved predictions and held-out labels: 100 engine rows
  per model, zero MAE/RMSE deltas, no tuning.
- `.venv/Scripts/python.exe scripts/audit_p22_report_results.py` passed: 60
  report claims checked, three labeled SVGs and 24 local links verified.
- `.venv/Scripts/python.exe scripts/smoke_p12_final_bundles_clean_process.py`
  previously loaded all five frozen bundles in separate processes on the same
  FD001/train engine 4 prefix through cycle 164, without fitting. Outputs are
  recorded in [`p26_a1_runnable_package.md`](p26_a1_runnable_package.md).
- The exact evidence file paths above were checked with `git ls-files`; model,
  manifest, and study checksums are recorded in
  [`artifact_inventory.md`](artifact_inventory.md), their JSON envelopes, or
  this repository history.

## Not included and limitations

The final result JSON preserves all 500 per-engine rows, but its original
`official_test_predictions.json`, `official_test_pairings.json`, and standalone
Plotly HTML files are local/ignored intermediates. P22's tracked SVGs and JSON
provide the included figures and diagnostics. Full raw FD001 files remain
untracked; their source and verification instructions are in
`docs/fd001_provenance.md`. The exact assembled submission ZIP has not yet been
inspected; P26-A3 performs that package-level check after Taha's archive exists.

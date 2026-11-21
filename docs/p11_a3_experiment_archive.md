# P11-A3: retained run archive and timing comparison

## Scope and audit

This closeout inventories the 12 saved P11-S1, P11-A1, and P11-A2 runs. Every
run retains its `results.json`, serialized model, parameters, timing, metrics,
and per-engine validation predictions in its own ignored experiment folder.
Weaker configurations were kept. All 12 runs completed; no execution failure
occurred in this declared budget, so there is no failed model run to report.

The audit command is:

```powershell
.venv/Scripts/python.exe scripts/audit_p11_experiment_runs.py
```

It requires the saved experiment directories from the three upstream scripts
and checks expected run counts, source provenance, model-file hashes, the 20
validation identities/cutoffs, and fresh-process prediction replay. It wrote
`artifacts/experiments/P11_A3_run_index.json` locally with 12 runs, 0 failures,
and exact 20/20 replay for every saved model. Generated models and that index
are Git-ignored; this tracked document records the observed summary.

## All retained runs

| Group/configuration | MAE (cycles) | RMSE (cycles) | Fit + predict (s) |
| --- | ---: | ---: | ---: |
| S1 Decision Tree, 10-cycle window | 23.710807 | 31.458965 | 0.499 |
| S1 Decision Tree, 20-cycle window | 23.478525 | 32.424615 | 0.492 |
| S1 Decision Tree, 30-cycle window | 22.494270 | 29.449574 | 0.533 |
| A1 Random Forest, depth 8 | 18.845563 | 24.877649 | 28.328 |
| A1 Random Forest, depth 12 | 18.833479 | 25.243023 | 46.708 |
| A1 Random Forest, depth 16 | 18.360242 | 24.563262 | 37.547 |
| A1 XGBoost, depth 3 | 16.063677 | 21.616009 | 1.463 |
| A1 XGBoost, depth 4 | **15.108697** | **21.092113** | 1.373 |
| A1 XGBoost, depth 6 | 17.008195 | 23.231008 | 2.411 |
| A2 XGBoost, full channels | 15.108697 | 21.092113 | 1.415 |
| A2 XGBoost, sensors only | 16.718605 | 21.282901 | 1.020 |
| A2 XGBoost, settings only | 53.064679 | 59.503527 | 0.237 |

All rows scored the same 20 FD001/train development-validation engines under
the saved P04 cutoffs. Runtimes are single-machine observations and should not
be compared as portable speed benchmarks. The weaker tree-depth and feature
ablation runs remain saved and visible rather than being discarded. The best
observed metrics belong to XGBoost depth 4 with full channels, but P11's final
model/configuration decision also depends on other P11 tasks and the shared
gate. These scores are not official-test performance.

## Provenance and limits

Python 3.12.10, scikit-learn 1.9.1, XGBoost 3.4.1. Source and split are
FD001/train with a fixed 80-engine fitting partition and 20 saved validation
prefixes. Validation was repeatedly inspected while conducting the declared
P11 development experiments; its scores are therefore selection evidence, not
an independent estimate of final generalization. No official test histories
or RUL answers were opened. The exact per-run configuration, hashes, and
predictions are stored in each run's ignored `results.json`.

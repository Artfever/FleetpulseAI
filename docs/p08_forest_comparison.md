# P08-T3 forest scores in Model Laboratory

**Issue:** [P08-T3 #247](https://github.com/Artfever/FleetpulseAI/issues/247)

The Model Laboratory now adds Random Forest MAE and RMSE to the same table as
the constant baseline and Decision Tree only after validating the saved forest
bundle against the exact fitting and validation feature-table hashes. It
reloads the forest and reproduces all saved validation predictions before
showing aggregate scores. A missing or invalid forest bundle leaves the
baseline/tree comparison available and explains why no forest score is shown.

## Observed comparison

Local AppTest on 2026-11-16 used the saved P08 Random Forest bundle and the
verified FD001/train development split. The table displayed:

| Model | MAE (cycles) | RMSE (cycles) | Samples |
| --- | ---: | ---: | ---: |
| Constant mean baseline | 65.728002 | 74.258987 | 20 engines |
| Decision Tree | 22.494270 | 29.449574 | 20 engines |
| Random Forest (`random-forest-rul-v1`) | 18.833479 | 25.243023 | 20 engines |

Python was 3.12.16 and scikit-learn was 1.9.1. The forest uses seed 42, 100
estimators, depth 12, minimum leaf size 5, and the existing 30-cycle,
`past-only-v1` feature contract. Its saved-output audit independently
recomputes these metrics from 20 unique validation engine/cutoff rows. The
verified split and model provenance are documented in
[`p08_forest_validation_audit.md`](p08_forest_validation_audit.md).

The forest has the lowest observed error among these three models on this
development-validation split. That result is limited to these 20 engines; it
does not establish official-test performance or predict future performance.
Official test engines and answers were not used. The comparison chart still
plots the baseline and tree predictions; the Random Forest's verified
aggregate metrics are shown in the table.

## Verification

- `pytest -q tests/test_model_comparison.py tests/test_model_laboratory_page.py` checks matching split hashes, verified model identity/metrics, mismatched-hash rejection, missing-bundle behavior, and UI table contents.
- Local AppTest verified the full table from the ignored saved development artifacts. The raw model and feature bundles remain Git-ignored.

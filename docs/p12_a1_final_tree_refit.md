# P12-A1: final tree and constant refit

The P11 candidate configurations were refit without further tuning on every
cycle row from all 100 original FD001/train engines: 20,631 rows with the
frozen 30-cycle, 24-channel, 96-feature `past-only-v1` input. Targets are
uncapped remaining cycles derived from each complete training-engine lifetime.
No official test histories or answers were read.

Frozen model settings:

- Decision Tree: version `decision-tree-rul-v1`, seed 42, max depth 8, minimum
  leaf size 5.
- Random Forest: version `random-forest-rul-v1`, seed 42, 100 estimators,
  max depth 16, minimum leaf size 5, all features, bootstrap, one CPU worker.
- XGBoost: version `xgboost-rul-v1`, seed 42, 100 estimators, max depth 4,
  learning rate 0.05, child weight 1, row/column sample 0.8, lambda 1, CPU
  histogram method, one worker.
- Constant baseline: the previously selected mean statistic, now refit from
  all 20,631 original training labels; no random seed applies.

## Reproduce

With `data/raw/train_FD001.txt` and the tracked P11 selection manifest present:

```powershell
.venv/Scripts/python.exe scripts/refit_p12_tree_models.py
```

The ignored outputs are in `artifacts/final/tree/refit/`. Each tree model is
saved as a joblib state and reloaded to require an exact prediction match on
FD001/train engine 4 at cycle 164. The constant baseline JSON is hash-checked
and round-tripped through its versioned loader. These training-row predictions
are serialization checks, not evaluation metrics. P12-A2/#140 and P12-A3/#204
consume these refit artifacts to prepare and package the final bundles.

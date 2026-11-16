# P09-T2: one bounded XGBoost experiment

## Protocol and input evidence

Ran exactly one fixed configuration through `scripts/run_xgboost_experiment.py`.
The runner loads the checked common feature tables and calls `train_xgboost`
once. It performs no parameter search, early stopping, test-set scoring, or
refit. The earlier read-only input audit passed before the fit:

- FD001/train source SHA-256:
  `963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8`.
- Semantic split-manifest SHA-256:
  `6fd1d32f7570f70f5a9513d900fbb9d26b182359aced425e2276700c84f08d92`.
- Fitting: 16,738 rows from 80 engines. Every fitting label was checked against
  the complete corresponding FD001/train lifetime.
- Validation: 20 held-out FD001/train engine prefixes, each identity and target
  matched the manifest and each feature vector was rebuilt from its raw prefix.
- Feature contract: `past-only-v1`, 30-cycle window, 96 ordered features.
- No official test histories or official RUL answers were opened.

The single run used Python 3.12.10 and XGBoost 3.4.1. Parameters: seed 42,
100 estimators, max depth 4, learning rate 0.05, minimum child weight 1.0,
subsample 0.8, column sample by tree 0.8, L2 lambda 1.0, squared-error
objective, RMSE evaluation metric, histogram tree method, CPU device, one
worker, verbosity 0. Fit plus validation prediction took 1.592586 seconds on
this machine.

## Observed development-validation results

MAE was **15.108697 cycles** and RMSE was **21.092113 cycles** over 20 distinct
held-out FD001/train engines. These are development-validation measurements,
not official-test or future performance claims.

| Engine | Cutoff cycle | Predicted RUL (cycles) | Validation target (cycles) | Absolute error (cycles) |
|---:|---:|---:|---:|---:|
| 4 | 164 | 28.693811 | 25 | 3.693811 |
| 5 | 58 | 171.816452 | 211 | 39.183548 |
| 12 | 7 | 174.498779 | 163 | 11.498779 |
| 14 | 71 | 138.884674 | 109 | 29.884674 |
| 15 | 63 | 163.097031 | 144 | 19.097031 |
| 18 | 58 | 146.108902 | 137 | 9.108902 |
| 28 | 36 | 168.230240 | 129 | 39.230240 |
| 29 | 27 | 193.679367 | 136 | 57.679367 |
| 30 | 174 | 22.759937 | 20 | 2.759937 |
| 32 | 190 | 15.224778 | 1 | 14.224778 |
| 36 | 140 | 16.509903 | 18 | 1.490097 |
| 55 | 23 | 162.345245 | 170 | 7.654755 |
| 65 | 152 | 7.819873 | 1 | 6.819873 |
| 70 | 109 | 22.909716 | 28 | 5.090284 |
| 72 | 9 | 207.913651 | 204 | 3.913651 |
| 76 | 8 | 190.372772 | 202 | 11.627228 |
| 78 | 24 | 202.232971 | 207 | 4.767029 |
| 82 | 56 | 176.896057 | 158 | 18.896057 |
| 87 | 60 | 107.467201 | 118 | 10.532799 |
| 95 | 259 | 29.021093 | 24 | 5.021093 |

The largest absolute error in this run was 57.679367 cycles for engine 29;
engine 5 was underpredicted by 39.183548 cycles and engine 28 was overpredicted
by 39.230240 cycles. The other predictions also vary by engine and observed
prefix. A single fixed run shows that the trainer executes and provides
traceable outputs; it does not establish stable accuracy, superiority to
another model, or maintenance readiness. The validation set contains only 20
engines from the FD001 training partition, and this configuration was not
tuned.

## Reproduction and saved run record

```powershell
.venv/Scripts/python.exe scripts/audit_xgboost_inputs.py
.venv/Scripts/python.exe scripts/run_xgboost_experiment.py
```

The exact JSON run record, including all prediction fields, source/split/table
hashes, parameters, metrics, runtime and package versions, is
`reports/P09_T2_xgboost_run.json`. It is intentionally Git-ignored with local
reports and model outputs; this tracked document preserves the observed values.

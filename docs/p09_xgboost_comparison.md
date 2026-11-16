# P09-S2: interpretation of the P09 validation comparison

All three candidates were compared on the same 20 held-out FD001/train engine
prefixes, one saved cutoff per engine. The independent audit found lower
aggregate error for XGBoost on this particular development-validation split:

| Candidate | MAE (cycles) | RMSE (cycles) |
| --- | ---: | ---: |
| Decision Tree | 22.494270 | 29.449574 |
| Random Forest | 18.833479 | 25.243023 |
| XGBoost | 15.108697 | 21.092113 |

That aggregate result does not mean boosting was better for every engine.
On 7 of 20 engines, XGBoost had greater absolute error than the Random Forest;
on 6 of 20 it had greater error than the Decision Tree. For example, engine 32
at cycle 190 has a validation target of 1 remaining cycle. Predictions and
absolute errors were:

| Candidate | Prediction (cycles) | Absolute error (cycles) |
| --- | ---: | ---: |
| Decision Tree | 4.201923 | 3.201923 |
| Random Forest | 2.929698 | 1.929698 |
| XGBoost | 15.224778 | 14.224778 |

There are examples where XGBoost does better too. For engine 4 at cycle 164,
with target 25 cycles, its error was 3.693811 cycles versus 9.196078 for the
Decision Tree and 9.043205 for the Random Forest. These examples illustrate
per-engine variation; neither one establishes which model will work better on
future engines.

The metrics and examples use the single fixed candidate runs, identical saved
validation cutoffs and the FD001 training partition. They are not official-test
results, evidence of performance on future data, or a reason to claim that
XGBoost is universally superior. The validation set has only 20 engines and no
hyperparameter search was done for this comparison. All scores are in
operating cycles. Full provenance, manifest hashes and reproduction command
are in [`p09_xgboost_error_audit.md`](p09_xgboost_error_audit.md); the local
audit script now prints both example rows as well as aggregate differences.

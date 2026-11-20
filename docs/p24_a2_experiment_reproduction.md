# P24-A2: Reproduce the declared P08 Random Forest run

**Issue:** [P24-A2 #176](https://github.com/Artfever/FleetpulseAI/issues/176)

## Run selected

Rebuilt the FD001/train feature tables and repeated the one fixed P08-A1
Random Forest development-validation run. No parameter search or official-test
evaluation was performed. The run used the existing checked-in split manifest
and `past-only-v1` feature contract.

The source `data/raw/train_FD001.txt` was supplied in the local checkout and
copied only into a temporary isolated clone for this reproduction. It was not
added to Git. Its SHA-256 matched the pinned source digest:
`963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8`.

## Commands and provenance

Executed from `C:\Users\hp\AppData\Local\Temp\fleetpulse-p24-a1-full-final`,
a separate clone with a new Python 3.12.16 environment:

```powershell
.venv/Scripts/python.exe scripts/build_feature_datasets.py
.venv/Scripts/python.exe scripts/train_random_forest.py
.venv/Scripts/python.exe scripts/audit_random_forest_validation.py
```

The feature builder saved and reloaded both datasets successfully:

| Dataset | Rows | Features | Rebuilt payload SHA-256 |
| --- | ---: | ---: | --- |
| fitting | 16,738 / 80 engines | 96 | `6e2a0f0c28b0fbcb3afdc2b81d8f0c3aa02920617df758053d4d33f43bbb712c` |
| validation | 20 / 20 engines | 96 | `f843a342a3f8799a22a1d386936e9f4c2ab57b6f7cb30cc21885a2b989fcdff7` |

Both match the previously recorded P08 table payload hashes. The split
manifest digest is
`6fd1d32f7570f70f5a9513d900fbb9d26b182359aced425e2276700c84f08d92`.

Environment: Python 3.12.16, scikit-learn 1.9.1, joblib 1.6.0, and NumPy 2.5.4.
The recorded run configuration was unchanged: seed 42, 100 estimators,
max depth 12, minimum leaf size 5, `max_features=1.0`, bootstrap enabled,
`n_jobs=1`.

## Reproduction comparison

| Measure | Recorded P08 run | Reproduction | Difference |
| --- | ---: | ---: | ---: |
| Validation engines and predictions | 20 rows | same 20 rows; all raw predictions exactly equal | 0 rows / 0 prediction difference |
| MAE | 18.833479230754712 cycles | 18.833479230754712 cycles | 0 cycles |
| RMSE | 25.24302308237894 cycles | 25.24302308237894 cycles | 0 cycles |
| Fit plus prediction runtime | 163.1630856999982 s | 99.4007896999974 s | −63.76229600000079 s |
| Serialized model SHA-256 | `b9a7bcf7b65ae82cf9c499170325de7b24ac5433d440f45746a07e1a4bb8c70b` | `ca3f739173ae180dbb558538990a410f33784d65fb24976318555d6bd1ca7874` | bytes differ |

The model parameters and all stored tree arrays for all 100 estimators compared
equal between the two fitted objects. The two models also produced exactly
equal predictions for all 20 validation feature rows. The serialized model
files nevertheless have different hashes; this audit does not explain that
serialization difference, so it does not claim byte-for-byte artifact
reproducibility. Runtime varied and is not a model-quality result.

Selected raw predictions are also unchanged:

| FD001/train engine | Cutoff cycle | Reproduced RUL (cycles) | Validation target (cycles) |
| ---: | ---: | ---: | ---: |
| 4 | 164 | 34.04320452065351 | 25 |
| 5 | 58 | 159.8769392222517 | 211 |
| 12 | 7 | 160.9491664455265 | 163 |

The audit script verified the output manifest, dataset/source/split hashes, all
20 identities and row errors, model checksum, and recomputed MAE/RMSE. It
reported the same fitting and validation payload hashes above. No official
test data or answer file was copied into or used by the isolated run.

These are development-validation results on 20 held-out training engines. They
do not establish official-test performance, future performance, or maintenance
readiness. The generated model and feature files remain ignored in the
temporary clone and are not included in this PR.

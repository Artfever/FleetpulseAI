# P08-S1 Random Forest validation audit

**Issue:** [P08-S1 #52](https://github.com/Artfever/FleetpulseAI/issues/52)

**Audit run:** 2026-11-16, local FD001 development data

**Command:** `.venv/Scripts/python.exe scripts/audit_random_forest_validation.py`

The audit reads the saved validation manifest and `metrics-v1` record from the
ignored `artifacts/development/` bundle. It verifies the manifest's canonical
hash; the original FD001/train, split, fitting-table, and validation-table
hashes; all 20 saved partition/engine/cutoff identities and RUL targets; each
row's absolute and squared error; and independently recomputes MAE and RMSE.
It also checks that the common metrics record has the same model identity,
feature configuration, seed, parameters, runtime, metrics, and input hashes.
The forest file's SHA-256 is checked against the manifest without loading or
deserializing the model.

## Observed result

- Model: `random-forest-rul@random-forest-rul-v1`; feature contract `past-only-v1`,
  30-cycle window.
- Configuration: 100 estimators, max depth 12, minimum leaf size 5,
  `max_features=1.0`, bootstrap enabled, one worker, seed 42. The saved fit plus
  prediction runtime is 163.1630856999982 seconds; this audit did not fit a
  model.
- Validation coverage: 20 unique engines from `FD001/train`; each saved
  partition/engine/cutoff key and target matched the saved validation feature
  table exactly. The complete engine/cutoff list is below.
- Independently recomputed MAE: **18.833479230754712 cycles**.
- Independently recomputed RMSE: **25.24302308237894 cycles**.
- Both values match the prediction manifest and the separately saved,
  schema-validated `metrics-v1` JSON record within 1e-12 absolute/relative
  tolerance. All row-level stored absolute and squared errors also match fresh
  calculations.
- Input provenance: source SHA-256
  `963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8`; split
  manifest digest `6fd1d32f7570f70f5a9513d900fbb9d26b182359aced425e2276700c84f08d92`;
  fitting-table payload SHA-256
  `6e2a0f0c28b0fbcb3afdc2b81d8f0c3aa02920617df758053d4d33f43bbb712c`; validation
  table payload SHA-256
  `f843a342a3f8799a22a1d386936e9f4c2ab57b6f7cb30cc21885a2b989fcdff7`.
- Validation manifest payload SHA-256
  `92385dacb77b333280f63a97474c2abcbae84e50c9c75d1c64a725492a99ce96`; model
  binary SHA-256
  `b9a7bcf7b65ae82cf9c499170325de7b24ac5433d440f45746a07e1a4bb8c70b`.

| Engine | Validation cutoff cycle |
| ---: | ---: |
| 4 | 164 |
| 5 | 58 |
| 12 | 7 |
| 14 | 71 |
| 15 | 63 |
| 18 | 58 |
| 28 | 36 |
| 29 | 27 |
| 30 | 174 |
| 32 | 190 |
| 36 | 140 |
| 55 | 23 |
| 65 | 152 |
| 70 | 109 |
| 72 | 9 |
| 76 | 8 |
| 78 | 24 |
| 82 | 56 |
| 87 | 60 |
| 95 | 259 |

## Interpretation and limits

These are development-validation results on one saved cutoff per engine. They
show that saved prediction rows, identities, targets, row errors, and aggregate
metrics are internally consistent for this candidate. They do not establish
official-test performance, compare models on the official test set, demonstrate
maintenance readiness, or establish individual-engine reliability. No official
test file or answer file was used. Detailed error-example interpretation is
reserved for P08-S2 #129.

The local prediction bundle, metrics record, model binary, raw FD001 files, and
feature tables remain Git-ignored. The script does not deserialize the model;
model reload/parity belongs to the artifact verifier and shared P08 gate.

# P13-A3: Evaluated bundle identity audit

This audit verifies that the model bundles named in the P13-S3 final result report are the frozen P12 bundles. It checks the tracked P12 tree inventory index, each local manifest checksum and serialized weight hash, and the manifest/file hashes saved by the P13 evaluator. It performs no training or refitting.

## Observed byte identity

Command: `.venv/Scripts/python.exe scripts/audit_p13_bundles.py`.

The audit checked ten files: the P12 tree index, five manifests, and four serialized model files. The four tree/baseline manifests match the P12 tree index. All five P12 manifest payload checksums and all four weight hashes match `docs/artifact_inventory.md`. For every evaluated model, the P13 report's manifest path, manifest byte hash, manifest payload hash, version, feature config hash, weight file and weight hash match the actual frozen bundle. All five are exact matches.

| Frozen final model | Version | Manifest payload SHA-256 | Weight SHA-256 |
| --- | --- | --- | --- |
| Decision Tree | `decision-tree-rul-v1` | `ed434d11ce7ae22cd6445aa18fc4fb7e5db5f3438dded681b596ef54d6d37e89` | `c33021cb9cf5931aba545b6a327d0547f14be1363cbd0c24d57f0ad923dd7bb3` |
| Random Forest | `random-forest-rul-v1` | `f52ff4dbcb9c3876ff045bebfaef791652d996a151e987815cf37a6afed69b1a` | `712a8313271565091fb3a7c9b4fce5e5d8bf2caf64b77e8da9c942a477605d81` |
| XGBoost | `xgboost-rul-v1` | `e40f4a9142caf5dc6d8c4880a2b0e526ef0e9e097e472ab70d44aa79169c2e11` | `719a5553ba938fb40dd4196d081d7d07f6c7490be0dbbd4d0cf653e776a4816e` |
| Constant mean | `constant-rul-v1` | `dff249bdfef9426b731a4d8fb8f74bd298571d38e86eb277f0a2a8b27d724c96` | No serialized weights; mean stored in manifest |
| Dense MLP | `dense-rul-mlp-final-v1` | `7a3e56ff80df0ba194067956842dab1e9d4148f85f5dc1ad9378bca0b234549c` | `fcbd5a67ca8bf307f049c7131fee08ee484f8f35e6cb4e8966aa3c45839b4650` |

P12 tree inventory index payload SHA-256: `33bc4051835a5125f46ae31fc9f8a812325eb900559671de65dbe2e0273f8c89`. P13-S3 report payload SHA-256: `2a0686eaa3d369c435938cf6bf6256b047b3265f8ac8893e08e040adc798a6a6`.

## Fresh clean-process reload check

Command: `.venv/Scripts/python.exe scripts/smoke_p12_final_bundles_clean_process.py`.

Five separate child processes checksum-loaded their final bundle and predicted from the same FD001/train engine 4 history through cycle 164. Input: 164 rows, 30 feature-window rows, 96 ordered `past-only-v1` features; history SHA-256 `b8febe5807d0095ae9828f6c8731622113923312df98271e46a361853cff1e97`. Each process reported `training_or_fit_performed: false`.

| Bundle version | Smoke prediction (cycles) |
| --- | ---: |
| `decision-tree-rul-v1` | 29.794117647058822 |
| `random-forest-rul-v1` | 28.02408950947574 |
| `xgboost-rul-v1` | 26.14615821838379 |
| `constant-rul-v1` | 107.80786195530997 |
| `dense-rul-mlp-final-v1` | 17.24176597595215 |

These are clean-process load/prediction smoke values on training history, not quality metrics. No test labels were used in this smoke check.

## Local report and verification

The generated audit is ignored by Git at `artifacts/final/p13/a3_frozen_bundle_audit.json`. Payload SHA-256: `fd4ab713fb8c0e03f7fea2c922cd38053e10d587a5f1d78ae2151b3e09ba82b8`; whole-file SHA-256: `90759a6811b281411b17e1d0d8945f74cd80c211791b3ef9f129c80637e2cd38`.

- `.venv/Scripts/python.exe -m pytest tests/test_p13_bundle_audit.py -q` — 3 passed, including changed-weight and reported-metadata rejection.
- `.venv/Scripts/python.exe -m pytest -q` — 505 passed.
- `.venv/Scripts/ruff.exe check .` — passed.
- `git diff --check` — passed.
- The actual audit and fresh clean-process smoke both passed for all five final versions.

No teammate inspection or approval is claimed. The local P12 bundles and audit report remain Git-ignored; this tracked script and evidence document describe the exact expected checksums and reproduction command.

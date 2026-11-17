# P12-T2: Final bundle clean-process smoke

Run `python scripts/smoke_p12_final_bundles_clean_process.py` from the project root. The parent reads only the original FD001/train source, extracts engine 4's history through cycle 164, writes one immutable JSON request in a temporary directory, and starts one separate Python child process per final registry entry. Each child reads the same request file, recomputes the same 30-cycle/96-feature `past-only-v1` vector, validates the bundle using its checksum-checking loader, and predicts without fitting. The script records process IDs, the shared history hash, manifest/weight hashes, versions, config, feature counts, and outputs to ignored `artifacts/final/p12_clean_process_smoke.json`.

## Observed local run

- Dataset: `FD001/train`; engine 4; observed cycle 164; 164 raw history rows supplied; 30 feature-window rows; 96 ordered features.
- Shared input-history SHA-256: `b8febe5807d0095ae9828f6c8731622113923312df98271e46a361853cff1e97`.
- Training-source SHA-256: `963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8`.
- All five child processes loaded independently; each reported the same input-history hash and `training_or_fit_performed: false`.

| Final bundle | Version | Process ID | Raw prediction (cycles) |
| --- | --- | ---: | ---: |
| Decision Tree | `decision-tree-rul-v1` | 9080 | 29.79411765 |
| Random Forest | `random-forest-rul-v1` | 22000 | 28.02408951 |
| XGBoost | `xgboost-rul-v1` | 6776 | 26.14615822 |
| Constant mean baseline | `constant-rul-v1` | 9044 | 107.80786196 |
| Dense MLP | `dense-rul-mlp-final-v1` | 9136 | 17.24176598 |

The local report SHA-256 is `b22860316c2558bcf6fe084de1c850685623e180fe12a40bddf5b8b97f421bf3`. The report and model bundles are Git-ignored; this document and the reproducing script are tracked. Predictions above are same-engine training-history serialization/smoke outputs, not performance estimates or validation/test metrics. They do not establish maintenance readiness. Official test data and labels were not opened.

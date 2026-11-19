# P22-S2: validation MAE engine bootstrap

**Reproduction command:** `.venv/Scripts/python.exe scripts/run_p22_s2_engine_bootstrap.py --replicates 10000 --seed 42 --output docs/p22_s2_validation_bootstrap.json`  
**Result:** PASS. The runner validates the saved P06 development manifests and model file hashes, confirms all predictions refer to the same 20 held-out engine prefixes, then resamples those whole engines with replacement.

## Method

- Evaluation unit: one saved FD001/train engine prefix from the original P06 development validation split. There are 20 distinct engines; each was held out from training.
- Error: absolute raw RUL prediction minus true RUL, in cycles.
- Observed validation MAE: arithmetic mean of the 20 engine-level absolute errors.
- Bootstrap: 10,000 replicates; each replicate samples 20 engine indices with replacement. The same sampled indices are applied to all five candidates to keep them paired. Python `random.Random`, seed 42.
- Interval: 2.5th and 97.5th percentiles using linear interpolation (type-7 quantiles). The table also gives the bootstrap standard error of MAE.

## Observed results

| Development model | Version | Observed MAE (cycles) | 95% bootstrap interval (cycles) | Bootstrap SE (cycles) |
| --- | --- | ---: | ---: | ---: |
| Decision Tree | `decision-tree-rul-v1` | 22.49 | 14.36–31.06 | 4.31 |
| Random Forest | `random-forest-rul-v1` | 18.83 | 11.70–26.52 | 3.81 |
| XGBoost | `xgboost-rul-v1` | 15.11 | 9.21–22.11 | 3.32 |
| Constant mean baseline | `constant-rul-v1` | 65.73 | 50.20–80.22 | 7.76 |
| Dense RUL MLP development | `dense-rul-mlp-v1` | 15.51 | 8.70–23.68 | 3.84 |

The saved Decision Tree example helps explain the unit: engine 4 at cutoff cycle 164 has true RUL 25 and prediction 34.1961, so its absolute error is 9.1961 cycles. Engine 5 at cutoff 58 has true RUL 211 and prediction 158.16, so its error is 52.84 cycles. If only those two errors were resampled, possible bootstrap MAEs would be 9.1961, 31.0181, and 52.84 cycles. The reported interval uses all 20 engines, not this two-row illustration.

## Provenance

- Partition: FD001/train. Train source SHA-256: `963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8`.
- Split manifest semantic digest: `6fd1d32f7570f70f5a9513d900fbb9d26b182359aced425e2276700c84f08d92`.
- Saved validation feature payload SHA-256: `f843a342a3f8799a22a1d386936e9f4c2ab57b6f7cb30cc21885a2b989fcdff7`; feature config is `past-only-v1`, 30-cycle window, 24 channels / 96 ordered outputs.
- The runner checks each model's manifest checksum and model-byte checksum, shared source/split/feature hashes, all 20 engine IDs, cutoff cycles, and true labels against the split and validation feature table. It recomputes saved MAE values and verifies them within `1e-12` cycles where the manifests record a metric.
- Full manifest/model hashes, model seeds/configs, row-level outputs, and exact interval values are in `p22_s2_validation_bootstrap.json`.

This is the variability of validation MAE across resampled held-out engines, conditional on these saved model fits. It does not measure retraining variability, and the interval is not for an individual engine's RUL. The study uses simulated C-MAPSS data and is descriptive; it does not select/tune a model, use official-test labels, or provide fleet/maintenance evidence.

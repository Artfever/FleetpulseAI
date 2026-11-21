# P22-S3: sensor perturbation sensitivity on development holdouts

**Reproduction:** `.venv/Scripts/python.exe scripts/run_p22_s3_sensor_sensitivity.py --output docs/p22_s3_sensor_sensitivity.json`

**Output:** `docs/p22_s3_sensor_sensitivity.json`
**Purpose:** measure how raw saved-model RUL predictions move after a bounded, artificial change to one observed sensor channel.

## Fixed protocol

- Use the original P06 FD001/train validation split: 20 held-out engine prefixes at their saved cutoffs. Input histories are truncated to each cutoff; no later cycles or validation labels enter prediction.
- Rebuild the saved `past-only-v1` feature vector from observed history using each model manifest's exact feature configuration. The 30-cycle window includes observed rows in `[max(1, cutoff - 29), cutoff]`; missing cycles are not filled.
- For each of the 21 sensor channels, make four separate copies: multiply that one channel by 0.99, 1.01, 0.95, or 1.05 on rows within the feature window. Settings and all other sensor channels remain unchanged. These are relative artificial shifts, not a noise model.
- Evaluate the frozen development Decision Tree, Random Forest, XGBoost and Dense MLP using their saved manifests and model bytes. Include the constant-mean baseline only as a sensor-insensitive control. Do not refit, tune or use FD001/test labels.
- Compare each perturbed raw RUL estimate with the same model's unperturbed raw estimate for that engine. Report movement in operating cycles, not prediction error or accuracy.

There are 21 sensors × 2 magnitudes × 2 directions × 20 engines = 1,680 perturbed engine inputs per candidate (8,400 across five candidates). For each model and magnitude, the aggregate below covers 840 absolute prediction changes. The JSON includes raw baseline predictions per engine, each per-engine prediction delta, summaries for every sensor/direction and provenance checksums. Each `per_engine_delta_cycles` entry is `[unit_id, cutoff_cycle, raw_prediction_change_cycles]`; add its delta to that engine/model baseline to reconstruct the perturbed prediction.

Aggregate 95th percentiles use the nearest-rank rule (`ceil(0.95 × n)`).

The service loader verifies each saved model manifest, model-file checksum, estimator shape and saved feature order once before the experiment. The runner checks the raw train source, split manifest and P06 validation-feature table against their frozen hashes, rebuilds the saved validation features, and reproduces P06 model predictions before applying any changes. The MLP's saved batch output and one-row service output can differ by float32 rounding; the baseline parity tolerance is `0.0001` cycles. Perturbation deltas always use the one-row inference result as their baseline.

## Results

| Saved development candidate | Mean absolute movement at ±1% (cycles) | 95th percentile (cycles) | Maximum (cycles) | Changed outputs / 840 | Mean absolute movement at ±5% (cycles) | 95th percentile (cycles) | Maximum (cycles) | Changed outputs / 840 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Decision Tree v1 | 4.08 | 27.43 | 134.41 | 80 | 4.44 | 33.12 | 134.41 | 82 |
| Random Forest v1 | 4.03 | 24.92 | 123.58 | 527 | 4.43 | 25.72 | 123.65 | 538 |
| XGBoost v1 | 4.85 | 27.81 | 62.99 | 448 | 4.35 | 23.94 | 62.99 | 454 |
| Dense MLP v1 | 557.96 | 2,498.38 | 14,098.01 | 840 | 2,830.33 | 12,626.59 | 70,505.38 | 839 |
| Constant mean control | 0.00 | 0.00 | 0.00 | 0 | 0.00 | 0.00 | 0.00 | 0 |

The ±5% aggregate is not uniformly larger than ±1% for the tree and boosting candidates because their predictions change at learned split boundaries. The Dense MLP moved on essentially every perturbation and produced very large raw shifts for some small input changes (sensor 6 was the largest contributor in this run). This flags substantial sensitivity of this frozen model under the chosen artificial perturbations; it is not evidence about real sensor faults or physical failure causes. Its displayed RUL may be clipped elsewhere, but these diagnostics intentionally preserve raw predictions.

## Provenance and limits

- Partition: simulated NASA C-MAPSS FD001/train; 20 original P06 validation engines.
- Train source SHA-256: `963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8`.
- Split-manifest semantic SHA-256: `6fd1d32f7570f70f5a9513d900fbb9d26b182359aced425e2276700c84f08d92`.
- Saved validation-feature payload SHA-256: `f843a342a3f8799a22a1d386936e9f4c2ab57b6f7cb30cc21885a2b989fcdff7`.
- Model versions, manifest hashes, model-file hashes, per-engine baselines and exact result summaries are recorded in the JSON output.

This is a bounded sensitivity experiment conditional on fixed model fits and one simulated dataset split. The percentage changes are chosen stress inputs, not observed measurement variation. Prediction movement is not per-engine RUL uncertainty, model accuracy, a causal effect, failure diagnosis, maintenance evidence, or a reason to tune against the holdout. The official test set is not used.

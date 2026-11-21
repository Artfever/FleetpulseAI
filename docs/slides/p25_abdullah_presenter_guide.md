# P25-A3 presenter guide: leakage and artifact loading

Companion to [Abdullah's architecture/data slides](p25_abdullah_architecture_data.md). This guide is for a short live explanation. The source code, audited documents, and packaged history are the evidence; the slides are only a teaching aid.

## Five-minute talk track

### 1. Start with the three files

`train_FD001.txt` contains complete simulated training histories. `test_FD001.txt` contains observed prefixes for the held-out test engines. `RUL_FD001.txt` is a separate vector of answers. The prediction program reads test histories first and does not accept the answer-file path. A later pairing step maps answer lines to sorted engine IDs, and the final scoring step calculates error metrics.

The same unit number can occur in both partitions, so I name an engine with its partition as well as its ID. For example, `FD001/train:4` and `FD001/test:4` are different histories.

### 2. Teach the leakage example

Use `FD001/train:4` at cycle 164. Its complete history ends at cycle 189, so the scoring target is `189 - 164 = 25` operating cycles. The predictor receives only rows through cycle 164; it does not receive cycles 165–189. For this contiguous history, the feature window is cycles 135–164, or 30 observed rows.

The packaged prefix used in the numeric preview is `data/demo/fd001_training_prefixes.txt`, SHA-256 `953489b722a33815ee08f360c1a677d314f318b7d74d8bef88cd31dde0a8f9a7`. It is a small simulated training-history example, not the complete source dataset.

The model input contains 96 sensor/setting summaries. It does not contain the target, the future sensor rows, the full lifetime, or the engine ID. The lifetime is used afterward to score the already-formed prediction. This keeps the label available for evaluation without letting it leak into the input.

The actual preview gives sensor 4 latest/mean/population-standard-deviation/slope values of 1411.81, 1411.7286666666666, 4.397945530459507, and 0.2859310344827574. These describe the observed simulation window; they do not identify a fault.

### 3. Explain a saved bundle

The registry names the supported bundle and version. Its manifest describes the feature version, ordered names, training provenance, and model identity. For a tree estimator, the loader verifies the manifest digest, supported schema/version, expected feature configuration, and serialized weight checksum before deserializing the estimator. The constant-mean baseline has no separate weights file; its selected statistic and training count are stored in its manifest. Inference then builds the saved feature contract and calls the loaded model. It does not retrain the model when a prediction is requested.

If a hash, version, shape, or feature order does not match, the loader rejects the artifact. A clear error is safer than silently predicting with a different model. Do not pass untrusted pickle/joblib files to the loader.

### 4. State the result limit

The development Decision Tree returned raw RUL 34.19607843137255 cycles for this one prefix, while the target is 25 cycles. The absolute error for this engine is 9.19607843137255 cycles. This one example is not a model score. We need the documented set of 20 held-out training engines for development MAE/RMSE, or the separate 100-engine FD001/test workflow for final held-out reporting.

The data are simulated and do not establish aircraft condition, maintenance urgency, or maintenance benefit. Planner durations and priorities are assumptions.

## Questions for the audience

1. **Why split by engine rather than by row?** So an engine's closely related rows cannot land on both sides of validation.
2. **Can cycles 165–189 enter the prediction for cycle 164?** No. They are future observations for that cutoff.
3. **Can the known final cycle be used at all?** It can produce the separate validation target after the input prefix is fixed; it cannot be an inference feature.
4. **What does `past-only-v1` contain?** 24 channels (settings and sensors), four fixed statistics per channel, in a saved order, over up to the previous 30 cycles ending at the chosen cutoff.
5. **What makes the model bundle compatible?** Registered name/version, artifact schema, feature version/order/count, and manifest hash must validate; an estimator's separate weight file must also match its recorded checksum.
6. **Does loading or predicting fit a model?** No. Training is a separate development workflow.
7. **Does 34.196 cycles mean the engine will fail in 34 cycles?** No. It is a model estimate from one simulated history, not a physical guarantee.

## Evidence to open during the discussion

- [P25-A1 data, feature, and service chapter](../report/p25_a1_data_features_architecture.md)
- [P25-A2 pipeline and reproduction report](../report/p25_a2_pipeline_and_reproduction.md)
- [P21-A2 actual feature preview](../p21_a2_feature_preview_service.md)
- [Training target definition](../training_targets.md)
- [Development split and manifest](../development_split.md)
- [Final tree bundle checks](../p12_a3_final_tree_bundles.md)
- Source: `src/fleetpulse/features/engineering.py`, `src/fleetpulse/data/split_manifest.py`, `src/fleetpulse/models/final_tree_bundle.py`, and `src/fleetpulse/services/predictor.py`.

The repository task can prepare this script, slides, and evidence links. A live teammate walkthrough still needs a person to present and confirm that it occurred; this guide does not claim that a session happened.

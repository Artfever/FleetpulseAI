---
marp: true
theme: default
paginate: true
---

# FleetPulse: Data to RUL estimate

### Data, leakage boundaries, features, and saved-model loading

Abdullah | P25 architecture and data

NASA C-MAPSS FD001 is simulated engine data. RUL is an estimate in operating cycles, not a physical fault diagnosis or a maintenance recommendation.

<!--
Source: docs/report/p25_a1_data_features_architecture.md
Audience: project teammates. Suggested duration: 5 minutes.
-->

---

## Three files, three roles

| File | What it contains | How FleetPulse uses it |
| --- | --- | --- |
| `train_FD001.txt` | Complete simulated engine histories | Build training examples, targets, and held-out validation prefixes |
| `test_FD001.txt` | Observed history prefixes for 100 other simulated engines | Make final predictions from sensor history only |
| `RUL_FD001.txt` | One separate remaining-life answer per test engine | Pair after predictions for final scoring only |

An engine number is not unique across partitions. Use identities such as `FD001/train:4` and `FD001/test:4`.

Full raw files stay local and are checked against the hashes in [FD001 provenance](../fd001_provenance.md). The small tracked demo subset is not the full dataset.

<!--
Say: "The separate test-answer file is not part of prediction input. It joins only after the raw predictions have been saved."
Source: docs/fd001_provenance.md; docs/report/p25_a1_data_features_architecture.md
-->

---

## Split whole engines before model comparison

**Development split:** 80 FD001/train engines for fitting; 20 different engines for validation. The checked-in manifest fixes engine membership and validation cutoffs.

```text
FD001/train engine 4: observed cycles 1 ... 164 | future cycles 165 ... 189
Predictor sees:       [========================]
Validation target:                              189 - 164 = 25 cycles
```

For the validation prediction, only rows through cycle 164 are visible. The complete training lifetime supplies the separate target of 25 cycles after prediction inputs are fixed.

This is the leakage boundary: target knowledge and future sensor rows do not enter the feature vector.

<!--
Ask: "If I shuffle rows instead of splitting engines, what leaks?"
Expected answer: rows from the same engine can appear in both fitting and validation, so the validation score no longer tests an unseen engine.
Source: docs/development_split.md; docs/training_targets.md; docs/evaluation_protocol.md
-->

---

## Past-only features at one cutoff

`past-only-v1` uses 24 channels: 3 settings + 21 sensors. For each channel it calculates:

**latest · mean · population standard deviation · slope by actual cycle**

24 channels × 4 statistics = **96 ordered numeric features**

At engine 4, cycle 164, the inclusive 30-cycle window is cycles 135–164. Its sensor 4 features were:

| Feature | Value | Meaning |
| --- | ---: | --- |
| latest | 1411.81 | Last observed reading in the window |
| mean | 1411.7287 | Average of the observed window readings |
| population std | 4.3979 | Spread of those readings (`ddof=0`) |
| slope | 0.2859 | Reading change per operating cycle |

Engine ID, cutoff, full lifetime, and true RUL are not standalone model columns. Cycle numbers are used only to calculate each slope; labels and metadata stay outside X.

<!--
This is a feature-preview output from one simulated FD001/train prefix, not a diagnosis or model-quality result.
Source: data/demo/fd001_training_prefixes.txt; docs/p21_a2_feature_preview_service.md
-->

---

## Loading a saved model does not train it

```text
Observed history
  -> validate schema and values
  -> registry selects the requested bundle
  -> verify manifest, version, feature schema and artifact hashes
  -> build the exact ordered feature vector
  -> predict raw RUL in cycles
```

The manifest pins model identity and feature configuration. The loader checks compatibility and hashes, including an estimator's weight checksum when it has a separate weights file. Prediction does not fit a model or preprocessing step.

The service keeps raw RUL separate from display clipping and planner metadata. Metadata identifies the engine and cutoff; it is not a sensor feature.

Source paths: `src/fleetpulse/models/registry.py`, `src/fleetpulse/models/final_tree_bundle.py`, `src/fleetpulse/services/predictor.py`, `src/fleetpulse/features/engineering.py`.

<!--
Ask: "What should happen if the manifest feature order or model-file hash changes?"
Expected answer: reject the incompatible/corrupt bundle with a clear artifact error rather than silently loading or training a replacement.
Source: docs/p12_a3_final_tree_bundles.md; docs/prediction_service.md
-->

---

## One observed prediction is not a performance claim

For the same simulated validation prefix, the saved development Decision Tree returned **34.1961 cycles**. The separate target was **25 cycles**. The difference on this one engine is **9.1961 cycles**.

That single example explains the data path; it does not summarize model quality. Aggregate development and final test metrics use separate, explicitly documented cohorts and protocols.

Maintenance durations, priorities, team capacity, buffers, and horizons are user-set simulation assumptions. FD001 does not establish maintenance needs or benefits.

See [P13 final results](../p13_s3_final_test_results.md) and [P20 prediction/planner boundary](../p20_prediction_planner_boundary.md) for the distinct evaluation and scheduling limits.

<!--
Close: "We can trace every number to a partition, cutoff, feature configuration, saved model, and evaluation role."
Do not describe a prediction as a fault, remaining hours, or confirmed maintenance timing.
-->

# Data, targets, features, and service architecture

Report material for P25-A1 / [issue #102](https://github.com/Artfever/FleetpulseAI/issues/102).

## Project scope

FleetPulse is a software prototype that estimates remaining useful life (RUL) for simulated turbofan-engine histories and compares illustrative maintenance schedules. The data are NASA C-MAPSS FD001 simulation trajectories, not records from aircraft in service. A prediction is an estimate in operating cycles; it does not diagnose a physical fault or establish that maintenance is needed. The scheduling interface consumes predictions together with user-supplied duration, priority, team-capacity, buffer, and horizon assumptions. Those controls are simulation inputs, not measured maintenance facts.

The implemented product is a Streamlit application. It supports a small tracked FD001/train demonstration subset and exact-schema CSV uploads, plus locally provisioned full histories for development workflows. A subset is not a substitute for the original source files. The product and evidence limits are summarized in the [course compliance note](../COURSE_COMPLIANCE.md) and [P24 release checklist](../p24_feature_freeze_and_release_checklist.md).

## Dataset and provenance

The project uses the NASA Prognostics Center of Excellence Turbofan Engine Degradation Simulation data. The project’s [provenance record](../fd001_provenance.md) identifies the archive/retrieval route, observed archive hashes, and local file checksums. The recorded hashes are locally observed values, not checksums published by NASA.

| File | Role | Key handling rule |
| --- | --- | --- |
| `train_FD001.txt` | 100 complete simulated training histories | Used for the fixed fitting/validation split, development targets, features, and candidate training. |
| `test_FD001.txt` | 100 simulated test histories, each observed only through its final recorded cycle | Predictions are generated from sensor histories before labels are paired. |
| `RUL_FD001.txt` | One separate answer per test engine | Reserved for final scoring; never a predictor input or model-selection signal. |

Each raw row has 26 normalized fields: `unit_id`, `cycle`, three operating settings, and 21 sensor readings. Unit IDs may repeat between train and test, so identity is always dataset-qualified, such as `(FD001/train, 4)` or `(FD001/test, 4)`. The full raw files are intentionally not tracked in Git. The small packaged demo histories under `data/demo/` are tracked and checksum-described separately. See [FD001 provenance](../fd001_provenance.md), [dataset notes](../dataset_notes.md), and the [data policy](../data_policy.md).

The loader validates the schema and record values; it does not download or alter source files. Quality checks cover integral positive IDs/cycles, finite readings, duplicate engine-cycle pairs, and cycle order. Gaps are reported as warnings rather than silently filled. The detailed checks and observed dataset counts are in [FD001 dataset notes](../dataset_notes.md).

## RUL target and safe evaluation split

For a complete training engine with last observed cycle `L`, the label for its row at cycle `t` is:

```text
RUL(engine, t) = L - t
```

The target is an integer count of operating cycles. It is uncapped, nonnegative, and zero on a complete training engine’s terminal row. The label builder keeps targets separate from raw sensor rows and returns a separate label vector. The model feature matrix does not contain engine identity, cycle, full lifetime, or true RUL. See [training targets](../training_targets.md).

Development validation separates whole engines, not random rows from the same engine. The checked-in manifest assigns 80 of the 100 `FD001/train` engines to fitting and 20 to validation, with split seed 42 and saved observed cutoffs. The manifest is verified against the source digest. The feature builder uses complete fitting-engine trajectories to create fitting examples but only each validation engine’s prefix through its saved cutoff to create its validation row. The validation label is derived separately from the complete training lifetime and cutoff. This is why the same engine’s future sensor rows can inform its scoring label but cannot enter its validation features.

For the official test set, every model predicts once from each engine’s observed test prefix. Only after the prediction artifact is created are the separate RUL answers paired to sorted test engine IDs and aggregate errors calculated. This separation avoids using answer labels to form predictions or choose a model. The [development split](../development_split.md), [evaluation protocol](../evaluation_protocol.md), and [P13 label-pairing report](../p13_s2_test_label_pairing.md) describe these contracts.

## Past-only feature representation

The versioned default is `past-only-v1`: all 24 reading channels (settings 1–3 and sensors 1–21), each summarized as `latest`, arithmetic `mean`, population standard deviation (`ddof=0`), and least-squares slope against actual cycle numbers. The channel order and then statistic order are fixed by `FeatureConfig`; the default model input has 96 numeric features. The default window is the inclusive 30-cycle interval `[cutoff - 29, cutoff]`. The extractor uses observed rows in that interval, does not fill cycle gaps, and does not look past the selected cutoff. A one-row history has standard deviation and slope zero. Metadata and labels remain outside the feature matrix. See the [feature contract](../feature_contract.md) and [feature example](../feature_example.md).

### Concrete engine-prefix example

The saved development manifest identifies `FD001/train` engine 4 at cutoff cycle 164 as a validation prefix. Its complete training trajectory ends at cycle 189, so its separate target is `189 - 164 = 25` cycles. The predictor input is the observed prefix through cycle 164; cycles 165–189 are hidden from prediction. The feature window uses 30 observed rows, cycles 135–164 for this contiguous history.

The local service-free feature-preview command was run against the tracked packaged history and returned 96 ordered `past-only-v1` values. For example:

| Feature | Observed value | Meaning |
| --- | ---: | --- |
| `sensor_1_latest` | 518.67 | Reading at the cutoff, in recorded source units |
| `sensor_1_mean` | 518.67 | Mean in the observed 30-row window |
| `sensor_1_std` | 0 | Population standard deviation in that window |
| `sensor_1_slope` | 0 | Change in recorded source units per operating cycle |
| `sensor_4_latest` | 1411.81 | Reading at the cutoff, in recorded source units |
| `sensor_4_mean` | 1411.728666666667 | Mean in the observed window |
| `sensor_4_std` | 4.397945530459507 | Population standard deviation in the window |
| `sensor_4_slope` | 0.2859310344827574 | Change in recorded source units per operating cycle |

This example shows the leakage boundary: the feature service receives the visible prefix only; target calculation may use the known complete training lifetime separately. A saved development Decision Tree later returned raw RUL `34.19607843137255` cycles for this same packaged prefix. That is one simulated training-prefix estimate, not a target, accuracy measure, or maintenance conclusion. Engine identity, selected cutoff, window-row count, and model version accompany the result as metadata; they are not added to the feature vector. Reproduce the numeric preview with `python scripts/check_p21_a2_feature_preview.py`.

## System architecture

The application keeps source data, feature construction, model inference, and presentation as distinct layers:

| Layer | Main implementation | Responsibility and boundary |
| --- | --- | --- |
| Source and validation | `src/fleetpulse/data/loader.py`, `quality.py`, `input_validation.py` | Parse normalized histories and reject malformed schemas/values; preserve source identity and provenance. |
| Targets and split | `src/fleetpulse/data/targets.py`, `split.py`, `split_manifest.py`, `split_audit.py` | Derive training labels separately; verify whole-engine membership, source hashes, saved cutoffs, and no fitting/validation overlap. |
| Feature engineering | `src/fleetpulse/features/config.py`, `engineering.py` | Validate the versioned channel/window configuration and produce the exact ordered past-only numeric vector. |
| Training tables and persistence | `src/fleetpulse/features/build_dataset.py`, `dataset_io.py` | Build fitting examples from training-engine prefixes and validation examples from fixed saved cutoffs; persist feature rows, keys, labels, config, and provenance with labels/metadata separate from X. |
| Models and artifacts | `src/fleetpulse/models/`, `src/fleetpulse/evaluation/` | Train bounded candidates on development fitting rows, verify saved bundles and ordered feature contracts, and evaluate with explicit split/model provenance. Final frozen bundles are separate from development candidates. |
| Prediction services | `src/fleetpulse/services/predictor.py`, `fleet.py`, `uploaded_history.py`, `final_comparison.py` | Validate an observed history, load a requested verified bundle, compute its saved feature schema, and return raw RUL with explicit identity/version context. Prediction inputs never contain true labels or future rows. |
| User interface | `app/Home.py`, `app/pages/`, `app/views/` | Present Home, Fleet Overview, Engine Explorer, RUL Prediction, Maintenance Planner, and Model Laboratory; render validation and unavailable-artifact states. |
| Planning | `src/fleetpulse/services/planner.py`, `schedule_comparison.py`, `src/fleetpulse/optimization/` | Convert successful estimates into a shared scenario and compare EDD/GA schedules. Raw model values, planner clamping, and simulation assumptions remain distinguishable. |

`PROJECT_ROOT` anchors repository-relative paths for an editable installation. The development and final registries are separate. Model loading validates artifact identity and hashes; the application does not train a replacement model during startup or prediction. See [prediction service](../prediction_service.md), [artifact inventory](../artifact_inventory.md), and [P20 prediction/planner boundary](../p20_prediction_planner_boundary.md).

## Reproduction, versions, and limits

The supported local runtime is Python 3.12 (`>=3.12,<3.13`), pinned in `pyproject.toml`; dependencies and clean-install results are recorded in the [setup guide](../SETUP.md) and [P24-A1 evidence](../../handoff_Abdullah.md). With the original raw training file provisioned and checksum-verified, `scripts/verify_split_manifest.py`, `scripts/verify_training_targets.py`, and `scripts/build_feature_datasets.py` check the source/split/targets and rebuild ignored development feature tables. Model-training and official-evaluation procedures are recorded with their exact configurations in the corresponding P08–P13 reports; they are not run implicitly by starting Streamlit.

The reproducible engine-4 example was run on Python 3.12.16 from the tracked demo file `data/demo/fd001_training_prefixes.txt`, SHA-256 `953489b722a33815ee08f360c1a677d314f318b7d74d8bef88cd31dde0a8f9a7`. The saved development model and complete raw FD001 histories may not be present in a fresh clone; check the artifact and data instructions before attempting reproduction. No results in this chapter are real-aircraft performance or evidence of maintenance benefit.

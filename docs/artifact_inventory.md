# FleetPulse model and artifact inventory

Updated 20 November 2026. This inventory distinguishes development candidates from P12 full-training refits. The code registry is `fleetpulse.models.registry`: `DEVELOPMENT_MODEL_REGISTRY` (backward-compatible alias `MODEL_REGISTRY`), `FINAL_MODEL_REGISTRY`, and the stage-grouped `MODEL_REGISTRIES`. Treat stage, model name, version, feature config, and artifact checksum together; a repeated name/version does not mean that the serialized weights are identical.

## Development candidates

The current shared predictor accepts these saved candidate formats. The files below are present in this workspace and Git-ignored; a fresh clone must reproduce/provision them before the app can use them.

| Model | Version | Manifest | Weights | Local status |
| --- | --- | --- | --- | --- |
| Decision Tree | `decision-tree-rul-v1` | `artifacts/development/decision_tree_validation.json` | `artifacts/development/decision_tree_model.joblib` | Present; P06 validation candidate |
| Random Forest | `random-forest-rul-v1` | `artifacts/development/random_forest_validation.json` | `artifacts/development/random_forest_model.joblib` | Present; P08 validation candidate |
| XGBoost | `xgboost-rul-v1` | `artifacts/development/xgboost_validation.json` | `artifacts/development/xgboost_model.joblib` | Present; P09 validation candidate |
| Dense MLP | `dense-rul-mlp-v1` | `artifacts/development/mlp/mlp_training_manifest.json` | `artifacts/development/mlp/mlp_best_state.pt` | Present; P10 validation candidate |

The P06 constant baseline is an evaluation comparator, not an entry in the runtime development-model registry. Its final full-training mean bundle is listed below.

## P12 final bundles

All five final bundles were checksum-loaded and independently smoke-predicted in new processes in P12-T2. They use the 30-cycle, 96-feature `past-only-v1` contract and were refit using all 100 original FD001/train engines (20,631 cycle rows). P23-T1 packages only the five serving bundles and their index for the full release; P13 evaluation outputs, refit intermediates, full source data, and development candidates remain excluded. The deployment registry resolves paths relative to the repository package root.

| Final model | Version | Manifest path | Payload SHA-256 | Weights SHA-256 / state |
| --- | --- | --- | --- | --- |
| Decision Tree | `decision-tree-rul-v1` | `artifacts/final/tree/bundles/decision-tree-rul_manifest.json` | `ed434d11ce7ae22cd6445aa18fc4fb7e5db5f3438dded681b596ef54d6d37e89` | `c33021cb9cf5931aba545b6a327d0547f14be1363cbd0c24d57f0ad923dd7bb3` |
| Random Forest | `random-forest-rul-v1` | `artifacts/final/tree/bundles/random-forest-rul_manifest.json` | `f52ff4dbcb9c3876ff045bebfaef791652d996a151e987815cf37a6afed69b1a` | `712a8313271565091fb3a7c9b4fce5e5d8bf2caf64b77e8da9c942a477605d81` |
| XGBoost | `xgboost-rul-v1` | `artifacts/final/tree/bundles/xgboost-rul_manifest.json` | `e40f4a9142caf5dc6d8c4880a2b0e526ef0e9e097e472ab70d44aa79169c2e11` | `719a5553ba938fb40dd4196d081d7d07f6c7490be0dbbd4d0cf653e776a4816e` |
| Constant mean baseline | `constant-rul-v1` | `artifacts/final/tree/bundles/constant-rul_manifest.json` | `dff249bdfef9426b731a4d8fb8f74bd298571d38e86eb277f0a2a8b27d724c96` | No weights file; mean `107.80786195530997` cycles from 20,631 training labels |
| Dense MLP | `dense-rul-mlp-final-v1` | `artifacts/final/mlp/bundle/final_mlp_manifest.json` | `7a3e56ff80df0ba194067956842dab1e9d4148f85f5dc1ad9378bca0b234549c` | `fcbd5a67ca8bf307f049c7131fee08ee484f8f35e6cb4e8966aa3c45839b4650` |

The tree bundle inventory index payload SHA-256 is `33bc4051835a5125f46ae31fc9f8a812325eb900559671de65dbe2e0273f8c89`. The MLP final bundle uses hidden widths `(128, 64)`, seed 42, eight fixed epochs, and its saved 96-feature StandardScaler. The scaler payload SHA-256 is `b56c8d1d3a8b946f48bbfd231ff9f6596c44a3ccbf2ab2235c69f4acde4f69cd`. Per-model configurations, row counts, source/selection provenance, and clean-process outputs are documented in the linked [P12 task reports](p12_a1_final_tree_refit.md), [P12 MLP bundle report](p12_s3_final_mlp_bundle.md), [P12 registry report](p12_t1_model_registry.md), [P12 clean-process report](p12_t2_clean_process_smoke.md), and [P12 final tree bundle report](p12_a3_final_tree_bundles.md).

## Cache-key verification

The bounded development history-replay cache key in `fleetpulse.services.history` contains partition, engine ID, cutoff, a canonical history hash, the artifact model version, a canonical feature-config hash, and an artifact fingerprint over manifest bytes plus serialized model bytes. This prevents reuse across changed model versions, feature configurations, model bytes, histories, or engine/cutoff identities.

`python -m pytest tests/test_history_service.py -q` passed 15 tests in 1.95 seconds. Those tests verify cache reuse for identical inputs and invalidation after changes to history, model version, feature config, or model bytes; they also verify the cache bound. P12 final bundles use their separate checksum-verifying loaders and are not routed through this development replay cache. If final inference is later added to that cache, include the final stage and bundle identity/config in its key or leave that path uncached.

## Evidence boundaries

The P12 clean-process numbers are smoke predictions on engine 4 from FD001/train, not validation or test metrics. No official test histories or RUL answers were used to fit, select, package, or score these artifacts. The model card at [model_card.md](model_card.md) describes the P06 development Decision Tree; it does not report final-refit model performance. The current RUL app selector still runs the development inference path. Final artifact files remain local/ignored, so registry entries do not establish that final bundles are deployed.

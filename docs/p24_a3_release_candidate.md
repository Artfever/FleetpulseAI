# P24-A3 release candidate inventory and audit

Date: 2026-11-21

Base: `dev` at `52fdd4f4e2830df13bfd0413631243ef701cf226` (PR #525)

Task: [P24-A3 #230](https://github.com/Artfever/FleetpulseAI/issues/230)

## Result

The tracked final serving artifacts were inventoried and the existing frozen-bundle audit passed. It checked 10 model manifest/weight files across five models and confirmed that all five evaluated models match the frozen inventory and reported P13 results. No artifact, model, dataset, or code was changed by this audit.

The candidate snapshot is proposed as annotated tag `p24-rc1` after this report is merged to `dev`. This is a release candidate, not a claim that the P24 shared gate passed. P23 gate #28 was closed as `NOT_PLANNED` at Abdullah's direction; the hosted SHAP check for #174 and fresh upload-failure check for #262 remain unverified. The P24 shared gate #29 remains open.

## Audit command and output

Command:

```powershell
.\.venv\Scripts\python.exe scripts/audit_p13_bundles.py --output $env:TEMP\fleetpulse-p24-a3-audit-current.json
```

Result: PASS. `bundle_files_checked=10`, `model_count=5`, `evaluated_models_match_frozen_inventory=true`, `test_artifact_used_for_training_or_tuning=false`. The generated audit JSON has SHA-256 `90759a6811b281411b17e1d0d8945f74cd80c211791b3ef9f129c80637e2cd38` (temporary file; it is not added to the repository). The normalized P13 result payload SHA-256 is `2a0686eaa3d369c435938cf6bf6256b047b3265f8ac8893e08e040adc798a6a6`; the tree bundle index payload SHA-256 is `33bc4051835a5125f46ae31fc9f8a812325eb900559671de65dbe2e0273f8c89`.

| Final model | Version | Manifest payload SHA-256 | Weights SHA-256 | Evaluation matched |
| --- | --- | --- | --- | --- |
| Decision Tree | `decision-tree-rul-v1` | `ed434d11ce7ae22cd6445aa18fc4fb7e5db5f3438dded681b596ef54d6d37e89` | `c33021cb9cf5931aba545b6a327d0547f14be1363cbd0c24d57f0ad923dd7bb3` | yes |
| Random Forest | `random-forest-rul-v1` | `f52ff4dbcb9c3876ff045bebfaef791652d996a151e987815cf37a6afed69b1a` | `712a8313271565091fb3a7c9b4fce5e5d8bf2caf64b77e8da9c942a477605d81` | yes |
| XGBoost | `xgboost-rul-v1` | `e40f4a9142caf5dc6d8c4880a2b0e526ef0e9e097e472ab70d44aa79169c2e11` | `719a5553ba938fb40dd4196d081d7d07f6c7490be0dbbd4d0cf653e776a4816e` | yes |
| Constant mean baseline | `constant-rul-v1` | `dff249bdfef9426b731a4d8fb8f74bd298571d38e86eb277f0a2a8b27d724c96` | not applicable (manifest-only) | yes |
| Dense MLP | `dense-rul-mlp-final-v1` | `7a3e56ff80df0ba194067956842dab1e9d4148f85f5dc1ad9378bca0b234549c` | `fcbd5a67ca8bf307f049c7131fee08ee484f8f35e6cb4e8966aa3c45839b4650` | yes |

The authoritative file inventory and training-source limitations are in [`artifact_inventory.md`](artifact_inventory.md). The audit validates frozen bundle identity and correspondence to the P13 report; it does not recompute the full evaluation from raw data.

## Tracked final artifact files

The following 13 paths are tracked under `artifacts/final/` (including the empty directory marker); no raw FD001 history, answer file, development checkpoint, or ignored evaluation intermediate is included by this task.

| Path | Bytes | File SHA-256 |
| --- | ---: | --- |
| `artifacts/final/.gitkeep` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `artifacts/final/explanation/decision-tree-background.json` | 127,519 | `12b8031a9eff3259e465aba666f4547eda32e1e5df664a2e43a5ffc4012e1b55` |
| `artifacts/final/mlp/bundle/final_mlp_manifest.json` | 12,096 | `ee16c7461d25d58498e411afea8f234fdb7b7b42af25ee33325037aa03d792fe` |
| `artifacts/final/mlp/bundle/final_mlp_state.pt` | 85,893 | `fcbd5a67ca8bf307f049c7131fee08ee484f8f35e6cb4e8966aa3c45839b4650` |
| `artifacts/final/p13/final_test_results.json` | 140,610 | `12f63b73f763cac0b70d133a3b1dd63c06422adf81e4900698383c0037b7925f` |
| `artifacts/final/tree/bundles/constant-rul_manifest.json` | 12,542 | `e0c24ad1d9cb514dde0905824f9d12da4dfe9b86dc2e2d14767e7e104dfcffb0` |
| `artifacts/final/tree/bundles/decision-tree-rul_manifest.json` | 12,431 | `6b223d2faba3bf3f05677fb66677218a3f2169f89599a9d1ee2d04ca8b869319` |
| `artifacts/final/tree/bundles/decision_tree_final_v1.joblib` | 34,193 | `c33021cb9cf5931aba545b6a327d0547f14be1363cbd0c24d57f0ad923dd7bb3` |
| `artifacts/final/tree/bundles/final_tree_bundle_index.json` | 1,302 | `17a244ff946744ce4e8b26c370e36894f3933036649e644b1ba25b53588baf2c` |
| `artifacts/final/tree/bundles/random-forest-rul_manifest.json` | 12,542 | `e07eb946136c38627dc1047b692059f4a36cf5e4cd5ea26902bf7e894a795ae7` |
| `artifacts/final/tree/bundles/random_forest_final_v1.joblib` | 25,337,345 | `712a8313271565091fb3a7c9b4fce5e5d8bf2caf64b77e8da9c942a477605d81` |
| `artifacts/final/tree/bundles/xgboost-rul_manifest.json` | 12,774 | `31165a1882ed4be3620f33534d7e5e5de93c093cdaced3fe8734e77575eb6fad` |
| `artifacts/final/tree/bundles/xgboost_final_v1.joblib` | 181,042 | `719a5553ba938fb40dd4196d081d7d07f6c7490be0dbbd4d0cf653e776a4816e` |

`final_test_results.json` is checked against its normalized Git payload by the audit. Its raw Windows working-tree byte hash can differ from the Git blob due to line-ending conversion (`core.autocrlf=true`); this is not an unexplained model or result change. The tree index file SHA above is the raw file checksum; its normalized payload checksum is also recorded above.

## P24 evidence and limits

- P24-A1 #99 is merged in PR #524; a fresh Python 3.12.16 setup passed Ruff, 727 tests (7 skipped), and Streamlit health. See the handoff for the exact commands.
- P24-A2 #176 is merged in PR #525; rebuilt source/features matched recorded hashes and all 20 validation predictions and MAE/RMSE matched exactly. Runtime and serialized model bytes differed; byte-identical model reproduction is not claimed. See [`p24_a2_experiment_reproduction.md`](p24_a2_experiment_reproduction.md).
- This A3 checksum audit does not establish public deployment, successful fresh remote flows, full metric recomputation, model quality, or maintenance readiness. P23 remote verification was waived, not completed.
- No artifact changes were made. Relevant verification is the bundle audit above and `git diff --check`; Ruff/pytest are not relevant to this docs-and-tag-only deliverable and are not run for it.

## Release candidate tag

Proposed annotated tag: `p24-rc1`, pointing to the dev merge commit that contains this report. The tag message must retain the caveat that this is an auditable candidate snapshot and that hosted P23 checks were waived/unverified. Do not treat it as a production release or as closure of P24 gate #29.

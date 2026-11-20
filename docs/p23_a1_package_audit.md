# P23-A1 deployment package audit

## Audited package

The package is the `dev` revision produced by P23-T1 PR #508. It tracks the five serving bundles referenced by `fleetpulse.models.registry.FINAL_MODEL_REGISTRY`, the approved small parity subset, and the pinned optional dependencies for XGBoost and PyTorch. Model manifests and payloads use repository-relative bundle paths and validated content hashes; inspection found no developer-specific absolute paths in the final model manifests.

The tracked deployment data is limited to:

- `data/demo/fd001_training_prefixes.txt` (229 owner-approved FD001/train rows; SHA-256 `e31832ddbc23fa32b4093e1325f00fd6bed2cb2d3f6d03ca13ace44886390b6d`).
- `data/demo/illustrative_history.txt` (the existing synthetic demo).
- Ten frozen P12 serving files under `artifacts/final/tree/bundles/` and `artifacts/final/mlp/bundle/`.
- The checksum-sealed P13 display report `artifacts/final/p13/final_test_results.json`. It contains final per-engine evaluation results and pinned source/model provenance for the Model Laboratory; it does not contain raw sensor histories or model weights.

`git ls-files data artifacts/final` confirms that full raw histories, development candidates, and refit intermediates are not part of the release package. Only the verified P13 display report is included; raw source datasets and derived figures remain Git-ignored. `requirements.txt` installs the project's pinned core dependencies and the pinned XGBoost/PyTorch extras. The app resolves final manifests from the repository package root, not from a developer's absolute path.

## Clean-checkout inference check

Run `.venv/Scripts/python.exe scripts/smoke_p23_a1_release.py` after installing the root `requirements.txt`. The script reads only the tracked parity subset, selects engine 4 through observed cycle 164, and calls the actual `compare_final_models` serving service with the checkout root as the artifact root. It loads all five final models and prints each raw RUL estimate and manifest/weight hash. It does not train, use labels, or open a full source dataset.

In a detached clean Git worktree at `aa6788795b18a117e6b3c9ad38d582aa11db29e5`, with no `data/raw/train_FD001.txt`, the service returned five predictions and zero failures:

| Model | Version | Raw RUL (cycles) |
| --- | --- | ---: |
| Decision Tree | `decision-tree-rul-v1` | 29.7941176471 |
| Random Forest | `random-forest-rul-v1` | 28.0240895095 |
| XGBoost | `xgboost-rul-v1` | 26.1461582184 |
| Constant mean | `constant-rul-v1` | 107.8078619553 |
| Dense MLP | `dense-rul-mlp-final-v1` | 17.2417659760 |

This is one smoke example from simulated FD001/train history. It is not accuracy, official-test, or maintenance evidence.

The older P12 clean-process audit script, `scripts/smoke_p12_final_bundles_clean_process.py`, failed in that same clean worktree because it explicitly reconstructs its test input from the absent `data/raw/train_FD001.txt`. That script is not the deployed inference path. The P23-A1 smoke above uses the packaged demo input and the real final-comparison service. The distinction is recorded so the P12 audit failure is not hidden or mistaken for an inference failure.

## Remote verification still pending

P23-T1 #98 remains open. An anonymous request to `https://fleetpulseai.streamlit.app/` still returns HTTP 303 to Streamlit sign-in, and the deployed commit/build log are not accessible from this environment. This local package audit cannot establish which bundle inventory the remote app serves. P23-A1's independent hosted check must use the exact deployed commit and an accessible public app before #96 can close.

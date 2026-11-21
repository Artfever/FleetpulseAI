# P12-T1: Development and final model registry

`fleetpulse.models.registry` exposes three read-only registries:

- `DEVELOPMENT_MODEL_REGISTRY` (also retained as `MODEL_REGISTRY` for current predictor compatibility) lists the four development candidates supported by the shared inference service, including the 30-cycle/96-feature input contract.
- `FINAL_MODEL_REGISTRY` lists the three final tree estimators, selected constant-mean baseline, and final MLP with their artifact versions, manifest paths, role, and 30-cycle/96-feature `past-only-v1` contract.
- `MODEL_REGISTRIES` groups the two registries under the explicit `development` and `final` stages. A cache or deployment inventory must retain the stage along with model name/version and feature config; a shared model name/version does not erase the stage distinction.

The current RUL page selector remains on the development inference path. Its displayed option identifies the development stage and exact registered version, and the page reports that version before a prediction. It also states the matching final-refit version and that this page does not load that final bundle. Prediction results continue to show the version read from the verified development artifact manifest.

The final manifest paths refer to ignored local artifacts; they are not checked into Git and are not available in the hosted app by virtue of this registry metadata. Local inventory checked for this task: the four tree bundle manifests/index and the final MLP manifest/weights are present under `artifacts/final/`. Tree bundle versions are `decision-tree-rul-v1`, `random-forest-rul-v1`, `xgboost-rul-v1`, and `constant-rul-v1`; the MLP is `dense-rul-mlp-final-v1`. The MLP manifest records the full-training fit (100 engines, 20,631 rows, 96 features, seed 42, eight epochs). No official test data or labels were inspected.

This registry documents artifact identity and routing metadata. It does not load or train a final model in the app; final bundle loading uses each versioned bundle's checksum-verifying loader. Task PRs to `dev` can add final bundle selection only after an explicit inference interface exists and deployment artifacts are provisioned.

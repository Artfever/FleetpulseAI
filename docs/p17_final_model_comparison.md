# P17-A1: compare frozen models on one observed history

Implementation status: code and synthetic checks available; actual frozen P12
bundle verification is **pending**. Issue #78 must remain open until that
output and required human inspection are recorded. Abdullah authorized takeover
of all remaining team assignments on 18 November 2026; earlier authorship and
existing GitHub assignees are not rewritten.

## Interface

```python
from fleetpulse.services import compare_final_models

comparison = compare_final_models(
    observed_prefix,
    dataset_partition="FD001/train",
    unit_id=4,
    artifact_root=project_root,
)
```

Input is one engine's observed prefix in the existing normalized 26-column raw
schema, with no RUL targets or hidden future rows. All records are validated
before bundle access. The cutoff is the last observed cycle. The service owns
normalized copies and passes exactly the same prefix to all five entries in
`FINAL_MODEL_REGISTRY`. It never reads a development candidate or fits a model.

`FinalHistoryComparison` contains dataset partition, unit ID, observed cycle,
a canonical input-history SHA, predictions, and failures. Each successful
`FinalModelPrediction` contains model name/version, final stage, raw unclipped
RUL in cycles, manifest payload SHA, weight SHA (None for the constant), feature
config SHA, training-source and selection hashes, and the exact
`EngineFeatureVector` used. The latter carries ordered feature values, config,
engine/cycle identity and actual input-window rows for subsequent explanation
consistency checks. These fields bind an output to its actual inputs/artifact;
they do not establish model quality or deployment.

The P12 tree loader and `predict_final_tree_bundle` handle Decision Tree,
Random Forest, XGBoost and the constant baseline. The P12 final MLP loader and
`predict_final_mlp` apply the saved scaler and frozen CPU weights. Each model
uses its own saved feature config/order through the existing shared extractor.
Registered identity/version/role/window/schema must match. Optional estimator
imports are deferred until a manifest is present.

Missing files, missing runtime dependencies and incompatible bundles produce
named failures with reasons, not numeric stand-ins. All-failed comparison has
an empty prediction tuple. Input/feature-overflow errors remain input errors.
There is no training fallback or cache in this component. Trusted project
bundles are required; checksums are integrity checks, not authenticity proofs.

## Synthetic verification

`tests/test_final_comparison.py` uses schema-only artificial manifests and a
tiny generated Decision Tree. Its declared training metadata is fixture data;
it does not represent a NASA-trained candidate. A toy observed prefix returned
raw **-2.5 cycles**, preserved without display clipping. Changing its visible
readings changed its prediction to **17.5 cycles** and its history hash.

Six invalid-history cases fail before model access. Missing/corrupt/version-
incompatible/development-role/reordered-schema bundles fail explicitly. A
five-model adapter check proves the same normalized history is used for every
entry. An MLP adapter double with an actual toy StandardScaler verifies that
raw features are delegated once to saved preprocessing; its **-4.75** output is
an adapter test, not native trained-MLP evidence. Existing native P12 MLP tests
run with the optional PyTorch extra. Numeric feature overflow is rejected.

On 18 November Abdullah accepted raw unclipped comparison and a downstream
max(raw)-min(raw) spread, labelled model disagreement in cycles with failed
models excluded and listed separately. Spread is not a probability/confidence
interval. Spread calculation belongs to #155; explanation identity #214 and
SHAP/UI tasks remain separate.

## Actual frozen-bundle execution (18 November 2026)

The previously unavailable P12 bundles were present locally under
`artifacts/final/` and passed the existing checksum, identity, version,
preprocessing and feature-schema loaders. The evidence script
`scripts/check_p17_a1_final_model_comparison.py` ran all five registered final
models on the same three packaged `FD001/train` prefixes. No model was fitted,
and the parsed 26-column raw histories contained no true-RUL labels or future
rows. Each model received identical feature values for a given history; the
same 96-feature schema and configuration were confirmed across the five
bundles. The MLP applied its saved scaler and frozen weights.

Command:

```powershell
.venv/Scripts/python.exe scripts/check_p17_a1_final_model_comparison.py
```

Result: `PASS: compared 5 final models on 3 histories`. Input file SHA-256 is
`953489b722a33815ee08f360c1a677d314f318b7d74d8bef88cd31dde0a8f9a7`; it
contains 229 packaged rows. The detailed machine-readable record is written to
the ignored `reports/P17_A1_final_model_comparison.json` and includes each
history, manifest, weights, feature-config and source/selection digest.

| FD001/train engine | Cutoff | Decision Tree | Random Forest | XGBoost | Constant mean | Dense MLP |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 164 | 29.794118 | 28.024090 | 26.146158 | 107.807862 | 17.241766 |
| 5 | 58 | 167.714286 | 206.979684 | 193.762833 | 107.807862 | 163.962723 |
| 12 | 7 | 185.114754 | 162.971080 | 162.781387 | 107.807862 | 163.483688 |

Frozen model versions and recorded training configurations from the verified
manifests:

| Model | Bundle version | Frozen configuration | Seed |
| --- | --- | --- | ---: |
| Decision Tree | `decision-tree-rul-v1` | `max_depth=8`, `min_samples_leaf=5` | 42 |
| Random Forest | `random-forest-rul-v1` | 100 trees, `max_depth=16`, `min_samples_leaf=5`, `max_features=1.0`, bootstrap, `n_jobs=1` | 42 |
| XGBoost | `xgboost-rul-v1` | 100 trees, `max_depth=4`, learning rate 0.05, subsample/column sample 0.8, CPU histogram method, `n_jobs=1` | 42 |
| Constant mean baseline | `constant-rul-v1` | training-label mean | None |
| Dense MLP | `dense-rul-mlp-final-v1` | 96-128-64-1, ReLU, CPU, saved scaler | 42 |

All outputs are raw, unclipped RUL estimates in cycles. Shared provenance:
training-source SHA-256 `963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8`,
selection-manifest SHA-256
`4cbda49f05eb928ebacc0a673de175dfae24b0017f7798db62a37cda8462fa2d`, and
feature-config SHA-256
`19dd0331ee3e0c3a90217446a4481507b7da94fe1424b1c251c4605a6b168dfe`.
Per-prefix history hashes and per-model manifest/weight hashes are in the JSON
record. To retain provenance even when the ignored JSON report is unavailable,
the observed digests are:

| Bundle | Manifest payload SHA-256 | Weights SHA-256 |
| --- | --- | --- |
| Decision Tree | `ed434d11ce7ae22cd6445aa18fc4fb7e5db5f3438dded681b596ef54d6d37e89` | `c33021cb9cf5931aba545b6a327d0547f14be1363cbd0c24d57f0ad923dd7bb3` |
| Random Forest | `f52ff4dbcb9c3876ff045bebfaef791652d996a151e987815cf37a6afed69b1a` | `712a8313271565091fb3a7c9b4fce5e5d8bf2caf64b77e8da9c942a477605d81` |
| XGBoost | `e40f4a9142caf5dc6d8c4880a2b0e526ef0e9e097e472ab70d44aa79169c2e11` | `719a5553ba938fb40dd4196d081d7d07f6c7490be0dbbd4d0cf653e776a4816e` |
| Constant mean | `dff249bdfef9426b731a4d8fb8f74bd298571d38e86eb277f0a2a8b27d724c96` | None (no estimator weights) |
| Dense MLP | `7a3e56ff80df0ba194067956842dab1e9d4148f85f5dc1ad9378bca0b234549c` | `fcbd5a67ca8bf307f049c7131fee08ee484f8f35e6cb4e8966aa3c45839b4650` |

History SHA-256 values are engine 4: `3e1219d1c416c7c8c8bc770e86515aaa4f0c263ed3ab0ab987e1b7b275c853b8`,
engine 5: `56c88f47d4b18b476213258e891864646675b1c46638b881c13d028555a19aad`,
and engine 12: `08afebc54e155f306a8891adc2b1d97d485f6a5283dc2ea4153bd5edd0733f1d`.
The ignored JSON report SHA-256 from this run is
`9aca5e6df92cac25ba99ba1f3fe1b2033fa2e079c4aa87d97062b6964f697d1c`.
These are point outputs on three small packaged histories, not an
accuracy comparison, confidence estimates or maintenance recommendations.
The frozen bundle files are ignored local artifacts and are not in Git; a clean
checkout needs the trusted bundle directory to reproduce this run.

The named teammate inspection of the actual output is still pending. Keep #78
open until that check is recorded. Local Streamlit smoke is NOT RUN because no
application code changed.

## Executed checks (18 November)

- `.venv/Scripts/python.exe -m ruff check .`: PASS after narrowing an exception handler.
- `.venv/Scripts/python.exe -m pytest -q tests/test_final_comparison.py`: 17 passed in9.83s.
- `.venv/Scripts/python.exe -m pytest -q`: initial 3 collection errors because torch was missing; after installing pinned CPU extras, **531 passed, 1 skipped in99.14s**.
- `.venv/Scripts/python.exe -m pip check`: PASS; torch2.15.0+cpu tensor smoke PASS.
- `git diff --check`: PASS. Local Streamlit launch NOT RUN:noappchange.
- Existing comparison service implementation is in [PR #446](https://github.com/Artfever/FleetpulseAI/pull/446) to dev; all five required CI checks passed. Actual-bundle execution is documented above and awaits human output inspection before #78 closes.

Follow-up verification in the workspace containing the frozen bundles:

- `.venv/Scripts/python.exe scripts/check_p17_a1_final_model_comparison.py`: PASS; 5 final models x 3 histories, 0 bundle failures, and equal feature values/config per history.
- `.venv/Scripts/python.exe -m pytest -q tests/test_final_comparison.py`: PASS (17 passed in 2.35s).
- `.venv/Scripts/python.exe -m pytest -q`: PASS (662 passed in 72.97s).
- `.venv/Scripts/python.exe -m ruff check .`: PASS. `git diff --check`: PASS.
- Streamlit smoke NOT RUN: no app code changed.

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
SHAP/UI tasks remain separate. No new human inspection of actual P12 comparison
outputs is claimed while those bundles are missing.

## Missing real input

No P12 final manifests were found by name under
`C:/Users/hp/Documents/FA26/AI`; `artifacts/final/` contains only `.gitkeep` in
this clone. The tracked artifact inventory describes ignored outputs in the
producer's workspace, not files supplied to every clone. Abdullah is asking
Shaheer for the existing bundle directory/archive. Reuse those frozen bytes
and check against `docs/artifact_inventory.md`; do not refit substitute models
or present recorded smoke numbers as a new execution.

Expected paths are the four tree/constant manifests and three tree weight
files under `artifacts/final/tree/bundles/`, plus the final MLP manifest and
weights under `artifacts/final/mlp/bundle/`. The index/selection/run metadata
should travel with the bundles for provenance audits.

When supplied, run one same-history comparison over all five models, verify
artifact hashes and source/config identity, inspect raw outputs with the owner,
then finish #78 acceptance. Local Streamlit smoke is NOT RUN: no app change.
Exact local/CI commands and results are recorded in the PR and owner handoff.

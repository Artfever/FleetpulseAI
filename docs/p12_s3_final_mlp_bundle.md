# P12-S3: final MLP bundle

P12-S3 packages the P12-S1 scaler and P12-S2 fixed-epoch checkpoint into a
versioned final-training-model bundle under `artifacts/final/mlp/bundle/`.
That location is separate from the development validation candidates and
ignored by Git along with the source data and trained weights.

The bundle contains `final_mlp_state.pt` and `final_mlp_manifest.json`. The
manifest hashes the weights and records model version, `(128, 64)` ReLU
architecture, ordered 96-feature `past-only-v1` schema, scaler statistics,
source and P11 decision hashes, 100 training-engine row counts, seed 42, and
the frozen eight-epoch CPU/Adam training configuration. The loader rejects
wrong hashes, feature order, model dimensions, source/selection lineage,
scaler lineage, or training-rule metadata. It reloads weights with
`weights_only=True` and keeps the model on CPU in evaluation mode.

## Reproduce

With the original FD001/train file, P11 selection manifest, and P12-S1/S2
artifacts available locally:

```powershell
.venv/Scripts/python.exe scripts/package_p12_final_mlp.py
.venv/Scripts/python.exe -m pytest tests/test_final_mlp_bundle.py -q
```

The script packages and reloads the bundle, verifies that scaler mean/scale/
variance are exactly preserved, and prints a prediction for FD001/train engine
4 at cycle 164. That training-row output is a serialization and inference
sanity check, not an evaluation score or performance estimate. Fresh-process
verification of each final model is P12-T2.
